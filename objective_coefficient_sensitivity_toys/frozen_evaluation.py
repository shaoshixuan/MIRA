from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "experiment_config.json"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PROTOCOL = ROOT / "protocol"
PER_USER = RESULTS / "per_user"
for directory in (RESULTS, FIGURES, PROTOCOL, PER_USER):
    directory.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as error:
            last_error = error
    raise ValueError(f"Cannot decode {path}: {last_error}")


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    required = {
        "user_id", "item_id", "is_positive", "original_pctr_score",
        "kg_interest_score", "llm_exploration_score", "explore_score_source",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing score columns: {sorted(missing)}")
    df["user_id"] = df["user_id"].astype(str).str.strip()
    df["item_id"] = df["item_id"].astype(str).str.strip()
    for column in (
        "is_positive", "original_pctr_score", "kg_interest_score",
        "llm_exploration_score",
    ):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["is_positive"] = df["is_positive"].fillna(0).astype(int)
    df["category"] = df.get("category", "").fillna("").astype(str)
    df["explore_score_source"] = df["explore_score_source"].fillna("").astype(str)
    if "llm_scoring_method" not in df:
        df["llm_scoring_method"] = ""
    else:
        df["llm_scoring_method"] = df["llm_scoring_method"].fillna("").astype(str)
    if df.duplicated(["user_id", "item_id"]).any():
        raise ValueError("Duplicate user/item score keys")
    return df


def percentile(values: pd.Series, groups: pd.Series, valid: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=values.index, dtype=float)
    frame = pd.DataFrame({"value": values[valid], "group": groups[valid]})
    if frame.empty:
        return out
    rank = frame.groupby("group", sort=False)["value"].rank(method="min")
    size = frame.groupby("group", sort=False)["value"].transform("size")
    out.loc[valid] = ((rank - 1) / (size - 1).replace(0, np.nan)).fillna(0.5).clip(0, 1)
    return out


def calibrate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pctr"] = (
        df.groupby("user_id", sort=False)["original_pctr_score"]
        .rank(method="average", pct=True).clip(0, 1)
    )
    valid_kg = df["kg_interest_score"].notna() & ~df["kg_interest_score"].isin([-10, -5])
    valid_llm = df["llm_exploration_score"].notna() & ~df["llm_scoring_method"].eq("LLM_failed")
    descriptions = df.get("description", pd.Series("", index=df.index)).fillna("").astype(str)
    valid_llm &= ~(df["llm_exploration_score"].fillna(0).eq(0) & descriptions.eq(""))
    df["kg"] = percentile(df["kg_interest_score"], pd.Series("KG", index=df.index), valid_kg)
    df["llm"] = percentile(df["llm_exploration_score"], df["explore_score_source"], valid_llm)
    df["kg_valid"] = valid_kg
    df["llm_valid"] = valid_llm

    source_flags = pd.DataFrame({
        "user_id": df["user_id"],
        "has_kg": df["explore_score_source"].eq("KG"),
        "has_low": df["explore_score_source"].eq("LLM_low_interaction"),
        "has_enhancement": df["explore_score_source"].eq("LLM_enhancement"),
    })
    flags = source_flags.groupby("user_id", sort=False).any()
    route = pd.Series("KG_ONLY", index=flags.index, dtype=object)
    route.loc[flags.has_enhancement] = "MIXED_ENHANCEMENT"
    route.loc[flags.has_low & ~flags.has_enhancement & ~flags.has_kg] = "LOW_INTERACTION"
    df["route"] = df["user_id"].map(route)
    return df


def score_variant(df: pd.DataFrame, variant: str, route_weights: Dict[str, float]) -> np.ndarray:
    pctr = df["pctr"].to_numpy(float)
    kg = df["kg"].to_numpy(float)
    llm = df["llm"].to_numpy(float)
    kg_valid = df["kg_valid"].to_numpy(bool)
    llm_valid = df["llm_valid"].to_numpy(bool)

    if variant == "A0_pCTR_only":
        return pctr
    if variant == "A1_pCTR_KG":
        explore, valid = np.nan_to_num(kg, nan=0.0), kg_valid
        return np.where(valid, 0.7 * pctr + 0.3 * explore, pctr)
    if variant == "A2_pCTR_LLM":
        explore, valid = np.nan_to_num(llm, nan=0.0), llm_valid
        return np.where(valid, 0.3 * pctr + 0.7 * explore, pctr)

    both = kg_valid & llm_valid
    only_kg = kg_valid & ~llm_valid
    only_llm = llm_valid & ~kg_valid
    explore = np.zeros(len(df), dtype=float)
    explore[both] = 0.5 * kg[both] + 0.5 * llm[both]
    explore[only_kg] = kg[only_kg]
    explore[only_llm] = llm[only_llm]
    valid = kg_valid | llm_valid

    if variant == "A3_simple_KG_LLM":
        return np.where(valid, 0.5 * pctr + 0.5 * explore, pctr)
    if variant == "A4_full_route_aware":
        weight = df["route"].map(route_weights).to_numpy(float)
        return np.where(valid, (1 - weight) * pctr + weight * explore, pctr)
    raise ValueError(variant)


def lambda_score(df: pd.DataFrame, lam: float) -> np.ndarray:
    kg, llm = df["kg"].to_numpy(float), df["llm"].to_numpy(float)
    kg_valid = df["kg_valid"].to_numpy(bool)
    llm_valid = df["llm_valid"].to_numpy(bool)
    both = kg_valid & llm_valid
    explore = np.zeros(len(df), dtype=float)
    explore[both] = 0.5 * kg[both] + 0.5 * llm[both]
    explore[kg_valid & ~llm_valid] = kg[kg_valid & ~llm_valid]
    explore[llm_valid & ~kg_valid] = llm[llm_valid & ~kg_valid]
    valid = kg_valid | llm_valid
    pctr = df["pctr"].to_numpy(float)
    return np.where(valid, (1 - lam) * pctr + lam * explore, pctr)


def stratified_user_split(
    df: pd.DataFrame, tune_ratio: float, seed: int
) -> Tuple[set[str], set[str]]:
    """Split users within each route so every dataset has its own tuning users."""
    user_routes = df[["user_id", "route"]].drop_duplicates("user_id")
    rng = np.random.default_rng(seed)
    tune_users: set[str] = set()
    holdout_users: set[str] = set()
    for _, group in user_routes.groupby("route", sort=True):
        users = group["user_id"].astype(str).to_numpy(copy=True)
        rng.shuffle(users)
        if len(users) <= 1:
            cut = len(users)
        else:
            cut = min(len(users) - 1, max(1, int(round(len(users) * tune_ratio))))
        tune_users.update(users[:cut].tolist())
        holdout_users.update(users[cut:].tolist())
    return tune_users, holdout_users


def objective_value(metric_row: pd.Series, objective: Dict[str, float]) -> float:
    return (
        float(objective["ndcg"]) * float(metric_row["ndcg"])
        + float(objective["recall"]) * float(metric_row["recall"])
    )


def tune_dataset_route_weights(
    scores: pd.DataFrame,
    popularity: pd.Series,
    config: dict,
    dataset: str,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, set[str], set[str]]:
    """Tune one independent exploration weight per route for one dataset."""
    tune_users, holdout_users = stratified_user_split(
        scores, config["weight_search_tune_ratio"], config["seed"]
    )
    tune = scores[scores["user_id"].isin(tune_users)].copy()
    holdout = scores[scores["user_id"].isin(holdout_users)].copy()
    routes = ["KG_ONLY", "MIXED_ENHANCEMENT", "LOW_INTERACTION"]
    selected: Dict[str, float] = {}
    search_rows: List[dict] = []

    for route in routes:
        subset = tune[tune["route"].eq(route)].copy()
        if subset.empty:
            selected[route] = 0.0
            search_rows.append({
                "dataset": dataset, "route": route, "weight": 0.0,
                "tune_users": 0, "k": config["weight_search_k"],
                "objective": np.nan, "recall": np.nan, "ndcg": np.nan,
                "category_coverage": np.nan, "ild": np.nan, "novelty": np.nan,
                "selected": True,
            })
            continue

        candidates = []
        for weight in config["weight_grid"]:
            weights = {name: 0.0 for name in routes}
            weights[route] = float(weight)
            metrics, _ = evaluate(
                subset,
                score_variant(subset, "A4_full_route_aware", weights),
                [config["weight_search_k"]],
                popularity,
            )
            metric = metrics.iloc[0]
            objective = objective_value(metric, config["objective"])
            row = {
                "dataset": dataset, "route": route, "weight": float(weight),
                "tune_users": int(metric["users"]), "k": int(metric["k"]),
                "objective": objective, "recall": float(metric["recall"]),
                "ndcg": float(metric["ndcg"]),
                "category_coverage": float(metric["category_coverage"]),
                "ild": float(metric["ild"]), "novelty": float(metric["novelty"]),
                "selected": False,
            }
            search_rows.append(row)
            candidates.append((objective, float(metric["ndcg"]), float(metric["recall"]), -float(weight), float(weight)))
        selected[route] = max(candidates)[-1]
        for row in search_rows:
            if row["dataset"] == dataset and row["route"] == route and row["weight"] == selected[route]:
                row["selected"] = True

    holdout_rows: List[dict] = []
    scopes = {"all_holdout_users": holdout}
    for route in routes:
        scopes[f"holdout_{route}"] = holdout[holdout["route"].eq(route)].copy()
    for scope, subset in scopes.items():
        if subset.empty:
            continue
        metrics, _ = evaluate(
            subset,
            score_variant(subset, "A4_full_route_aware", selected),
            config["ks"],
            popularity,
        )
        metrics.insert(0, "scope", scope)
        metrics.insert(0, "dataset", dataset)
        holdout_rows.extend(metrics.to_dict("records"))
    return selected, pd.DataFrame(search_rows), pd.DataFrame(holdout_rows), tune_users, holdout_users


def parse_category(value: str) -> str:
    return "|".join(sorted(part.strip() for part in str(value).split("|") if part.strip()))


def load_popularity(history_path: Path) -> pd.Series:
    history = pd.read_csv(history_path, sep="\t", header=None, usecols=[1], names=["item_id"])
    history["item_id"] = history["item_id"].astype(str).str.strip()
    counts = history["item_id"].value_counts()
    return counts / counts.sum()


def evaluate(df: pd.DataFrame, score: np.ndarray, ks: Iterable[int], popularity: pd.Series) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
    work = df[["user_id", "item_id", "is_positive", "category"]].copy()
    work["score"] = score
    work = work.sort_values(["user_id", "score", "item_id"], ascending=[True, False, True], kind="mergesort")
    work["rank"] = work.groupby("user_id", sort=False).cumcount() + 1
    work["pos_total"] = work.groupby("user_id", sort=False)["is_positive"].transform("sum")
    total_categories = set()
    for value in work["category"]:
        total_categories.update(part.strip() for part in value.split("|") if part.strip())
    metric_rows, per_user_outputs = [], {}

    for k in ks:
        top = work[work["rank"] <= k].copy()
        top["discounted_gain"] = top["is_positive"] / np.log2(top["rank"] + 1)
        user = top.groupby("user_id", sort=False).agg(
            hits=("is_positive", "sum"), dcg=("discounted_gain", "sum")
        )
        totals = work.groupby("user_id", sort=False)["pos_total"].first()
        user = user.reindex(totals.index, fill_value=0)
        user["recall"] = user["hits"] / totals.clip(lower=1)
        ideal = totals.map(lambda n: sum(1 / math.log2(position + 1) for position in range(1, min(int(n), k) + 1)))
        user["ndcg"] = user["dcg"] / ideal.replace(0, np.nan)

        pop = top["item_id"].map(popularity)
        top["novelty"] = -np.log(pop.fillna(1e-10).clip(lower=1e-10))
        user["novelty"] = top.groupby("user_id", sort=False)["novelty"].mean().reindex(user.index)

        top["cat_key"] = top["category"].map(parse_category)
        def ild(group: pd.Series) -> float:
            values = [value for value in group if value]
            if len(values) < 2:
                return 0.0
            unequal = sum(values[i] != values[j] for i in range(len(values)) for j in range(i + 1, len(values)))
            return unequal / (len(values) * (len(values) - 1) / 2)
        user["ild"] = top.groupby("user_id", sort=False)["cat_key"].apply(ild).reindex(user.index).fillna(0)
        per_user_outputs[k] = user.reset_index()[["user_id", "recall", "ndcg", "ild", "novelty"]]

        covered = set()
        for value in top["category"]:
            covered.update(part.strip() for part in value.split("|") if part.strip())
        coverage = len(covered) / len(total_categories) if total_categories else 0.0
        metric_rows.append({
            "k": k, "users": len(user), "recall": user["recall"].mean(),
            "ndcg": user["ndcg"].mean(), "category_coverage": coverage,
            "ild": user["ild"].mean(), "novelty": user["novelty"].mean(),
        })
    return pd.DataFrame(metric_rows), per_user_outputs


def recommendations_to_user_metrics(df: pd.DataFrame, candidates: pd.DataFrame, k: int, popularity: pd.Series, users: set[str] | None = None) -> pd.DataFrame:
    rec = df[df["k"] == k].copy()
    rec["user_id"] = rec["user_id"].astype(str).str.strip()
    rec["item_id"] = rec["item_id"].astype(str).str.strip()
    universe = candidates if users is None else candidates[candidates["user_id"].isin(users)]
    rec = rec[rec["user_id"].isin(set(universe["user_id"]))]
    labels = universe[["user_id", "item_id", "is_positive", "category"]]
    merged = rec.merge(labels, on=["user_id", "item_id"], how="left", validate="many_to_one")
    merged = merged.dropna(subset=["is_positive"])
    merged["rank"] = pd.to_numeric(merged["rank"], errors="coerce")
    merged["gain"] = merged["is_positive"] / np.log2(merged["rank"] + 1)
    totals = universe.groupby("user_id", sort=False)["is_positive"].sum()
    out = pd.DataFrame(index=totals.index)
    out["recall"] = merged.groupby("user_id")["is_positive"].sum().reindex(out.index, fill_value=0) / totals.clip(lower=1)
    ideal = totals.map(lambda n: sum(1 / math.log2(position + 1) for position in range(1, min(int(n), k) + 1)))
    out["ndcg"] = merged.groupby("user_id")["gain"].sum().reindex(out.index, fill_value=0) / ideal.replace(0, np.nan)
    merged["novelty"] = -np.log(merged["item_id"].map(popularity).fillna(1e-10).clip(lower=1e-10))
    out["novelty"] = merged.groupby("user_id")["novelty"].mean().reindex(out.index)
    merged["cat_key"] = merged["category"].fillna("").map(parse_category)
    def rec_ild(group: pd.Series) -> float:
        values = [value for value in group if value]
        if len(values) < 2:
            return 0.0
        unequal = sum(values[i] != values[j] for i in range(len(values)) for j in range(i + 1, len(values)))
        return unequal / (len(values) * (len(values) - 1) / 2)
    out["ild"] = merged.groupby("user_id")["cat_key"].apply(rec_ild).reindex(out.index).fillna(0)
    return out.reset_index()[["user_id", "recall", "ndcg", "ild", "novelty"]]


def paired_bootstrap(ours: pd.DataFrame, baseline: pd.DataFrame, samples: int, seed: int) -> List[dict]:
    merged = ours.merge(baseline, on="user_id", suffixes=("_ours", "_baseline"), validate="one_to_one")
    rng = np.random.default_rng(seed)
    output = []
    for metric in ("recall", "ndcg", "novelty"):
        delta = (merged[f"{metric}_ours"] - merged[f"{metric}_baseline"]).to_numpy(float)
        delta = delta[np.isfinite(delta)]
        boot = np.empty(samples, dtype=float)
        chunk = 250
        for start in range(0, samples, chunk):
            count = min(chunk, samples - start)
            indices = rng.integers(0, len(delta), size=(count, len(delta)))
            boot[start:start + count] = delta[indices].mean(axis=1)
        lower, upper = np.quantile(boot, [0.025, 0.975])
        p_value = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
        output.append({
            "metric": metric, "users": len(delta), "ours": merged[f"{metric}_ours"].mean(),
            "baseline": merged[f"{metric}_baseline"].mean(), "difference": delta.mean(),
            "ci_lower": lower, "ci_upper": upper, "p_value": p_value,
        })
    return output


def holm_adjust(p_values: pd.Series) -> pd.Series:
    order = np.argsort(p_values.to_numpy(float))
    sorted_p = p_values.to_numpy(float)[order]
    adjusted = np.maximum.accumulate((len(sorted_p) - np.arange(len(sorted_p))) * sorted_p)
    adjusted = np.clip(adjusted, 0, 1)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return pd.Series(result, index=p_values.index)


def plot_sensitivity(frame: pd.DataFrame) -> None:
    for (dataset, population), subset in frame.groupby(["dataset", "population"]):
        primary = subset[subset["k"] == 10].sort_values("lambda")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        axes[0].plot(primary["lambda"], primary["recall"], marker="o", label="Recall")
        axes[0].plot(primary["lambda"], primary["ndcg"], marker="s", label="NDCG")
        axes[0].set(xlabel="Exploration weight λ", ylabel="Accuracy", title=f"{dataset}: accuracy sensitivity")
        axes[0].set_title(f"{dataset} / {population}: accuracy")
        axes[0].grid(alpha=0.3); axes[0].legend()
        axes[1].plot(primary["ndcg"], primary["novelty"], marker="o")
        for _, row in primary.iterrows():
            axes[1].annotate(f"{row['lambda']:.1f}", (row["ndcg"], row["novelty"]), fontsize=8)
        axes[1].set(xlabel="NDCG@10", ylabel="Novelty@10", title=f"{dataset}: NDCG–Novelty Pareto")
        axes[1].set_title("NDCG-Novelty Pareto")
        axes[1].grid(alpha=0.3)
        axes[2].plot(primary["ndcg"], primary["category_coverage"], marker="o")
        for _, row in primary.iterrows():
            axes[2].annotate(f"{row['lambda']:.1f}", (row["ndcg"], row["category_coverage"]), fontsize=8)
        axes[2].set(xlabel="NDCG@10", ylabel="Category Coverage@10", title="NDCG-Coverage Pareto")
        axes[2].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES / f"{dataset}_{population}_lambda_sensitivity.png", dpi=220)
        plt.close(fig)


def plot_sensitivity_svg(frame: pd.DataFrame) -> None:
    """Write one route-specific parameter curve for every dataset and scope."""
    for (dataset, population, varied_route), subset in frame.groupby(
        ["dataset", "population", "varied_route"]
    ):
        primary = subset[subset["k"] == 10].sort_values("lambda")
        panels = [("lambda", "ndcg", "lambda", "NDCG@10", "Accuracy sensitivity"),
                  ("ndcg", "novelty", "NDCG@10", "Novelty@10", "NDCG-Novelty Pareto"),
                  ("ndcg", "category_coverage", "NDCG@10", "Coverage@10", "NDCG-Coverage Pareto")]
        svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="430">',
               '<rect width="100%" height="100%" fill="white"/>',
               f'<text x="20" y="28" font-family="Arial" font-size="18" font-weight="bold">{dataset} / {population} / vary {varied_route}</text>']
        for index, (xcol, ycol, xlabel, ylabel, title) in enumerate(panels):
            left, top, plot_w, plot_h = 55 + index * 500, 70, 390, 280
            x = primary[xcol].to_numpy(float); y = primary[ycol].to_numpy(float)
            xmin, xmax = float(x.min()), float(x.max()); ymin, ymax = float(y.min()), float(y.max())
            if xmax == xmin: xmax = xmin + 1
            if ymax == ymin: ymax = ymin + 1
            sx = lambda value: left + (value - xmin) / (xmax - xmin) * plot_w
            sy = lambda value: top + plot_h - (value - ymin) / (ymax - ymin) * plot_h
            points = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(x, y))
            svg += [f'<text x="{left+plot_w/2}" y="55" text-anchor="middle" font-family="Arial" font-size="15">{title}</text>',
                    f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#555"/>',
                    f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#555"/>',
                    f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2"/>']
            for a, b, lam in zip(x, y, primary["lambda"]):
                svg += [f'<circle cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="4" fill="#dc2626"/>',
                        f'<text x="{sx(a)+5:.1f}" y="{sy(b)-5:.1f}" font-family="Arial" font-size="10">{lam:.1f}</text>']
            svg += [f'<text x="{left+plot_w/2}" y="{top+plot_h+40}" text-anchor="middle" font-family="Arial" font-size="12">{xlabel}</text>',
                    f'<text x="{left-35}" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 {left-35} {top+plot_h/2})" font-family="Arial" font-size="12">{ylabel}</text>']
        svg.append('</svg>')
        (FIGURES / f"{dataset}_{population}_{varied_route}_weight_sensitivity.svg").write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    variants = ["A0_pCTR_only", "A1_pCTR_KG", "A2_pCTR_LLM", "A3_simple_KG_LLM", "A4_full_route_aware"]
    protocol = {"config": config, "sources": {}, "checks": {}, "notes": [
        "All variants use identical full-domain candidate rows and labels.",
        "Popularity is computed from training histories only.",
        "A3 is equal-weight KG/LLM exploration with fixed 0.5 total exploration weight.",
        "A4 uses route-specific weights tuned independently inside each dataset.",
        "Weights maximize 0.6*NDCG@15+0.4*Recall@15 on an 80% route-stratified tuning-user split.",
        "The selected weights are then used to regenerate both all-user and target-subgroup experiment versions.",
        "Parameter curves vary one route weight at a time while holding the other dataset-specific route weights fixed.",
        "Full-population and route-defined subgroup results are reported separately.",
        "Automotive/Toys subgroup is LLM_enhancement users; Grocery has no enhancement route and uses low-interaction users.",
        "Legacy spreadsheet metrics are not mixed into recomputed comparisons."
    ]}
    ablation_all, sensitivity_all = [], []
    search_all, holdout_all = [], []
    selected_by_dataset: Dict[str, Dict[str, float]] = {}

    for dataset, paths in config["datasets"].items():
        score_path, history_path = Path(paths["scores"]), Path(paths["history"])
        protocol["sources"][dataset] = {
            "scores": str(score_path), "scores_sha256": checksum(score_path),
            "history": str(history_path), "history_sha256": checksum(history_path),
        }
        scores = calibrate(normalize_scores(read_csv(score_path)))
        popularity = load_popularity(history_path)
        counts = scores.groupby("user_id").size()
        protocol["checks"][dataset] = {
            "rows": len(scores), "users": scores.user_id.nunique(), "items": scores.item_id.nunique(),
            "positive_rows": int(scores.is_positive.sum()), "duplicate_keys": int(scores.duplicated(["user_id", "item_id"]).sum()),
            "candidates_min": int(counts.min()), "candidates_max": int(counts.max()),
            "source_counts": scores.explore_score_source.value_counts().astype(int).to_dict(),
        }

        selected_weights, search_frame, holdout_frame, tune_users, holdout_users = tune_dataset_route_weights(
            scores, popularity, config, dataset
        )
        selected_by_dataset[dataset] = selected_weights
        search_all.append(search_frame)
        holdout_all.append(holdout_frame)
        protocol["checks"][dataset]["selected_route_weights"] = selected_weights
        protocol["checks"][dataset]["weight_tune_users"] = len(tune_users)
        protocol["checks"][dataset]["weight_holdout_users"] = len(holdout_users)
        protocol["checks"][dataset]["route_user_counts"] = (
            scores[["user_id", "route"]].drop_duplicates()["route"].value_counts().astype(int).to_dict()
        )

        populations = {
            "all_users": scores,
            paths["subgroup_label"]: scores[scores["route"].eq(paths["subgroup_route"])].copy(),
        }
        protocol["checks"][dataset]["subgroup_route"] = paths["subgroup_route"]
        protocol["checks"][dataset]["subgroup_users"] = int(populations[paths["subgroup_label"]].user_id.nunique())
        for population, population_scores in populations.items():
            if population_scores.empty:
                continue
            full_per_user = None
            for variant in variants:
                metric_frame, per_user = evaluate(
                    population_scores,
                    score_variant(population_scores, variant, selected_weights),
                    config["ks"],
                    popularity,
                )
                metric_frame.insert(0, "variant", variant); metric_frame.insert(0, "population", population); metric_frame.insert(0, "dataset", dataset)
                ablation_all.append(metric_frame)
                for k, user_frame in per_user.items():
                    user_frame.to_csv(PER_USER / f"{dataset}_{population}_{variant}_k{k}.csv", index=False)
                if variant == "A4_full_route_aware":
                    full_per_user = per_user[config["primary_k"]]

            routes_in_population = sorted(population_scores["route"].dropna().unique().tolist())
            for varied_route in routes_in_population:
                for lam in config["weight_grid"]:
                    curve_weights = selected_weights.copy()
                    curve_weights[varied_route] = float(lam)
                    metric_frame, _ = evaluate(
                        population_scores,
                        score_variant(population_scores, "A4_full_route_aware", curve_weights),
                        config["ks"],
                        popularity,
                    )
                    metric_frame.insert(0, "selected_weight", selected_weights[varied_route])
                    metric_frame.insert(0, "lambda", lam)
                    metric_frame.insert(0, "varied_route", varied_route)
                    metric_frame.insert(0, "population", population)
                    metric_frame.insert(0, "dataset", dataset)
                    sensitivity_all.append(metric_frame)

            population_users = set(population_scores["user_id"])
            for baseline_method, baseline_path_text in paths["baselines"].items():
                baseline_user = recommendations_to_user_metrics(read_csv(Path(baseline_path_text)), scores, config["primary_k"], popularity, population_users)
                baseline_user.to_csv(PER_USER / f"{dataset}_{population}_{baseline_method.replace('-', '_')}_k{config['primary_k']}.csv", index=False)

    ablation = pd.concat(ablation_all, ignore_index=True)
    sensitivity = pd.concat(sensitivity_all, ignore_index=True)
    search = pd.concat(search_all, ignore_index=True)
    holdout = pd.concat(holdout_all, ignore_index=True)
    ablation.to_csv(RESULTS / "core_ablation_metrics.csv", index=False)
    sensitivity.to_csv(RESULTS / "route_weight_sensitivity_metrics.csv", index=False)
    search.to_csv(RESULTS / "route_weight_search.csv", index=False)
    holdout.to_csv(RESULTS / "selected_weights_holdout_metrics.csv", index=False)
    (RESULTS / "selected_route_weights.json").write_text(
        json.dumps(selected_by_dataset, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_sensitivity_svg(sensitivity)
    (PROTOCOL / "frozen_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")

    primary = ablation[ablation.k == config["primary_k"]]
    primary.to_csv(RESULTS / "paper_core_ablation_k10.csv", index=False)
    summary = {
        "core_ablation_rows": len(ablation),
        "route_sensitivity_rows": len(sensitivity),
        "weight_search_rows": len(search),
        "holdout_metric_rows": len(holdout),
        "datasets": list(config["datasets"]),
        "primary_k": config["primary_k"],
        "selected_route_weights": selected_by_dataset,
    }
    (RESULTS / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
