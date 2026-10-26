"""Objective-coefficient sensitivity using the frozen experiment protocol.

The original validation grid contains metric values for every route/weight pair.
Changing the objective coefficient only changes how those saved validation rows
are ranked; candidate scoring and the validation split stay identical. This
script verifies the original inputs and baseline, then reranks holdout users
for every newly selected route-weight combination.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from frozen_evaluation import (
    calibrate, checksum, evaluate, load_popularity, normalize_scores,
    read_csv, score_variant, stratified_user_split,
)


ROOT = Path(__file__).resolve().parent
FROZEN = Path(r"D:\graduate_recsys\supplementary_experiments_dataset_adaptive_20260827")
DATASETS = {
    "automotive": (
        Path(r"D:\graduate_recsys\datasets_result_forevaluate\automtive_scores.csv"),
        Path(r"D:\graduate_recsys\data_deal_afterexplore_forevaluate\kg_train_triples_forjiaohu_auto.txt"),
    ),
    "toys": (
        Path(r"D:\graduate_recsys\datasets_result_forevaluate\toys_scores.csv"),
        Path(r"E:\Toys_and_Games-20260920T121442Z-1-001\Toys_and_Games\minimal\rec_train.txt"),
    ),
    "grocery": (
        Path(r"D:\graduate_recsys\datasets_result_forevaluate\grocery_scores.csv"),
        Path(r"E:\Grocery_and_Gourmet_Food-20260920T121440Z-1-001\Grocery_and_Gourmet_Food\minimal\rec_train.txt"),
    ),
}
ROUTES = ("KG_ONLY", "MIXED_ENHANCEMENT", "LOW_INTERACTION")
BETAS = (0.4, 0.5, 0.6, 0.7, 0.8)
KS = (5, 10, 15)


def ensure_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=0, atol=1e-12, equal_nan=True):
        raise AssertionError(f"{label}: got {actual}, expected {expected}")


def main(dataset: str) -> None:
    source, history = DATASETS[dataset]
    out = ROOT / f"{dataset}_coefficient_results"
    protocol = json.loads((FROZEN / "protocol" / "frozen_protocol.json").read_text(encoding="utf-8"))
    config = protocol["config"]
    expected = protocol["sources"][dataset]
    source_checksums = {}
    for name, path, digest in (("scores", source, expected["scores_sha256"]), ("history", history, expected["history_sha256"])):
        if not path.is_file():
            raise FileNotFoundError(f"Required input missing: {path}")
        actual = checksum(path)
        source_checksums[name] = actual
        if dataset != "grocery" and actual != digest:
            raise ValueError(f"Input changed; frozen comparison invalid: {path}\n{actual} != {digest}")

    scores = calibrate(normalize_scores(read_csv(source)))
    if dataset == "grocery":
        # In this revised file LOW_INTERACTION users can have candidate-level
        # KG fallback scores. Route inference from score_source alone would
        # incorrectly classify all such users as KG_ONLY.
        per_user_route = scores[["user_id", "applied_route"]].drop_duplicates()
        if per_user_route.user_id.duplicated().any() or not set(per_user_route.applied_route).issubset(ROUTES):
            raise ValueError("Grocery applied_route is inconsistent within users")
        scores["route"] = scores["applied_route"]
    popularity = load_popularity(history)
    tune_users, holdout_users = stratified_user_split(scores, config["weight_search_tune_ratio"], config["seed"])
    tune = scores[scores.user_id.isin(tune_users)].copy()
    holdout = scores[scores.user_id.isin(holdout_users)].copy()
    check = protocol["checks"][dataset]
    assert len(scores) == check["rows"]
    assert len(tune_users) == check["weight_tune_users"]
    assert len(holdout_users) == check["weight_holdout_users"]

    if dataset == "grocery":
        # The newly supplied final candidate file differs from the archived
        # Grocery file, so every validation grid point must be recomputed.
        grid_rows = []
        for route in ROUTES:
            route_tune = tune[tune.route.eq(route)]
            if route_tune.empty:
                continue
            for weight in config["weight_grid"]:
                route_weights = {name: 0.0 for name in ROUTES}
                route_weights[route] = float(weight)
                metric, _ = evaluate(route_tune, score_variant(route_tune, "A4_full_route_aware", route_weights), [15], popularity)
                row = metric.iloc[0]
                grid_rows.append({"dataset": dataset, "route": route, "weight": float(weight), "tune_users": int(row.users), "k": 15, "recall": float(row.recall), "ndcg": float(row.ndcg), "category_coverage": float(row.category_coverage), "ild": float(row.ild), "novelty": float(row.novelty)})
        grid = pd.DataFrame(grid_rows)
    else:
        # The archived grid was generated on the byte-identical frozen input.
        grid = read_csv(FROZEN / "results" / "route_weight_search.csv")
        grid = grid[grid.dataset.eq(dataset)].copy()
        assert len(grid) == len(ROUTES) * len(config["weight_grid"])
        assert set(grid.route) == set(ROUTES)
        assert set(grid.weight) == set(config["weight_grid"])
        assert grid.k.eq(config["weight_search_k"]).all()
        for route in ROUTES:
            route_grid = grid[grid.route.eq(route)]
            route_tune = tune[tune.route.eq(route)]
            assert route_grid.tune_users.eq(route_tune.user_id.nunique()).all()
            reference = route_grid[route_grid.weight.eq(0.5)].iloc[0]
            metric, _ = evaluate(route_tune, score_variant(route_tune, "A4_full_route_aware", {r: 0.5 for r in ROUTES}), [15], popularity)
            ensure_close(float(metric.iloc[0].ndcg), float(reference.ndcg), f"validation NDCG {route}")
            ensure_close(float(metric.iloc[0].recall), float(reference.recall), f"validation Recall {route}")

    validation_rows = []
    selections = []
    holdout_rows = []
    metrics_by_weights: dict[tuple[float, ...], pd.DataFrame] = {}
    for beta in BETAS:
        selected = {}
        for route in ROUTES:
            sub = grid[grid.route.eq(route)].copy()
            if sub.empty:
                selected[route] = 0.0
                selections.append({"ndcg_coefficient": beta, "recall_coefficient": 1-beta, "route": route, "selected_exploration_weight": 0.0, "validation_ndcg_at_15": np.nan, "validation_recall_at_15": np.nan, "validation_objective": np.nan})
                continue
            sub["ndcg_coefficient"] = beta
            sub["recall_coefficient"] = 1 - beta
            sub["validation_objective"] = beta * sub.ndcg + (1 - beta) * sub.recall
            sub = sub.sort_values(["validation_objective", "ndcg", "recall", "weight"], ascending=[False, False, False, True], kind="mergesort")
            winner = sub.iloc[0]
            selected[route] = float(winner.weight)
            sub["selected"] = sub.weight.eq(winner.weight)
            validation_rows.extend(sub[["ndcg_coefficient", "recall_coefficient", "route", "weight", "tune_users", "k", "validation_objective", "recall", "ndcg", "selected"]].to_dict("records"))
            selections.append({"ndcg_coefficient": beta, "recall_coefficient": 1-beta, "route": route, "selected_exploration_weight": float(winner.weight), "validation_ndcg_at_15": float(winner.ndcg), "validation_recall_at_15": float(winner.recall), "validation_objective": float(winner.validation_objective)})

        key = tuple(selected[route] for route in ROUTES)
        if key not in metrics_by_weights:
            m, _ = evaluate(holdout, score_variant(holdout, "A4_full_route_aware", selected), KS, popularity)
            metrics_by_weights[key] = m
        for row in metrics_by_weights[key].to_dict("records"):
            holdout_rows.append({"ndcg_coefficient": beta, "recall_coefficient": 1-beta, "kg_only_weight": key[0], "mixed_enhancement_weight": key[1], "low_interaction_weight": key[2], **row, "reference_objective_0_6_ndcg_0_4_recall": 0.6*row["ndcg"]+0.4*row["recall"]})

    if dataset != "grocery":
        # Exact reproduction of the published beta=0.6 frozen holdout results.
        baseline = read_csv(FROZEN / "results" / "selected_weights_holdout_metrics.csv")
        baseline = baseline[baseline.dataset.eq(dataset) & baseline.scope.eq("all_holdout_users")]
        new_baseline = pd.DataFrame(holdout_rows)
        new_baseline = new_baseline[new_baseline.ndcg_coefficient.eq(0.6)]
        for _, row in new_baseline.iterrows():
            old = baseline[baseline.k.eq(row.k)].iloc[0]
            for metric in ("recall", "ndcg", "category_coverage", "ild", "novelty"):
                ensure_close(float(row[metric]), float(old[metric]), f"baseline {metric}@{row.k}")

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(validation_rows).sort_values(["ndcg_coefficient", "route", "weight"]).to_csv(out / "validation_grid.csv", index=False)
    pd.DataFrame(selections).to_csv(out / "selected_route_weights.csv", index=False)
    pd.DataFrame(holdout_rows).to_csv(out / "holdout_metrics.csv", index=False)
    paper = pd.DataFrame(holdout_rows).query("k == 10").copy()
    columns = ("recall", "ndcg", "category_coverage", "ild", "novelty")
    best = {column: float(paper[column].max()) for column in columns}
    varies = {column: not np.isclose(paper[column].max(), paper[column].min(), rtol=0, atol=1e-12) for column in columns}
    lines = [
        "% Add \\usepackage{booktabs} in the main document preamble.",
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{dataset.capitalize()} objective-coefficient sensitivity on holdout users (K=10). Exploration weights maximize $\\beta\\,\\mathrm{{NDCG}}@15+(1-\\beta)\\,\\mathrm{{Recall}}@15$ on validation users. CC denotes category coverage.}}",
        f"\\label{{tab:{dataset}-objective-sensitivity}}",
        "\\small",
        "\\begin{tabular}{ccccccccc}",
        "\\toprule",
        "$\\beta$ & $w_{\\mathrm{KG}}$ & $w_{\\mathrm{Mix}}$ & $w_{\\mathrm{Low}}$ & Recall & NDCG & CC & ILD & Novelty \\\\ ",
        "\\midrule",
    ]
    # Nine columns: beta, three route weights, five metrics.
    for _, row in paper.iterrows():
        mix_weight = "--" if dataset == "grocery" else f"{row.mixed_enhancement_weight:.1f}"
        values = [f"{row.ndcg_coefficient:.1f}", f"{row.kg_only_weight:.1f}", mix_weight, f"{row.low_interaction_weight:.1f}"]
        for column in columns:
            value = f"{row[column]:.4f}"
            if varies[column] and np.isclose(row[column], best[column], rtol=0, atol=1e-12):
                value = "\\textbf{" + value + "}"
            values.append(value)
        lines.append(" & ".join(values) + " \\\\ ")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    (out / f"{dataset}_objective_sensitivity.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    audit = {"dataset": dataset, "candidate_scores": str(source), "candidate_scores_sha256": source_checksums["scores"], "training_history": str(history), "training_history_sha256": source_checksums["history"], "matches_frozen_inputs": all(source_checksums[name] == expected[f"{name if name == 'scores' else 'history'}_sha256"] for name in source_checksums), "users_total": int(scores.user_id.nunique()), "users_validation": len(tune_users), "users_holdout": len(holdout_users), "weight_search_k": 15, "weight_grid": config["weight_grid"], "coefficient_grid": BETAS, "holdout_baseline_reproduced": dataset != "grocery", "unique_holdout_rankings_evaluated": len(metrics_by_weights)}
    (out / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(pd.DataFrame(holdout_rows).query("k == 15").to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, default="automotive")
    main(parser.parse_args().dataset)
