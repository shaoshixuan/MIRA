from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

COMMON_ROOT = Path(r"D:\graduation_recsys\supplementary_experiments_dataset_adaptive_20260827")
sys.path.insert(0, str(COMMON_ROOT))
import run_all_experiments as common


ROOT = Path(__file__).resolve().parent
INPUT_PCTR = Path(r"D:\graduation_recsys\evaluate\原始数\test_predictions_4000_for_reranking.csv")
INPUT_KG = Path(r"D:\graduation_recsys\evaluate\原始数\reranked_predictions_with_kg.csv")
LEGACY_HYBRID = Path(r"D:\graduation_recsys\evaluate\原始数\exploration_scores_with_llm.csv")
KG_ROOT = Path(r"E:\graduate_rec_dataset\Grocery_and_Gourmet_Food-20251221T122107Z-3-001\Grocery_and_Gourmet_Food\KG-related_Files")
KG_ITEMS = KG_ROOT / "kg_items_Grocery_and_Gourmet_Food.txt"
KG_TRIPLES = KG_ROOT / "kg_other_triples_Grocery_and_Gourmet_Food.txt"
KG_TRAIN = KG_ROOT / "kg_train_triples_Grocery_and_Gourmet_Food.txt"
KG_ENTITIES = KG_ROOT / "kg_entities_Grocery_and_Gourmet_Food.txt"
META = Path(r"E:\meta_Grocery_and_Gourmet_Food.jsonl\meta_Grocery_and_Gourmet_Food.jsonl")

PROCESSED = ROOT / "processed"
LLM_INPUT = ROOT / "llm_input"
LLM_OUTPUT = ROOT / "llm_output"
RESULTS = ROOT / "results"
for directory in (PROCESSED, LLM_INPUT, LLM_OUTPUT, RESULTS):
    directory.mkdir(parents=True, exist_ok=True)


def clean_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = " ".join(str(part) for part in value if part is not None)
    text = str(value).replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
    return "" if text.lower() in {"nan", "none", "[]"} else " ".join(text.split())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_item_map() -> tuple[dict[str, str], dict[str, str]]:
    item_to_asin: dict[str, str] = {}
    asin_to_item: dict[str, str] = {}
    with KG_ITEMS.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            entity, item_id = parts
            asin = entity.split("::", 1)[1] if "::" in entity else entity
            item_id = item_id.strip()
            item_to_asin[item_id] = asin.strip()
            asin_to_item[asin.strip()] = item_id
    return item_to_asin, asin_to_item


def load_entity_names() -> dict[str, str]:
    names: dict[str, str] = {}
    with KG_ENTITIES.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            name, entity_id = parts
            names[entity_id.strip()] = name.split("::", 1)[-1].strip()
    return names


def load_kg_categories(candidate_items: set[str], entity_names: dict[str, str]) -> dict[str, list[str]]:
    categories: dict[str, set[str]] = defaultdict(set)
    with KG_TRIPLES.open("r", encoding="utf-8") as handle:
        for line in handle:
            head, tail, relation = line.rstrip("\n").split("\t")
            if relation == "8" and head in candidate_items:
                name = entity_names.get(tail, "")
                if name:
                    categories[head].add(name)
    return {item: sorted(values) for item, values in categories.items()}


def load_history() -> tuple[dict[str, list[str]], pd.Series]:
    histories: dict[str, list[str]] = defaultdict(list)
    popularity = Counter()
    with KG_TRAIN.open("r", encoding="utf-8") as handle:
        for line in handle:
            head, tail, relation = line.rstrip("\n").split("\t")
            if relation != "0":
                continue
            histories[head].append(tail)
            popularity[tail] += 1
    histories = {user: list(dict.fromkeys(items)) for user, items in histories.items()}
    total = sum(popularity.values())
    popularity_series = pd.Series({item: count / total for item, count in popularity.items()}, dtype=float)
    return histories, popularity_series


def stream_metadata(needed_asins: set[str]) -> tuple[dict[str, dict], dict]:
    metadata: dict[str, dict] = {}
    scanned = malformed = duplicate_matches = 0
    with META.open("r", encoding="utf-8") as handle:
        for line in handle:
            scanned += 1
            try:
                row = json.loads(line)
            except Exception:
                malformed += 1
                continue
            keys = [clean_text(row.get("asin")), clean_text(row.get("parent_asin"))]
            matched_keys = [key for key in keys if key and key in needed_asins]
            if not matched_keys:
                continue
            entry = {
                "title": clean_text(row.get("title")),
                "description": clean_text(row.get("description")),
                "brand": clean_text(row.get("store") or row.get("brand")),
                "categories": [clean_text(x) for x in row.get("categories", []) if clean_text(x)],
            }
            for key in set(matched_keys):
                previous = metadata.get(key)
                if previous is not None:
                    duplicate_matches += 1
                    # Prefer the record with a non-empty description, then title.
                    previous_quality = (bool(previous["description"]), bool(previous["title"]))
                    new_quality = (bool(entry["description"]), bool(entry["title"]))
                    if new_quality <= previous_quality:
                        continue
                metadata[key] = entry
    stats = {
        "metadata_lines_scanned": scanned,
        "metadata_malformed_lines": malformed,
        "needed_asins": len(needed_asins),
        "matched_asins": len(metadata),
        "duplicate_matching_records": duplicate_matches,
    }
    return metadata, stats


def load_candidates() -> tuple[pd.DataFrame, dict]:
    pctr = pd.read_csv(INPUT_PCTR)
    pctr["user_id"] = pctr["user_id"].astype(str)
    pctr["item_id"] = pctr["item_id"].astype(str)
    raw_rows = len(pctr)
    duplicate_rows = int(pctr.duplicated(["user_id", "item_id"]).sum())
    # Repeated test events contain identical duplicate candidate rows. Keep one key and
    # preserve a positive label if the same key occurs more than once.
    pctr = pctr.groupby(["user_id", "item_id"], as_index=False, sort=False).agg(
        pctr_score=("pctr_score", "first"),
        is_positive=("is_positive", "max"),
        rank_in_candidates=("rank_in_candidates", "min"),
    )

    kg = pd.read_csv(INPUT_KG)
    kg["user_id"] = kg["user_id"].astype(str)
    kg["item_id"] = kg["item_id"].astype(str)
    kg = kg.drop_duplicates(["user_id", "item_id"], keep="first")
    merged = pctr.merge(
        kg[["user_id", "item_id", "kg_interest_score", "final_combined_score"]],
        on=["user_id", "item_id"], how="left", validate="one_to_one",
    )
    merged = merged.rename(columns={"pctr_score": "original_pctr_score"})
    stats = {
        "raw_pctr_rows": raw_rows,
        "duplicate_pctr_rows_removed": duplicate_rows,
        "unique_candidate_rows": len(merged),
        "users": int(merged["user_id"].nunique()),
        "positive_rows": int(merged["is_positive"].sum()),
        "missing_kg_rows": int(merged["kg_interest_score"].isna().sum()),
    }
    return merged, stats


def kg_needs_enhancement(group: pd.DataFrame) -> tuple[bool, pd.Series, dict]:
    valid = group["kg_interest_score"].notna() & ~group["kg_interest_score"].isin([-10, -5])
    values = group.loc[valid, "kg_interest_score"].astype(float)
    max_score = float(values.max()) if len(values) else 0.0
    min_score = float(values.min()) if len(values) else 0.0
    score_range = max_score - min_score
    coverage = float(valid.mean()) if len(group) else 0.0
    needs = max_score < 0.001 or score_range < 0.0005 or coverage < 0.3 or len(values) < 5
    if not needs or not len(values):
        poor = pd.Series(False, index=group.index)
    else:
        average = float(values.mean())
        poor = (~valid) | (group["kg_interest_score"].astype(float) < average * 0.1)
    quality = {
        "kg_max": max_score,
        "kg_min": min_score,
        "kg_range": score_range,
        "kg_coverage": coverage,
        "kg_valid_candidates": int(valid.sum()),
        "needs_enhancement": bool(needs),
        "poor_candidates": int(poor.sum()),
    }
    return bool(needs and poor.any()), poor, quality


def prepare() -> dict:
    candidates, stats = load_candidates()
    item_to_asin, _ = load_item_map()
    histories, popularity = load_history()
    candidate_items = set(candidates["item_id"])
    history_items = {item for user in candidates["user_id"].unique() for item in histories.get(user, [])}
    needed_items = candidate_items | history_items
    needed_asins = {item_to_asin[item] for item in needed_items if item in item_to_asin}
    metadata, metadata_stats = stream_metadata(needed_asins)
    entity_names = load_entity_names()
    kg_categories = load_kg_categories(candidate_items, entity_names)

    candidates["asin"] = candidates["item_id"].map(item_to_asin).fillna("")
    candidates["title"] = candidates["asin"].map(lambda x: metadata.get(x, {}).get("title", ""))
    candidates["description"] = candidates["asin"].map(lambda x: metadata.get(x, {}).get("description", ""))
    candidates["brand"] = candidates["asin"].map(lambda x: metadata.get(x, {}).get("brand", ""))
    candidates["category"] = candidates.apply(
        lambda row: "|".join(metadata.get(row["asin"], {}).get("categories", []) or kg_categories.get(row["item_id"], [])), axis=1
    )

    route_rows = []
    poor_indices: set[int] = set()
    for user_id, group in candidates.groupby("user_id", sort=False):
        history = histories.get(user_id, [])
        history_count = len(history)
        history_metadata = [metadata.get(item_to_asin.get(item, ""), {}) for item in history]
        history_has_text = [bool(info.get("title", "") or info.get("description", "")) for info in history_metadata]
        candidate_has_text = group["title"].ne("") | group["description"].ne("")
        missing_history = sum(not value for value in history_has_text)
        missing_candidates = int((~candidate_has_text).sum())
        # User-level availability gate: at least one textual history item and at
        # least one textual candidate are enough to call the LLM. Individual
        # candidates without text are excluded from semantic scoring and fall back.
        llm_text_eligible = any(history_has_text) and bool(candidate_has_text.any())

        if history_count < 5:
            intended_route = "LOW_INTERACTION"
            # Low-interaction users stay on the low-interaction route even when
            # the LLM gate is closed. Candidate rows without usable text fall
            # back to KG rather than to pCTR-only scoring.
            applied_route = "LOW_INTERACTION"
            weak = False
            poor = pd.Series(False, index=group.index)
            quality = {"kg_max": 0.0, "kg_min": 0.0, "kg_range": 0.0, "kg_coverage": 0.0, "kg_valid_candidates": 0, "needs_enhancement": False, "poor_candidates": 0}
        else:
            weak, poor, quality = kg_needs_enhancement(group)
            intended_route = "MIXED_ENHANCEMENT" if weak else "KG_ONLY"
            if weak and llm_text_eligible:
                applied_route = "MIXED_ENHANCEMENT"
                poor_indices.update(poor[poor].index.tolist())
            else:
                applied_route = "KG_ONLY"

        route_rows.append({
            "user_id": user_id,
            "history_count": history_count,
            "history_missing_description_count": missing_history,
            "history_items_with_text": int(sum(history_has_text)),
            "candidate_count": int(len(group)),
            "candidate_missing_description_count": missing_candidates,
            "candidate_items_with_text": int(candidate_has_text.sum()),
            "llm_text_eligible": llm_text_eligible,
            "intended_route": intended_route,
            "applied_route": applied_route,
            **quality,
        })

    routes = pd.DataFrame(route_rows)
    candidates = candidates.merge(routes[["user_id", "history_count", "llm_text_eligible", "intended_route", "applied_route"]], on="user_id", how="left", validate="many_to_one")
    candidates["kg_score_for_routing"] = candidates["kg_interest_score"]
    low_eligible = candidates["applied_route"].eq("LOW_INTERACTION")
    candidates.loc[low_eligible, "kg_score_for_routing"] = -10
    if poor_indices:
        candidates.loc[list(poor_indices), "kg_score_for_routing"] = -5

    row_has_text = candidates["title"].ne("") | candidates["description"].ne("")
    llm_rows = candidates[candidates["kg_score_for_routing"].isin([-10, -5]) & candidates["llm_text_eligible"] & row_has_text].copy()
    llm_rows["kg_interest_score"] = llm_rows["kg_score_for_routing"]
    llm_rows["final_combined_score"] = llm_rows["original_pctr_score"]
    llm_columns = ["user_id", "item_id", "asin", "title", "description", "is_positive", "original_pctr_score", "kg_interest_score", "final_combined_score"]
    llm_rows[llm_columns].to_csv(LLM_INPUT / "llm_input_data.csv", index=False, encoding="utf-8-sig")

    history_json = {}
    for user_id in llm_rows["user_id"].drop_duplicates():
        items = []
        for item in histories.get(user_id, []):
            asin = item_to_asin.get(item, "")
            info = metadata.get(asin, {})
            if info.get("title", "") or info.get("description", ""):
                items.append({"item_id": item, "asin": asin, "title": info.get("title", ""), "description": info.get("description", ""), "brand": info.get("brand", ""), "categories": info.get("categories", [])})
        history_json[user_id] = {"interaction_count": len(histories.get(user_id, [])), "matched_history_count": len(items), "user_type": routes.set_index("user_id").at[user_id, "intended_route"], "history_items": items}
    (LLM_INPUT / "llm_user_history.json").write_text(json.dumps(history_json, ensure_ascii=False, indent=2), encoding="utf-8")

    candidates.to_csv(PROCESSED / "candidates_with_metadata_and_routes.csv", index=False, encoding="utf-8-sig")
    routes.to_csv(PROCESSED / "user_route_audit.csv", index=False, encoding="utf-8-sig")
    popularity.rename("popularity").to_csv(PROCESSED / "history_item_popularity.csv", encoding="utf-8-sig")

    stats.update(metadata_stats)
    stats.update({
        "candidate_item_to_asin_coverage": float(candidates["asin"].ne("").mean()),
        "candidate_title_coverage": float(candidates["title"].ne("").mean()),
        "candidate_description_coverage": float(candidates["description"].ne("").mean()),
        "candidate_category_coverage": float(candidates["category"].ne("").mean()),
        "history_items_needed": len(history_items),
        "history_item_to_asin_coverage": sum(item in item_to_asin for item in history_items) / max(1, len(history_items)),
        "llm_text_eligible_users": int(routes["llm_text_eligible"].sum()),
        "llm_input_users": int(llm_rows["user_id"].nunique()),
        "llm_input_rows": int(len(llm_rows)),
        "intended_route_counts": routes["intended_route"].value_counts().to_dict(),
        "applied_route_counts": routes["applied_route"].value_counts().to_dict(),
        "source_sha256": {str(path): sha256(path) for path in (INPUT_PCTR, INPUT_KG, KG_ITEMS, KG_TRIPLES, KG_TRAIN, KG_ENTITIES, META)},
    })
    (PROCESSED / "preprocessing_report.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: stats[key] for key in stats if key != "source_sha256"}, ensure_ascii=False, indent=2))
    return stats


def build_merged_score_file(llm_result: Path | None = None) -> pd.DataFrame:
    data = pd.read_csv(PROCESSED / "candidates_with_metadata_and_routes.csv", encoding="utf-8-sig", low_memory=False)
    for column in ("user_id", "item_id"):
        data[column] = data[column].astype(str).str.replace(r"\.0$", "", regex=True)

    data["llm_exploration_score"] = np.nan
    data["llm_scoring_method"] = ""
    if llm_result is not None and llm_result.exists():
        llm = pd.read_csv(llm_result, encoding="utf-8-sig", low_memory=False)
        for column in ("user_id", "item_id"):
            llm[column] = llm[column].astype(str).str.replace(r"\.0$", "", regex=True)
        llm = llm[["user_id", "item_id", "llm_exploration_score", "scoring_method"]].drop_duplicates(["user_id", "item_id"])
        data = data.drop(columns=["llm_exploration_score", "llm_scoring_method"]).merge(
            llm.rename(columns={"scoring_method": "llm_scoring_method"}),
            on=["user_id", "item_id"], how="left", validate="one_to_one",
        )
        # The original scorer's pCTR-based fallback is not an LLM observation.
        failed = data["llm_scoring_method"].isin(["LLM_failed", "LLM_fallback"])
        data.loc[failed, "llm_exploration_score"] = np.nan
        data.loc[failed, "llm_scoring_method"] = "LLM_failed"

    # The original low-interaction run already contains the graph-structure
    # fallback used for candidates without usable text. Reuse only that KG
    # branch; old semantic scores are deliberately not reused.
    legacy = pd.read_csv(LEGACY_HYBRID, encoding="gb18030", low_memory=False)
    legacy["user_id"] = legacy["user_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    legacy["item_id"] = legacy["item_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    legacy = legacy[["user_id", "item_id", "exploration_score", "scoring_method"]].drop_duplicates(["user_id", "item_id"])
    legacy = legacy.rename(columns={"exploration_score": "legacy_graph_score", "scoring_method": "legacy_scoring_method"})
    data = data.merge(legacy, on=["user_id", "item_id"], how="left", validate="one_to_one")

    is_low = data["applied_route"].eq("LOW_INTERACTION")
    is_mixed_llm = data["kg_score_for_routing"].eq(-5) & data["applied_route"].eq("MIXED_ENHANCEMENT")
    row_has_text = data["title"].ne("") | data["description"].ne("")
    low_llm = is_low & row_has_text & data["llm_exploration_score"].notna()
    low_kg_fallback = is_low & ~low_llm
    data["explore_score_source"] = "KG"
    data.loc[low_llm, "explore_score_source"] = "LLM_low_interaction"
    data.loc[is_mixed_llm, "explore_score_source"] = "LLM_enhancement"

    # Sentinels are retained only where that route was actually applied. A blocked
    # LLM route falls back to the original pCTR/KG evidence.
    data["kg_interest_score_original"] = data["kg_interest_score"]
    data.loc[low_llm, "kg_interest_score"] = -10
    legacy_is_kg = data["legacy_scoring_method"].fillna("").astype(str).str.startswith(("KG", "备用"))
    legacy_kg_available = low_kg_fallback & legacy_is_kg & data["legacy_graph_score"].notna()
    data.loc[legacy_kg_available, "kg_interest_score"] = data.loc[legacy_kg_available, "legacy_graph_score"]
    data.loc[is_mixed_llm, "kg_interest_score"] = -5
    data["explore_score"] = np.where(
        data["explore_score_source"].eq("KG"),
        data["kg_interest_score"],
        data["llm_exploration_score"],
    )
    output = PROCESSED / "grocery_new_strategy_merged_scores.csv"
    data.to_csv(output, index=False, encoding="utf-8-sig")
    return data


def score_corrected_variant(df: pd.DataFrame, variant: str, route_weights: dict[str, float], global_weight: float) -> np.ndarray:
    pctr = df["pctr"].to_numpy(float)
    kg = np.nan_to_num(df["kg"].to_numpy(float), nan=0.0)
    llm = np.nan_to_num(df["llm"].to_numpy(float), nan=0.0)
    kg_valid = df["kg_valid"].to_numpy(bool)
    llm_valid = df["llm_valid"].to_numpy(bool)
    route_weight = df["route"].map(route_weights).to_numpy(float)
    if variant == "A0_pCTR_only":
        return pctr
    if variant == "A1_pCTR_KG_route_aware":
        return np.where(kg_valid, (1 - route_weight) * pctr + route_weight * kg, pctr)
    if variant == "A2_pCTR_LLM_route_aware":
        return np.where(llm_valid, (1 - route_weight) * pctr + route_weight * llm, pctr)
    both = kg_valid & llm_valid
    only_kg = kg_valid & ~llm_valid
    only_llm = llm_valid & ~kg_valid
    explore = np.zeros(len(df), dtype=float)
    explore[both] = 0.5 * kg[both] + 0.5 * llm[both]
    explore[only_kg] = kg[only_kg]
    explore[only_llm] = llm[only_llm]
    valid = kg_valid | llm_valid
    if variant == "A3_KG_LLM_global_weight":
        return np.where(valid, (1 - global_weight) * pctr + global_weight * explore, pctr)
    if variant == "A4_full_route_aware":
        return np.where(valid, (1 - route_weight) * pctr + route_weight * explore, pctr)
    raise ValueError(variant)


def tune_global_weight(tune: pd.DataFrame, popularity: pd.Series, config: dict) -> tuple[float, pd.DataFrame]:
    records = []
    choices = []
    zero = {route: 0.0 for route in ("KG_ONLY", "MIXED_ENHANCEMENT", "LOW_INTERACTION")}
    for weight in config["weight_grid"]:
        metrics, _ = common.evaluate(tune, score_corrected_variant(tune, "A3_KG_LLM_global_weight", zero, float(weight)), [config["weight_search_k"]], popularity)
        row = metrics.iloc[0]
        objective = common.objective_value(row, config["objective"])
        records.append({"weight": float(weight), "objective": objective, **row.to_dict(), "selected": False})
        choices.append((objective, float(row["ndcg"]), float(row["recall"]), -float(weight), float(weight)))
    selected = max(choices)[-1]
    for record in records:
        record["selected"] = record["weight"] == selected
    return selected, pd.DataFrame(records)


def evaluate_pipeline(llm_result: Path | None = None) -> dict:
    merged = build_merged_score_file(llm_result)
    scores = common.calibrate(common.normalize_scores(merged))
    # Candidate-level methods may coexist inside one user. The user-level route
    # remains determined by interaction status / KG quality.
    explicit_route = merged["applied_route"]
    scores["route"] = explicit_route.to_numpy()
    popularity_frame = pd.read_csv(PROCESSED / "history_item_popularity.csv", encoding="utf-8-sig")
    if "item_id" not in popularity_frame.columns:
        popularity_frame = popularity_frame.rename(columns={popularity_frame.columns[0]: "item_id"})
    popularity_frame["item_id"] = popularity_frame["item_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    popularity = popularity_frame.set_index("item_id")["popularity"]
    config = {
        "seed": 2026,
        "weight_search_tune_ratio": 0.8,
        "weight_grid": [round(x, 1) for x in np.arange(0, 1.01, 0.1)],
        "weight_search_k": 15,
        "ks": [5, 10, 15, 20, 25],
        "objective": {"ndcg": 0.6, "recall": 0.4},
    }
    route_weights, route_search, holdout_route_metrics, tune_users, holdout_users = common.tune_dataset_route_weights(scores, popularity, config, "grocery")
    tune = scores[scores["user_id"].isin(tune_users)].copy()
    holdout = scores[scores["user_id"].isin(holdout_users)].copy()
    global_weight, global_search = tune_global_weight(tune, popularity, config)

    variants = [
        "A0_pCTR_only",
        "A1_pCTR_KG_route_aware",
        "A2_pCTR_LLM_route_aware",
        "A3_KG_LLM_global_weight",
        "A4_full_route_aware",
    ]
    scopes = {
        "all_users": scores,
        "holdout_users": holdout,
        "low_interaction_users": scores[scores["route"].eq("LOW_INTERACTION")],
        "kg_only_users": scores[scores["route"].eq("KG_ONLY")],
    }
    metric_frames = []
    per_user_dir = RESULTS / "per_user"
    per_user_dir.mkdir(exist_ok=True)
    for scope, subset in scopes.items():
        if subset.empty:
            continue
        for variant in variants:
            values = score_corrected_variant(subset, variant, route_weights, global_weight)
            metrics, per_user = common.evaluate(subset, values, [5, 10, 15, 20, 25], popularity)
            metrics.insert(0, "global_weight_A3", global_weight)
            for route, weight in reversed(list(route_weights.items())):
                metrics.insert(0, f"A4_weight_{route}", weight)
            metrics.insert(0, "variant", variant)
            metrics.insert(0, "scope", scope)
            metric_frames.append(metrics)
            for k, frame in per_user.items():
                frame.to_csv(per_user_dir / f"{scope}_{variant}_k{k}.csv", index=False, encoding="utf-8-sig")

    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics.to_csv(RESULTS / "ablation_and_final_metrics.csv", index=False, encoding="utf-8-sig")
    route_search.to_csv(RESULTS / "A4_route_weight_search.csv", index=False, encoding="utf-8-sig")
    global_search.to_csv(RESULTS / "A3_global_weight_search.csv", index=False, encoding="utf-8-sig")
    holdout_route_metrics.to_csv(RESULTS / "A4_holdout_route_metrics.csv", index=False, encoding="utf-8-sig")

    final_scores = score_corrected_variant(scores, "A4_full_route_aware", route_weights, global_weight)
    ranking = merged[["user_id", "item_id", "is_positive", "asin", "title", "description", "category", "applied_route", "original_pctr_score", "kg_interest_score", "llm_exploration_score", "explore_score_source"]].copy()
    ranking["final_score_new_strategy"] = final_scores
    ranking = ranking.sort_values(["user_id", "final_score_new_strategy", "item_id"], ascending=[True, False, True], kind="mergesort")
    ranking["final_rank"] = ranking.groupby("user_id", sort=False).cumcount() + 1
    ranking.to_csv(RESULTS / "final_ranked_recommendations.csv", index=False, encoding="utf-8-sig")

    selected = {
        "selection_protocol": "80% route-stratified tuning users; seed=2026; maximize 0.6*NDCG@15+0.4*Recall@15; tie -> smaller exploration weight",
        "A4_route_weights": route_weights,
        "A3_global_weight": global_weight,
        "tune_users": len(tune_users),
        "holdout_users": len(holdout_users),
        "route_user_counts": scores[["user_id", "route"]].drop_duplicates()["route"].value_counts().to_dict(),
        "llm_result": str(llm_result) if llm_result else None,
    }
    (RESULTS / "selected_weights_and_protocol.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(selected, ensure_ascii=False, indent=2))
    print(metrics[(metrics["scope"] == "all_users") & (metrics["variant"].isin(["A0_pCTR_only", "A4_full_route_aware"]))].to_string(index=False))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "evaluate", "all"], default="all", nargs="?")
    parser.add_argument("--llm-result", type=Path, default=None)
    args = parser.parse_args()
    if args.stage in {"prepare", "all"}:
        prepare()
    if args.stage in {"evaluate", "all"}:
        evaluate_pipeline(args.llm_result)


if __name__ == "__main__":
    main()
