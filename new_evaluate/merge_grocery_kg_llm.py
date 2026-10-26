"""Merge Grocery KG and LLM exploration scores for evaluate.py.

The KG table is the complete candidate universe.  LLM scores replace the
sentinel KG score only for exactly matching (user_id, item_id) records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_csv_flexible(path: Path) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"Unable to decode {path}: {last_error}")


def normalized_keys(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = result.columns.str.strip()
    result["user_id"] = result["user_id"].astype(str).str.strip()
    result["item_id"] = result["item_id"].astype(str).str.strip()
    return result


def load_labels(candidate_file: Path) -> pd.DataFrame:
    raw = normalized_keys(read_csv_flexible(candidate_file))
    # Repeated negatives are collapsed exactly as in the score tables.  A
    # positive flag is never discarded if a key appears more than once.
    return (
        raw.groupby(["user_id", "item_id"], as_index=False)
        .agg(
            is_positive=("is_positive", "max"),
            rank_in_candidates=("rank_in_candidates", "min"),
        )
    )


def load_categories(category_workbook: Path) -> pd.DataFrame:
    entities = pd.read_excel(category_workbook, sheet_name="Sheet1")
    relations = pd.read_excel(category_workbook, sheet_name="Sheet3")
    entities = entities.rename(columns={"categry": "category"})
    relations = relations[relations["relation_id"] == 8]
    mapped = relations.merge(
        entities[["entities_id", "category"]], on="entities_id", how="left"
    )
    mapped["item_id"] = mapped["item_id"].astype(str).str.strip()
    mapped["category"] = mapped["category"].fillna("").astype(str).str.strip()
    mapped = mapped[mapped["category"] != ""]
    return (
        mapped.groupby("item_id", as_index=False)
        .agg(
            category=("category", lambda s: " | ".join(dict.fromkeys(s))),
            category_entity_ids=(
                "entities_id",
                lambda s: " | ".join(dict.fromkeys(s.astype(str))),
            ),
        )
    )


def merge(args: argparse.Namespace) -> dict:
    kg = normalized_keys(read_csv_flexible(args.kg_file))
    llm = normalized_keys(read_csv_flexible(args.llm_file))

    for name, frame in (("KG", kg), ("LLM", llm)):
        duplicates = int(frame.duplicated(["user_id", "item_id"]).sum())
        if duplicates:
            raise ValueError(f"{name} contains {duplicates} duplicate keys")

    llm_keys = set(zip(llm["user_id"], llm["item_id"]))
    kg_keys = set(zip(kg["user_id"], kg["item_id"]))
    required = kg["kg_interest_score"].isin([-10, -5])
    required_keys = set(zip(kg.loc[required, "user_id"], kg.loc[required, "item_id"]))
    if llm_keys != required_keys:
        raise ValueError(
            "LLM keys must exactly match sentinel KG keys: "
            f"missing={len(required_keys - llm_keys)}, extra={len(llm_keys - required_keys)}"
        )

    llm_payload = llm[
        ["user_id", "item_id", "exploration_score", "scoring_method"]
    ].rename(
        columns={
            "exploration_score": "llm_exploration_score",
            "scoring_method": "llm_scoring_method",
        }
    )
    merged = kg.merge(llm_payload, on=["user_id", "item_id"], how="left", validate="one_to_one")
    merged["original_final_score"] = merged["final_combined_score"]

    low = merged["kg_interest_score"] == -10
    enhancement = merged["kg_interest_score"] == -5
    valid_kg = ~(low | enhancement)
    merged["explore_score_source"] = np.select(
        [low, enhancement], ["LLM_low_interaction", "LLM_enhancement"], default="KG"
    )
    merged["explore_score"] = merged["kg_interest_score"]
    merged.loc[low | enhancement, "explore_score"] = merged.loc[
        low | enhancement, "llm_exploration_score"
    ]

    pctr = merged["original_pctr_score"]
    merged.loc[valid_kg, "final_combined_score"] = (
        0.7 * pctr[valid_kg] + 0.3 * merged.loc[valid_kg, "kg_interest_score"]
    )
    merged.loc[low, "final_combined_score"] = (
        0.7 * pctr[low] + 0.3 * merged.loc[low, "llm_exploration_score"]
    )
    merged.loc[enhancement, "final_combined_score"] = (
        0.5 * pctr[enhancement] + 0.5 * merged.loc[enhancement, "llm_exploration_score"]
    )

    labels = load_labels(args.candidate_file)
    merged = merged.merge(labels, on=["user_id", "item_id"], how="left", validate="one_to_one")
    if merged["is_positive"].isna().any():
        raise ValueError(f"Missing labels for {int(merged['is_positive'].isna().sum())} rows")
    merged["is_positive"] = merged["is_positive"].astype(int)

    categories = load_categories(args.category_workbook)
    merged = merged.merge(categories, on="item_id", how="left", validate="many_to_one")
    merged["category"] = merged["category"].fillna("")
    merged["category_entity_ids"] = merged["category_entity_ids"].fillna("")

    merged["asin"] = ""
    merged["title"] = ""
    merged["description"] = ""
    llm_metadata = llm[["user_id", "item_id", "item_code", "description"]].rename(
        columns={"item_code": "asin_llm", "description": "description_llm"}
    )
    merged = merged.merge(llm_metadata, on=["user_id", "item_id"], how="left", validate="one_to_one")
    merged["asin"] = merged["asin_llm"].fillna("")
    merged["description"] = merged["description_llm"].fillna("")
    merged = merged.drop(columns=["asin_llm", "description_llm"])

    output_columns = [
        "user_id", "item_id", "asin", "title", "description", "is_positive",
        "original_pctr_score", "kg_interest_score", "llm_exploration_score",
        "explore_score", "explore_score_source", "original_final_score",
        "final_combined_score", "llm_scoring_method", "category",
        "category_entity_ids", "rank_in_candidates",
    ]
    merged = merged[output_columns].sort_values(
        ["user_id", "final_combined_score"], ascending=[True, False]
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_file, index=False, encoding="utf-8-sig")

    per_user = merged.groupby("user_id").size()
    report = {
        "kg_file": str(args.kg_file),
        "llm_file": str(args.llm_file),
        "output_file": str(args.output_file),
        "rows": int(len(merged)),
        "users": int(merged["user_id"].nunique()),
        "items": int(merged["item_id"].nunique()),
        "positive_rows": int(merged["is_positive"].sum()),
        "duplicate_keys": int(merged.duplicated(["user_id", "item_id"]).sum()),
        "candidate_count_min": int(per_user.min()),
        "candidate_count_median": float(per_user.median()),
        "candidate_count_max": int(per_user.max()),
        "source_counts": {k: int(v) for k, v in merged["explore_score_source"].value_counts().items()},
        "missing_llm_for_sentinel": int(merged.loc[low | enhancement, "llm_exploration_score"].isna().sum()),
        "category_coverage_rows": int((merged["category"] != "").sum()),
    }
    args.report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg-file", type=Path, required=True)
    parser.add_argument("--llm-file", type=Path, required=True)
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--category-workbook", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--report-file", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(merge(parse_args()), ensure_ascii=False, indent=2))
