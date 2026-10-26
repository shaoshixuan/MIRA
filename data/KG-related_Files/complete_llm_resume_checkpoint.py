"""Rebuild a complete evaluation-ready CSV from an LLM resume checkpoint.

The append-only checkpoint intentionally stores only four durable fields.
This script joins those fields back to the pre-resume full result, preserves
all metadata, recomputes the final score with the original fusion rule, and
validates the resulting 11-column file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FULL_COLUMNS = [
    "user_id",
    "item_id",
    "asin",
    "title",
    "description",
    "is_positive",
    "original_pctr_score",
    "kg_interest_score",
    "llm_exploration_score",
    "final_combined_score",
    "scoring_method",
]
CHECKPOINT_COLUMNS = [
    "user_id",
    "item_id",
    "llm_exploration_score",
    "scoring_method",
]
KEY_COLUMNS = ["user_id", "item_id"]
FAILED_METHODS = {"LLM_failed", "LLM_fallback", "unknown"}

DEFAULT_DIRECTORY = Path(
    "D:/graduation_recsys/data/KG-related_Files/llm_interest_scores"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand the compact LLM resume checkpoint to a full result CSV"
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_DIRECTORY / "llm_exploration_scores_20260807_212313.csv",
        help="Full 11-column result saved after the interrupted run",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_DIRECTORY / "llm_resume_checkpoint_20260807.csv",
        help="Compact four-column append-only resume checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DIRECTORY / "llm_resume_checkpoint_20260807_complete.csv",
        help="New evaluation-ready 11-column CSV",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_DIRECTORY / "llm_exploration_scores_20260810_033653.csv",
        help="Optional known complete result used only for verification",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow replacing an existing output path; source files are still protected",
    )
    return parser.parse_args()


def read_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    missing = set(required_columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    return frame


def normalized_key(values: pd.Series) -> pd.Series:
    # Avoid an integer ID becoming e.g. "61830.0" after CSV type inference.
    return values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def add_keys(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["_user_key"] = normalized_key(frame["user_id"])
    frame["_item_key"] = normalized_key(frame["item_id"])
    return frame


def assert_unique(frame: pd.DataFrame, label: str) -> None:
    duplicated = int(frame.duplicated(["_user_key", "_item_key"]).sum())
    if duplicated:
        raise ValueError(f"{label} has {duplicated} duplicate (user_id, item_id) keys")


def rebuild(base: pd.DataFrame, checkpoint: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    base = add_keys(base)
    checkpoint = add_keys(checkpoint)
    assert_unique(base, "base")
    assert_unique(checkpoint, "checkpoint")

    checkpoint_updates = checkpoint[
        ["_user_key", "_item_key", "llm_exploration_score", "scoring_method"]
    ].rename(
        columns={
            "llm_exploration_score": "_checkpoint_llm_score",
            "scoring_method": "_checkpoint_method",
        }
    )
    merged = base.merge(
        checkpoint_updates,
        on=["_user_key", "_item_key"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing_checkpoint_keys = int((merged["_merge"] == "right_only").sum())
    # A left merge cannot expose right_only rows, so verify the count directly.
    matched_checkpoint_rows = int((merged["_merge"] == "both").sum())
    missing_checkpoint_keys = len(checkpoint) - matched_checkpoint_rows
    if missing_checkpoint_keys:
        raise ValueError(
            f"{missing_checkpoint_keys} checkpoint keys do not exist in the base result"
        )

    update_mask = merged["_merge"] == "both"
    merged.loc[update_mask, "llm_exploration_score"] = pd.to_numeric(
        merged.loc[update_mask, "_checkpoint_llm_score"], errors="raise"
    )
    merged.loc[update_mask, "scoring_method"] = merged.loc[
        update_mask, "_checkpoint_method"
    ]

    pctr = pd.to_numeric(merged["original_pctr_score"], errors="raise")
    kg = pd.to_numeric(merged["kg_interest_score"], errors="raise")
    llm = pd.to_numeric(merged["llm_exploration_score"], errors="raise")
    original_final = pd.to_numeric(merged["final_combined_score"], errors="raise")
    merged["final_combined_score"] = np.select(
        [np.isclose(kg, -10.0), np.isclose(kg, -5.0)],
        [pctr * 0.3 + llm * 0.7, pctr * 0.5 + llm * 0.5],
        default=original_final,
    ).round(8)
    merged["llm_exploration_score"] = llm.round(8)

    result = merged[FULL_COLUMNS].copy()
    result = result.sort_values(
        ["user_id", "final_combined_score"],
        ascending=[True, False],
        kind="stable",
    ).reset_index(drop=True)

    remaining_failed = int(
        result["scoring_method"].astype(str).isin(FAILED_METHODS).sum()
    )
    report = {
        "base_rows": int(len(base)),
        "checkpoint_rows": int(len(checkpoint)),
        "checkpoint_users": int(checkpoint["_user_key"].nunique()),
        "matched_checkpoint_rows": matched_checkpoint_rows,
        "output_rows": int(len(result)),
        "output_users": int(result["user_id"].nunique()),
        "output_columns": list(result.columns),
        "duplicate_output_keys": int(result.duplicated(KEY_COLUMNS).sum()),
        "remaining_failed_or_fallback_rows": remaining_failed,
    }
    return result, report


def compare_reference(result: pd.DataFrame, reference_path: Path) -> dict:
    if not reference_path.exists():
        return {"reference_exists": False}
    reference = read_csv(reference_path, FULL_COLUMNS)[FULL_COLUMNS]
    left = add_keys(result)
    right = add_keys(reference)
    comparison = left.merge(
        right,
        on=["_user_key", "_item_key"],
        suffixes=("_rebuilt", "_reference"),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    both = comparison["_merge"] == "both"
    llm_equal = np.isclose(
        pd.to_numeric(comparison.loc[both, "llm_exploration_score_rebuilt"]),
        pd.to_numeric(comparison.loc[both, "llm_exploration_score_reference"]),
        rtol=0,
        atol=1e-8,
    )
    final_equal = np.isclose(
        pd.to_numeric(comparison.loc[both, "final_combined_score_rebuilt"]),
        pd.to_numeric(comparison.loc[both, "final_combined_score_reference"]),
        rtol=0,
        atol=1e-8,
    )
    method_equal = (
        comparison.loc[both, "scoring_method_rebuilt"].astype(str).to_numpy()
        == comparison.loc[both, "scoring_method_reference"].astype(str).to_numpy()
    )
    numeric_columns = {
        "is_positive",
        "original_pctr_score",
        "kg_interest_score",
        "llm_exploration_score",
        "final_combined_score",
    }
    column_mismatches = {}
    for column in FULL_COLUMNS:
        if column in KEY_COLUMNS:
            continue
        rebuilt = comparison.loc[both, f"{column}_rebuilt"]
        expected = comparison.loc[both, f"{column}_reference"]
        if column in numeric_columns:
            equal = np.isclose(
                pd.to_numeric(rebuilt),
                pd.to_numeric(expected),
                rtol=0,
                atol=1e-8,
                equal_nan=True,
            )
        else:
            equal = (
                rebuilt.fillna("").astype(str).to_numpy()
                == expected.fillna("").astype(str).to_numpy()
            )
        column_mismatches[column] = int((~equal).sum())
    return {
        "reference_exists": True,
        "reference_rows": int(len(reference)),
        "key_mismatches": int((comparison["_merge"] != "both").sum()),
        "llm_score_mismatches": int((~llm_equal).sum()),
        "final_score_mismatches": int((~final_equal).sum()),
        "method_mismatches": int((~method_equal).sum()),
        "column_mismatches": column_mismatches,
        "all_11_columns_match_by_key": all(
            count == 0 for count in column_mismatches.values()
        ),
    }


def main() -> None:
    args = parse_args()
    protected = {args.base.resolve(), args.checkpoint.resolve()}
    if args.output.resolve() in protected:
        raise ValueError("Output must not overwrite the raw base or compact checkpoint")
    if args.output.exists() and not args.allow_overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}. Use --allow-overwrite to replace it."
        )

    base = read_csv(args.base, FULL_COLUMNS)
    checkpoint = read_csv(args.checkpoint, CHECKPOINT_COLUMNS)
    result, report = rebuild(base, checkpoint)
    report["reference_comparison"] = compare_reference(result, args.reference)

    passed = (
        report["output_rows"] == report["base_rows"]
        and report["matched_checkpoint_rows"] == report["checkpoint_rows"]
        and report["duplicate_output_keys"] == 0
        and report["remaining_failed_or_fallback_rows"] == 0
        and report["output_columns"] == FULL_COLUMNS
    )
    reference = report["reference_comparison"]
    if reference.get("reference_exists"):
        passed = passed and all(
            reference[key] == 0
            for key in [
                "key_mismatches",
                "llm_score_mismatches",
                "final_score_mismatches",
                "method_mismatches",
            ]
        )
        passed = passed and reference["all_11_columns_match_by_key"]
    report["passed"] = bool(passed)
    if not passed:
        raise RuntimeError(json.dumps(report, indent=2, ensure_ascii=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    report_path = args.output.with_suffix(".validation.json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"完整结果已保存: {args.output}")
    print(f"验证报告已保存: {report_path}")


if __name__ == "__main__":
    main()
