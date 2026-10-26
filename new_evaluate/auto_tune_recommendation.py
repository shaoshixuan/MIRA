#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动搜索推荐融合参数，并生成最终优化评分。

必须与 recompute_recommendation_scores.py 放在同一目录。

流程：
1. 按用户随机划分调参集和留出评估集；
2. 分别为 KG_ONLY、MIXED_ENHANCEMENT、LOW_INTERACTION 搜索最佳多样性权重；
3. 可选：在调参用户样本上搜索若干组 MMR 参数；
4. 只有当 MMR 在调参集上的 NDCG/Recall 综合目标优于统一加权时，才采用 MMR；
5. 将最佳方案应用到全部数据；
6. 输出新评分 CSV、参数 JSON、评估结果 CSV。

默认优化目标：
    0.6 * NDCG@K + 0.4 * Recall@K

注意：
- “最佳”仅指在自动划分的调参集上目标值最高。
- 最终可信指标应查看留出评估集结果。
"""

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自动调优推荐融合与 MMR 参数")
    parser.add_argument("--input", required=True, help="输入评分 CSV")
    parser.add_argument(
        "--scorer",
        default=None,
        help=(
            "recompute_recommendation_scores.py 路径；"
            "默认使用本脚本同目录下的文件"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出评分 CSV；默认生成 *_optimized.csv",
    )
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--top-m", type=int, default=50)
    parser.add_argument(
        "--tune-ratio",
        type=float,
        default=0.80,
        help="用于搜索参数的用户比例，剩余用户用于留出评估",
    )
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--objective",
        choices=["balanced", "ndcg", "recall"],
        default="balanced",
        help="balanced = 0.6*NDCG + 0.4*Recall",
    )
    parser.add_argument(
        "--llm-zero-policy",
        choices=["valid", "missing_if_no_description", "missing_all"],
        default="missing_if_no_description",
    )
    parser.add_argument(
        "--skip-mmr-search",
        action="store_true",
        help="仅搜索统一加权，不搜索 MMR",
    )
    parser.add_argument(
        "--mmr-search-users",
        type=int,
        default=500,
        help="搜索 MMR 参数时最多使用的调参用户数",
    )
    parser.add_argument(
        "--similarity-mode",
        choices=["category", "text", "hybrid"],
        default="hybrid",
    )
    parser.add_argument(
        "--text-similarity-weight",
        type=float,
        default=0.70,
    )
    parser.add_argument("--encoding", default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.k <= 0:
        raise ValueError("--k 必须大于 0")
    if args.top_m < args.k:
        raise ValueError("--top-m 必须大于或等于 --k")
    if not 0.5 <= args.tune_ratio < 1.0:
        raise ValueError("--tune-ratio 必须位于 [0.5, 1.0)")
    if args.mmr_search_users <= 0:
        raise ValueError("--mmr-search-users 必须大于 0")


def load_scorer(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"找不到评分脚本：{path}\n"
            "请把 auto_tune_recommendation.py 和 "
            "recompute_recommendation_scores.py 放在同一目录。"
        )

    spec = importlib.util.spec_from_file_location("recommendation_scorer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载评分脚本：{path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def objective_value(metrics: Dict[str, float], objective: str) -> float:
    if objective == "ndcg":
        return float(metrics["NDCG"])
    if objective == "recall":
        return float(metrics["Recall"])
    return 0.60 * float(metrics["NDCG"]) + 0.40 * float(metrics["Recall"])


def stratified_user_split(
    data: pd.DataFrame,
    tune_ratio: float,
    seed: int,
) -> Tuple[set, set]:
    rng = random.Random(seed)
    tune_users = set()
    holdout_users = set()

    routes = (
        data[["user_id", "user_route"]]
        .drop_duplicates("user_id")
        .groupby("user_route", sort=False)
    )

    for route, route_frame in routes:
        users = route_frame["user_id"].tolist()
        rng.shuffle(users)

        if len(users) == 1:
            tune_count = 1
        else:
            tune_count = int(round(len(users) * tune_ratio))
            tune_count = min(max(tune_count, 1), len(users) - 1)

        tune_users.update(users[:tune_count])
        holdout_users.update(users[tune_count:])

        print(
            f"用户划分 [{route}]："
            f"调参 {tune_count}，留出 {len(users) - tune_count}"
        )

    return tune_users, holdout_users


def route_weight_grid(route: str) -> List[float]:
    if route == "KG_ONLY":
        return [round(x, 2) for x in np.arange(0.00, 0.31, 0.05)]
    if route == "MIXED_ENHANCEMENT":
        return [round(x, 2) for x in np.arange(0.00, 0.41, 0.05)]
    if route == "LOW_INTERACTION":
        return [round(x, 2) for x in np.arange(0.00, 0.71, 0.05)]
    return [0.0]


def candidate_weighted_score(
    subset: pd.DataFrame,
    diversity_weight: float,
) -> np.ndarray:
    pctr = subset["pctr_percentile"].to_numpy(dtype=float)
    diversity = (
        subset["diversity_percentile"]
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    valid = subset["diversity_score_valid"].to_numpy(dtype=bool)

    score = (
        (1.0 - diversity_weight) * pctr
        + diversity_weight * diversity
    )
    return np.where(valid, score, pctr)


def tune_route_weights(
    tune_data: pd.DataFrame,
    scorer,
    k: int,
    objective: str,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    best_weights: Dict[str, float] = {}
    records: List[Dict[str, float]] = []

    for route in ["KG_ONLY", "MIXED_ENHANCEMENT", "LOW_INTERACTION"]:
        subset = tune_data[tune_data["user_route"] == route].copy()
        if subset.empty:
            best_weights[route] = 0.0
            continue

        best_candidate = None
        for weight in route_weight_grid(route):
            subset["candidate_score"] = candidate_weighted_score(
                subset, weight
            )
            metrics = scorer.evaluate_score(
                subset, "candidate_score", k
            )
            objective_score = objective_value(metrics, objective)

            record = {
                "stage": "weighted_route_search",
                "route": route,
                "diversity_weight": weight,
                "objective": objective_score,
                **metrics,
            }
            records.append(record)

            candidate = (
                objective_score,
                metrics["NDCG"],
                metrics["Recall"],
                -weight,  # 同分时选择更保守的较小多样性权重
            )
            if best_candidate is None or candidate > best_candidate[0]:
                best_candidate = (candidate, weight, metrics)

        best_weights[route] = float(best_candidate[1])
        print(
            f"最佳权重 [{route}]："
            f"diversity_weight={best_candidate[1]:.2f}, "
            f"Recall={best_candidate[2]['Recall']:.4f}, "
            f"NDCG={best_candidate[2]['NDCG']:.4f}"
        )

    return best_weights, pd.DataFrame(records)


def scoring_namespace(
    args: argparse.Namespace,
    weights: Dict[str, float],
    mmr_preset: Tuple[float, float, float] = (0.75, 0.15, 0.10),
) -> SimpleNamespace:
    rel, history, redundancy = mmr_preset
    return SimpleNamespace(
        k=args.k,
        top_m=args.top_m,
        kg_diversity_weight=weights["KG_ONLY"],
        mixed_diversity_weight=weights["MIXED_ENHANCEMENT"],
        low_diversity_weight=weights["LOW_INTERACTION"],
        llm_zero_policy=args.llm_zero_policy,
        mmr_relevance_weight=rel,
        mmr_history_diversity_weight=history,
        mmr_redundancy_weight=redundancy,
        similarity_mode=args.similarity_mode,
        text_similarity_weight=args.text_similarity_weight,
    )


def mmr_presets() -> Sequence[Tuple[float, float, float]]:
    # relevance, history_diversity, redundancy
    return [
        (0.90, 0.05, 0.05),
        (0.85, 0.10, 0.05),
        (0.80, 0.10, 0.10),
        (0.75, 0.15, 0.10),
        (0.70, 0.20, 0.10),
        (0.70, 0.15, 0.15),
    ]


def sample_users(
    data: pd.DataFrame,
    max_users: int,
    seed: int,
) -> pd.DataFrame:
    users = data["user_id"].drop_duplicates().tolist()
    rng = random.Random(seed)
    rng.shuffle(users)
    chosen = set(users[: min(max_users, len(users))])
    return data[data["user_id"].isin(chosen)].copy()


def tune_mmr(
    weighted_tune_data: pd.DataFrame,
    scorer,
    args: argparse.Namespace,
    weights: Dict[str, float],
) -> Tuple[
    bool,
    Tuple[float, float, float],
    pd.DataFrame,
]:
    search_data = sample_users(
        weighted_tune_data,
        max_users=args.mmr_search_users,
        seed=args.seed + 17,
    )

    weighted_metrics = scorer.evaluate_score(
        search_data, "final_score_weighted", args.k
    )
    weighted_objective = objective_value(
        weighted_metrics, args.objective
    )

    records = [
        {
            "stage": "mmr_search",
            "strategy": "weighted_baseline",
            "relevance_weight": 1.0,
            "history_diversity_weight": 0.0,
            "redundancy_weight": 0.0,
            "objective": weighted_objective,
            **weighted_metrics,
        }
    ]

    best = (
        weighted_objective,
        weighted_metrics["NDCG"],
        weighted_metrics["Recall"],
        (1.0, 0.0, 0.0),
        False,
    )

    for preset in mmr_presets():
        config = scoring_namespace(args, weights, preset)
        candidate = scorer.add_mmr_score(search_data.copy(), config)
        metrics = scorer.evaluate_score(
            candidate, "final_score_mmr", args.k
        )
        value = objective_value(metrics, args.objective)

        rel, history, redundancy = preset
        records.append(
            {
                "stage": "mmr_search",
                "strategy": "mmr",
                "relevance_weight": rel,
                "history_diversity_weight": history,
                "redundancy_weight": redundancy,
                "objective": value,
                **metrics,
            }
        )

        comparison = (
            value,
            metrics["NDCG"],
            metrics["Recall"],
            preset,
            True,
        )
        if comparison[:3] > best[:3]:
            best = comparison

        print(
            "MMR候选："
            f"{rel:.2f}/{history:.2f}/{redundancy:.2f}, "
            f"Recall={metrics['Recall']:.4f}, "
            f"NDCG={metrics['NDCG']:.4f}"
        )

    use_mmr = bool(best[4])
    best_preset = best[3] if use_mmr else (1.0, 0.0, 0.0)

    if use_mmr:
        print(
            "MMR优于统一加权，采用参数："
            f"relevance/history/redundancy="
            f"{best_preset[0]:.2f}/"
            f"{best_preset[1]:.2f}/"
            f"{best_preset[2]:.2f}"
        )
    else:
        print("MMR未提升目标指标，最终采用统一加权。")

    return use_mmr, best_preset, pd.DataFrame(records)


def evaluate_holdout(
    holdout: pd.DataFrame,
    scorer,
    args: argparse.Namespace,
    chosen_score_column: str,
) -> pd.DataFrame:
    columns = [
        ("pCTR", "original_pctr_score"),
        ("旧融合", "final_combined_score"),
        ("统一加权", "final_score_weighted"),
    ]
    if chosen_score_column == "final_score_mmr":
        columns.append(("MMR", "final_score_mmr"))

    rows = []
    for name, column in columns:
        if column not in holdout.columns:
            continue
        metrics = scorer.evaluate_score(holdout, column, args.k)
        rows.append(
            {
                "dataset": "holdout_users",
                "strategy": name,
                "score_column": column,
                "objective": objective_value(metrics, args.objective),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    validate_args(args)

    script_path = Path(__file__).resolve()
    scorer_path = (
        Path(args.scorer)
        if args.scorer
        else script_path.with_name("recompute_recommendation_scores.py")
    )
    scorer = load_scorer(scorer_path)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}_optimized.csv")
    )
    params_path = output_path.with_name(
        f"{output_path.stem}_best_params.json"
    )
    search_path = output_path.with_name(
        f"{output_path.stem}_search_results.csv"
    )
    metrics_path = output_path.with_name(
        f"{output_path.stem}_holdout_metrics.csv"
    )

    data, used_encoding = scorer.read_csv_with_fallback(
        input_path, args.encoding
    )
    data = scorer.prepare_data(data)

    # 校准过程不使用 is_positive 标签。
    base_config = scoring_namespace(
        args,
        {
            "KG_ONLY": 0.0,
            "MIXED_ENHANCEMENT": 0.0,
            "LOW_INTERACTION": 0.0,
        },
    )
    data = scorer.add_calibrated_scores(data, base_config)

    tune_users, holdout_users = stratified_user_split(
        data, args.tune_ratio, args.seed
    )
    tune_data = data[data["user_id"].isin(tune_users)].copy()
    holdout_data = data[data["user_id"].isin(holdout_users)].copy()

    best_weights, weighted_search = tune_route_weights(
        tune_data, scorer, args.k, args.objective
    )

    final_config = scoring_namespace(args, best_weights)
    data = scorer.add_weighted_score(data, final_config)
    tune_data = data[data["user_id"].isin(tune_users)].copy()

    mmr_search = pd.DataFrame()
    use_mmr = False
    best_mmr = (1.0, 0.0, 0.0)

    if not args.skip_mmr_search:
        use_mmr, best_mmr, mmr_search = tune_mmr(
            tune_data,
            scorer,
            args,
            best_weights,
        )

    chosen_score_column = "final_score_weighted"
    if use_mmr:
        final_config = scoring_namespace(
            args, best_weights, best_mmr
        )
        data = scorer.add_mmr_score(data, final_config)
        chosen_score_column = "final_score_mmr"

    data["final_score_optimized"] = data[chosen_score_column]

    # 只在留出用户上报告最终可信评估。
    holdout = data[data["user_id"].isin(holdout_users)].copy()
    holdout_metrics = evaluate_holdout(
        holdout,
        scorer,
        args,
        chosen_score_column,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False, encoding="utf-8-sig")

    search_results = pd.concat(
        [weighted_search, mmr_search],
        ignore_index=True,
        sort=False,
    )
    search_results.to_csv(
        search_path, index=False, encoding="utf-8-sig"
    )
    holdout_metrics.to_csv(
        metrics_path, index=False, encoding="utf-8-sig"
    )

    params = {
        "input": str(input_path),
        "input_encoding": used_encoding,
        "k": args.k,
        "top_m": args.top_m,
        "objective": args.objective,
        "objective_formula": (
            "0.6*NDCG+0.4*Recall"
            if args.objective == "balanced"
            else args.objective
        ),
        "llm_zero_policy": args.llm_zero_policy,
        "best_route_diversity_weights": best_weights,
        "selected_strategy": "mmr" if use_mmr else "weighted",
        "best_mmr_weights": {
            "relevance": best_mmr[0],
            "history_diversity": best_mmr[1],
            "redundancy": best_mmr[2],
        },
        "similarity_mode": args.similarity_mode,
        "text_similarity_weight": args.text_similarity_weight,
        "tune_users": len(tune_users),
        "holdout_users": len(holdout_users),
        "final_score_column": "final_score_optimized",
    }
    with params_path.open("w", encoding="utf-8") as file:
        json.dump(params, file, ensure_ascii=False, indent=2)

    print("\n留出用户评估：")
    print(holdout_metrics.to_string(index=False))
    print(f"\n最终评分文件：{output_path}")
    print(f"最佳参数：{params_path}")
    print(f"搜索明细：{search_path}")
    print(f"留出评估：{metrics_path}")
    print("\n后续排序统一使用列：final_score_optimized")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"\n执行失败：{error}", file=sys.stderr)
        sys.exit(1)
