#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新计算推荐分数：统一加权融合 + pCTR Top-M / MMR 重排

适用输入列（与你当前 CSV 一致）：
- user_id
- item_id
- is_positive                    可选，仅用于离线评估
- original_pctr_score
- explore_score
- explore_score_source
- title                          可选，用于 MMR 文本相似度
- description                    可选，用于 MMR 文本相似度
- category                       可选，用于 MMR 类别相似度
- llm_scoring_method             可选，用于识别 LLM_failed
- final_combined_score           可选，用于和旧策略对比

主要策略：
1. weighted：
   - pCTR 在每个用户候选集内转换为百分位分数；
   - KG、LLM_low_interaction、LLM_enhancement 分别校准到 [0, 1]；
   - 同一个用户路由内，所有商品使用相同的多样性权重；
   - 多样性分数无效时直接回退到 pCTR，不把无效分数当成 0 惩罚商品。

2. mmr：
   - 先按 pCTR 取 Top-M；
   - 再综合相关性、历史兴趣多样性和列表内冗余，逐个选出 Top-K；
   - Top-K 之外的商品仍保留完整排序，便于输出和评估。

注意：
- 参数只能在验证集上调优，不能根据测试集 NDCG/Recall 反复调参。
- 若只想验证“纯加权策略”，使用 --strategy weighted。
- 若要严格保留原始 0 分含义，使用 --llm-zero-policy valid。
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.feature_extraction.text import HashingVectorizer
except ImportError:
    HashingVectorizer = None


REQUIRED_COLUMNS = {
    "user_id",
    "item_id",
    "original_pctr_score",
    "explore_score",
    "explore_score_source",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统一加权融合与 MMR 多样性重排"
    )
    parser.add_argument("--input", required=True, help="输入评分 CSV")
    parser.add_argument(
        "--output",
        default=None,
        help="输出 CSV；默认在输入文件旁生成 *_rescored.csv",
    )
    parser.add_argument(
        "--strategy",
        choices=["weighted", "mmr", "both"],
        default="both",
        help="执行统一加权、MMR，或两者都执行",
    )
    parser.add_argument("--k", type=int, default=15, help="最终推荐列表长度")
    parser.add_argument(
        "--top-m",
        type=int,
        default=50,
        help="MMR 只在 pCTR Top-M 中挑选 Top-K",
    )

    # 统一加权策略：这是“多样性权重”，pCTR 权重为 1 - diversity_weight。
    parser.add_argument(
        "--kg-diversity-weight",
        type=float,
        default=0.10,
        help="纯 KG 用户的多样性权重",
    )
    parser.add_argument(
        "--mixed-diversity-weight",
        type=float,
        default=0.20,
        help="KG + LLM_enhancement 用户的统一多样性权重",
    )
    parser.add_argument(
        "--low-diversity-weight",
        type=float,
        default=0.35,
        help="低交互 LLM 用户的多样性权重",
    )

    # MMR 三部分权重。
    parser.add_argument(
        "--mmr-relevance-weight",
        type=float,
        default=0.75,
        help="MMR 中 pCTR 相关性权重",
    )
    parser.add_argument(
        "--mmr-history-diversity-weight",
        type=float,
        default=0.15,
        help="MMR 中现有 KG/LLM 多样性分数权重",
    )
    parser.add_argument(
        "--mmr-redundancy-weight",
        type=float,
        default=0.10,
        help="MMR 中列表内部相似性惩罚权重",
    )
    parser.add_argument(
        "--similarity-mode",
        choices=["category", "text", "hybrid"],
        default="hybrid",
        help="MMR 商品间相似度计算方式",
    )
    parser.add_argument(
        "--text-similarity-weight",
        type=float,
        default=0.70,
        help="hybrid 模式中文本相似度占比，类别占比为 1-该值",
    )

    parser.add_argument(
        "--llm-zero-policy",
        choices=["valid", "missing_if_no_description", "missing_all"],
        default="missing_if_no_description",
        help=(
            "LLM 的 0 分处理：valid=当作真实 0；"
            "missing_if_no_description=描述缺失且为 0 时回退 pCTR；"
            "missing_all=所有 LLM 0 分都回退 pCTR"
        ),
    )
    parser.add_argument(
        "--encoding",
        default=None,
        help="指定 CSV 编码；不指定则依次尝试 utf-8-sig/utf-8/gb18030/gbk",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="不计算 Recall/NDCG 等离线指标",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.k <= 0:
        raise ValueError("--k 必须大于 0")
    if args.top_m < args.k:
        raise ValueError("--top-m 必须大于或等于 --k")

    scalar_weights = {
        "kg_diversity_weight": args.kg_diversity_weight,
        "mixed_diversity_weight": args.mixed_diversity_weight,
        "low_diversity_weight": args.low_diversity_weight,
        "text_similarity_weight": args.text_similarity_weight,
    }
    for name, value in scalar_weights.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} 必须位于 [0, 1]")

    mmr_weights = [
        args.mmr_relevance_weight,
        args.mmr_history_diversity_weight,
        args.mmr_redundancy_weight,
    ]
    if any(weight < 0 for weight in mmr_weights):
        raise ValueError("MMR 权重不能为负数")
    if sum(mmr_weights) <= 0:
        raise ValueError("MMR 权重之和必须大于 0")


def read_csv_with_fallback(
    path: Path, encoding: Optional[str] = None
) -> Tuple[pd.DataFrame, str]:
    encodings = (
        [encoding]
        if encoding
        else ["utf-8-sig", "utf-8", "gb18030", "gbk"]
    )
    errors: List[str] = []

    for enc in encodings:
        try:
            print(f"尝试使用编码 {enc} 读取：{path}")
            data = pd.read_csv(path, encoding=enc, low_memory=False)
            print(f"读取成功：encoding={enc}, rows={len(data)}, cols={len(data.columns)}")
            return data, enc
        except (UnicodeDecodeError, LookupError) as exc:
            errors.append(f"{enc}: {exc}")

    raise ValueError(
        "无法读取 CSV，请通过 --encoding 指定编码。\n" + "\n".join(errors)
    )


def check_columns(data: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"输入文件缺少必要列：{sorted(missing)}")


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.columns = data.columns.str.strip()

    check_columns(data)

    data["user_id"] = data["user_id"].astype(str).str.strip()
    data["item_id"] = data["item_id"].astype(str).str.strip()

    numeric_columns = [
        "original_pctr_score",
        "explore_score",
        "final_combined_score",
        "is_positive",
    ]
    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    if data["original_pctr_score"].isna().any():
        count = int(data["original_pctr_score"].isna().sum())
        raise ValueError(f"original_pctr_score 存在 {count} 个无法解析的值")

    for column in [
        "explore_score_source",
        "title",
        "description",
        "category",
        "llm_scoring_method",
    ]:
        if column not in data.columns:
            data[column] = ""
        else:
            data[column] = data[column].fillna("").astype(str).str.strip()

    return data


def derive_user_route(data: pd.DataFrame) -> pd.Series:
    """
    用户路由：
    - LOW_INTERACTION：该用户候选全部来自 LLM_low_interaction；
    - MIXED_ENHANCEMENT：候选中存在 LLM_enhancement；
    - KG_ONLY：其余情况。
    """
    flags = pd.DataFrame(
        {
            "user_id": data["user_id"],
            "is_low": data["explore_score_source"].eq("LLM_low_interaction"),
            "is_enhancement": data["explore_score_source"].eq("LLM_enhancement"),
            "is_kg": data["explore_score_source"].eq("KG"),
        }
    ).groupby("user_id", sort=False).agg(
        has_low=("is_low", "any"),
        has_enhancement=("is_enhancement", "any"),
        has_kg=("is_kg", "any"),
    )

    route = pd.Series("KG_ONLY", index=flags.index, dtype="object")
    route.loc[flags["has_enhancement"]] = "MIXED_ENHANCEMENT"

    pure_low = flags["has_low"] & ~flags["has_enhancement"] & ~flags["has_kg"]
    route.loc[pure_low] = "LOW_INTERACTION"
    return route


def percentile_from_min_rank(
    values: pd.Series, group: pd.Series, valid: pd.Series
) -> pd.Series:
    """
    分来源经验百分位校准：
    - 各来源独立映射到 [0, 1]，解决 KG 与 LLM 原始量纲不同；
    - 使用 method='min'，最低的一组并列分数映射为 0；
    - 不使用 is_positive，因此不会发生标签泄漏。
    """
    output = pd.Series(np.nan, index=values.index, dtype=float)
    valid_frame = pd.DataFrame(
        {
            "value": values.loc[valid],
            "group": group.loc[valid],
        }
    )

    rank = valid_frame.groupby("group", sort=False)["value"].rank(
        method="min", ascending=True
    )
    size = valid_frame.groupby("group", sort=False)["value"].transform("size")
    denominator = (size - 1).replace(0, np.nan)
    calibrated = (rank - 1) / denominator
    calibrated = calibrated.fillna(0.5).clip(0.0, 1.0)

    output.loc[valid] = calibrated
    return output


def build_valid_diversity_mask(
    data: pd.DataFrame, zero_policy: str
) -> pd.Series:
    valid = data["explore_score"].notna()

    # 明确失败的 LLM 结果无效。
    if "llm_scoring_method" in data.columns:
        valid &= ~data["llm_scoring_method"].eq("LLM_failed")

    is_llm = data["explore_score_source"].isin(
        ["LLM_low_interaction", "LLM_enhancement"]
    )
    is_zero = data["explore_score"].fillna(0).eq(0)

    if zero_policy == "missing_if_no_description":
        no_description = data["description"].eq("")
        valid &= ~(is_llm & is_zero & no_description)
    elif zero_policy == "missing_all":
        valid &= ~(is_llm & is_zero)
    elif zero_policy != "valid":
        raise ValueError(f"未知 llm_zero_policy：{zero_policy}")

    return valid


def add_calibrated_scores(
    data: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    data = data.copy()

    # pCTR 只在当前用户的候选集内比较，因此做用户内百分位。
    data["pctr_percentile"] = (
        data.groupby("user_id", sort=False)["original_pctr_score"]
        .rank(method="average", pct=True)
        .clip(0.0, 1.0)
    )

    route_by_user = derive_user_route(data)
    data["user_route"] = data["user_id"].map(route_by_user)

    data["diversity_score_valid"] = build_valid_diversity_mask(
        data, args.llm_zero_policy
    )

    data["diversity_percentile"] = percentile_from_min_rank(
        values=data["explore_score"],
        group=data["explore_score_source"],
        valid=data["diversity_score_valid"],
    )

    # MMR 中缺失的历史多样性使用 0.5 作为中性值；
    # 统一加权时则直接回退到 pCTR，不使用这个中性值。
    data["diversity_neutral_filled"] = data["diversity_percentile"].fillna(0.5)
    return data


def add_weighted_score(
    data: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    """
    统一融合：
      valid:
        score = (1-lambda_route) * pctr_percentile
                + lambda_route * diversity_percentile
      invalid:
        score = pctr_percentile

    关键点：同一个 MIXED_ENHANCEMENT 用户中，
    KG 商品和 LLM_enhancement 商品使用完全相同的 lambda。
    """
    data = data.copy()

    route_weight = {
        "KG_ONLY": args.kg_diversity_weight,
        "MIXED_ENHANCEMENT": args.mixed_diversity_weight,
        "LOW_INTERACTION": args.low_diversity_weight,
    }
    data["route_diversity_weight"] = (
        data["user_route"].map(route_weight).astype(float)
    )

    pctr = data["pctr_percentile"].to_numpy(dtype=float)
    diversity = data["diversity_percentile"].fillna(0.0).to_numpy(dtype=float)
    valid = data["diversity_score_valid"].to_numpy(dtype=bool)
    weight = data["route_diversity_weight"].to_numpy(dtype=float)

    weighted = (1.0 - weight) * pctr + weight * diversity
    weighted = np.where(valid, weighted, pctr)

    data["final_score_weighted"] = weighted
    data["weighted_fallback_to_pctr"] = ~valid
    return data


def parse_category_set(value: str) -> set:
    if not value:
        return set()
    return {part.strip().lower() for part in value.split("|") if part.strip()}


def category_similarity_matrix(categories: Sequence[str]) -> np.ndarray:
    sets = [parse_category_set(value) for value in categories]
    n = len(sets)
    matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        matrix[i, i] = 1.0 if sets[i] else 0.0
        for j in range(i + 1, n):
            union = sets[i] | sets[j]
            similarity = len(sets[i] & sets[j]) / len(union) if union else 0.0
            matrix[i, j] = similarity
            matrix[j, i] = similarity
    return matrix


def build_texts(group: pd.DataFrame) -> List[str]:
    title = group["title"].fillna("").astype(str)
    description = group["description"].fillna("").astype(str)
    return (title + " " + description).str.strip().tolist()


def text_similarity_matrix(
    texts: Sequence[str], vectorizer: "HashingVectorizer"
) -> np.ndarray:
    n = len(texts)
    if n == 0 or not any(text.strip() for text in texts):
        return np.zeros((n, n), dtype=np.float32)

    matrix = vectorizer.transform(texts)
    # HashingVectorizer 默认 norm='l2'，点积即余弦相似度。
    similarity = (matrix @ matrix.T).toarray().astype(np.float32)
    return np.clip(similarity, 0.0, 1.0)


def item_similarity_matrix(
    group: pd.DataFrame,
    args: argparse.Namespace,
    vectorizer: Optional["HashingVectorizer"],
) -> np.ndarray:
    category_sim = None
    text_sim = None

    if args.similarity_mode in {"category", "hybrid"}:
        category_sim = category_similarity_matrix(group["category"].tolist())

    if args.similarity_mode in {"text", "hybrid"}:
        if vectorizer is None:
            raise RuntimeError(
                "text/hybrid 模式需要 scikit-learn：pip install scikit-learn"
            )
        text_sim = text_similarity_matrix(build_texts(group), vectorizer)

    if args.similarity_mode == "category":
        return category_sim
    if args.similarity_mode == "text":
        return text_sim

    text_weight = args.text_similarity_weight
    return text_weight * text_sim + (1.0 - text_weight) * category_sim


def select_mmr_topk(
    group: pd.DataFrame,
    k: int,
    args: argparse.Namespace,
    vectorizer: Optional["HashingVectorizer"],
) -> List[int]:
    """
    在一个用户的 pCTR Top-M 内，逐个选择 MMR Top-K。

    每轮候选商品 i 的分数：
      relevance_weight * pctr_percentile(i)
      + history_diversity_weight * diversity(i)
      - redundancy_weight * max_similarity(i, 已选商品)

    第一个商品没有列表冗余惩罚。
    """
    n = len(group)
    if n == 0:
        return []

    similarity = item_similarity_matrix(group, args, vectorizer)
    relevance = group["pctr_percentile"].to_numpy(dtype=float)
    history_diversity = group["diversity_neutral_filled"].to_numpy(dtype=float)

    rel_weight = args.mmr_relevance_weight
    hist_weight = args.mmr_history_diversity_weight
    redundancy_weight = args.mmr_redundancy_weight

    selected_positions: List[int] = []
    remaining = np.ones(n, dtype=bool)
    max_similarity_to_selected = np.zeros(n, dtype=float)

    for _ in range(min(k, n)):
        scores = (
            rel_weight * relevance
            + hist_weight * history_diversity
            - redundancy_weight * max_similarity_to_selected
        )
        scores[~remaining] = -np.inf

        position = int(np.argmax(scores))
        selected_positions.append(position)
        remaining[position] = False

        max_similarity_to_selected = np.maximum(
            max_similarity_to_selected,
            similarity[:, position],
        )

    return selected_positions


def add_mmr_score(
    data: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    """
    生成完整 MMR 排序：
    1. pCTR Top-M 中通过 MMR 选出 Top-K；
    2. Top-M 中未被选中的商品按 final_score_weighted 排列；
    3. Top-M 之外按 pCTR 排列；
    4. 将完整名次映射为 final_score_mmr，数值越大排名越靠前。
    """
    data = data.copy()
    data["mmr_rank"] = np.nan

    vectorizer = None
    if args.similarity_mode in {"text", "hybrid"}:
        if HashingVectorizer is None:
            raise RuntimeError(
                "缺少 scikit-learn，请执行：pip install scikit-learn"
            )
        vectorizer = HashingVectorizer(
            n_features=2**14,
            alternate_sign=False,
            norm="l2",
            stop_words="english",
            lowercase=True,
        )

    total_users = data["user_id"].nunique()
    for user_number, (_, original_group) in enumerate(
        data.groupby("user_id", sort=False), start=1
    ):
        group = original_group.sort_values(
            "original_pctr_score", ascending=False, kind="mergesort"
        )
        top_m = group.head(min(args.top_m, len(group))).copy()

        selected_positions = select_mmr_topk(
            top_m, args.k, args, vectorizer
        )
        selected_indices = top_m.iloc[selected_positions].index.tolist()
        selected_set = set(selected_indices)

        remaining_top_m = top_m.loc[
            [idx for idx in top_m.index if idx not in selected_set]
        ].sort_values(
            ["final_score_weighted", "original_pctr_score"],
            ascending=[False, False],
            kind="mergesort",
        )

        outside_top_m = group.iloc[len(top_m):].sort_values(
            "original_pctr_score",
            ascending=False,
            kind="mergesort",
        )

        complete_order = (
            selected_indices
            + remaining_top_m.index.tolist()
            + outside_top_m.index.tolist()
        )
        ranks = np.arange(1, len(complete_order) + 1, dtype=float)
        data.loc[complete_order, "mmr_rank"] = ranks

        if user_number % 500 == 0 or user_number == total_users:
            print(f"MMR 进度：{user_number}/{total_users} 用户")

    # 对每个用户，将 rank=1 映射到 1，最后一名映射到接近 0。
    group_size = data.groupby("user_id", sort=False)["item_id"].transform("size")
    denominator = (group_size - 1).replace(0, 1)
    data["final_score_mmr"] = (
        1.0 - (data["mmr_rank"] - 1.0) / denominator
    ).clip(0.0, 1.0)

    return data


def get_topk(data: pd.DataFrame, score_column: str, k: int) -> pd.DataFrame:
    return (
        data.sort_values(
            ["user_id", score_column],
            ascending=[True, False],
            kind="mergesort",
        )
        .groupby("user_id", sort=False)
        .head(k)
    )


def evaluate_score(
    data: pd.DataFrame, score_column: str, k: int
) -> Dict[str, float]:
    if "is_positive" not in data.columns:
        raise ValueError("缺少 is_positive，无法执行离线评估")

    eval_data = data.loc[
        data[score_column].notna() & data["is_positive"].notna(),
        ["user_id", "is_positive", score_column],
    ].copy()
    eval_data["is_positive"] = eval_data["is_positive"].astype(int)

    topk = get_topk(eval_data, score_column, k)
    positive_total = eval_data.groupby("user_id", sort=False)["is_positive"].sum()
    valid_users = positive_total[positive_total > 0].index

    hits = topk.groupby("user_id", sort=False)["is_positive"].sum()
    recall = (
        hits.reindex(valid_users, fill_value=0)
        / positive_total.loc[valid_users]
    ).mean()

    precision = float(topk["is_positive"].mean()) if len(topk) else 0.0
    hit_rate = float(
        topk.groupby("user_id", sort=False)["is_positive"].max().mean()
    )

    ndcg_values: List[float] = []
    reciprocal_ranks: List[float] = []

    for _, group in eval_data.groupby("user_id", sort=False):
        ranked = group.sort_values(
            score_column, ascending=False, kind="mergesort"
        )
        relevances = ranked["is_positive"].to_numpy(dtype=int)[:k]
        positive_count = int(group["is_positive"].sum())

        if positive_count > 0:
            dcg = sum(
                rel / math.log2(rank + 2)
                for rank, rel in enumerate(relevances)
            )
            ideal_length = min(positive_count, k)
            idcg = sum(
                1.0 / math.log2(rank + 2)
                for rank in range(ideal_length)
            )
            ndcg_values.append(dcg / idcg if idcg > 0 else 0.0)

        positive_positions = np.flatnonzero(relevances == 1)
        reciprocal_ranks.append(
            1.0 / (int(positive_positions[0]) + 1)
            if len(positive_positions)
            else 0.0
        )

    return {
        "Precision": precision,
        "Recall": float(recall) if not pd.isna(recall) else 0.0,
        "NDCG": float(np.mean(ndcg_values)) if ndcg_values else 0.0,
        "HR": hit_rate,
        "MRR": float(np.mean(reciprocal_ranks))
        if reciprocal_ranks
        else 0.0,
    }


def evaluate_available_scores(
    data: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    candidates = [
        ("pCTR", "original_pctr_score"),
        ("旧融合", "final_combined_score"),
        ("统一加权", "final_score_weighted"),
        ("MMR", "final_score_mmr"),
    ]

    rows = []
    for name, column in candidates:
        if column not in data.columns:
            continue
        metrics = evaluate_score(data, column, args.k)
        rows.append({"strategy": name, "score_column": column, **metrics})

    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    validate_args(args)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}_rescored.csv")
    )
    report_path = output_path.with_name(f"{output_path.stem}_metrics.csv")
    config_path = output_path.with_name(f"{output_path.stem}_config.json")

    data, used_encoding = read_csv_with_fallback(
        input_path, encoding=args.encoding
    )
    data = prepare_data(data)
    data = add_calibrated_scores(data, args)
    data = add_weighted_score(data, args)

    if args.strategy in {"mmr", "both"}:
        data = add_mmr_score(data, args)

    # strategy=weighted 时只输出 weighted；
    # strategy=mmr 仍需要 weighted 作为 Top-M 剩余商品的稳定排序基线。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"新评分文件已保存：{output_path}")

    config = vars(args).copy()
    config["input_encoding"] = used_encoding
    config["rows"] = len(data)
    config["users"] = int(data["user_id"].nunique())
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
    print(f"配置已保存：{config_path}")

    if not args.skip_evaluation and "is_positive" in data.columns:
        metrics = evaluate_available_scores(data, args)
        metrics.to_csv(report_path, index=False, encoding="utf-8-sig")
        print("\n离线评估结果：")
        print(metrics.to_string(index=False))
        print(f"\n指标文件已保存：{report_path}")
    elif not args.skip_evaluation:
        print("未找到 is_positive，跳过离线评估。")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"\n执行失败：{error}", file=sys.stderr)
        sys.exit(1)
