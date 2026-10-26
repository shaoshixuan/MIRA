from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OLD_ROOT = Path(r"D:\graduation_recsys\supplementary_experiments_dataset_adaptive_20260827")
NEW_GROCERY = Path(r"D:\graduation_recsys\grocery_new_strategy_20260909")
sys.path.insert(0, str(OLD_ROOT))
sys.path.insert(0, str(NEW_GROCERY))
import run_all_experiments as common
import run_grocery_pipeline as grocery

ABLATION = ROOT / "01_ablation"
PARAMETER = ROOT / "02_parameter"
SUBGROUP = ROOT / "03_user_subgroup"
for directory in (ABLATION, PARAMETER, SUBGROUP):
    directory.mkdir(parents=True, exist_ok=True)

METRICS = ["recall", "ndcg", "category_coverage", "ild", "novelty"]
VARIANT_MAP = {
    "A0_pCTR_only": "A0_pCTR_only",
    "A1_pCTR_KG": "A1_pCTR_KG",
    "A1_pCTR_KG_route_aware": "A1_pCTR_KG",
    "A2_pCTR_LLM": "A2_pCTR_LLM",
    "A2_pCTR_LLM_route_aware": "A2_pCTR_LLM",
    "A3_simple_KG_LLM": "A3_fixed_weight_KG_LLM",
    "A3_KG_LLM_global_weight": "A3_fixed_weight_KG_LLM",
    "A4_full_route_aware": "A4_full_route_aware",
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def build_ablation() -> pd.DataFrame:
    old = read_csv(OLD_ROOT / "results" / "core_ablation_metrics.csv")
    old = old[old["dataset"].isin(["automotive", "toys"])].copy()

    new = read_csv(NEW_GROCERY / "results" / "ablation_and_final_metrics.csv")
    new = new.dropna(subset=["scope", "variant", "k"]).copy()
    new.insert(0, "dataset", "grocery")
    new = new.rename(columns={"scope": "population"})
    new = new[new["population"].isin(["all_users", "low_interaction_users", "kg_only_users"])]
    new = new[["dataset", "population", "variant", "k", "users", *METRICS]]

    combined = pd.concat([old, new], ignore_index=True)
    combined["variant_source_name"] = combined["variant"]
    combined["variant"] = combined["variant"].map(VARIANT_MAP)
    combined["k"] = combined["k"].astype(int)
    combined["users"] = combined["users"].astype(int)
    combined = combined.sort_values(["dataset", "population", "variant", "k"])
    combined.to_csv(ABLATION / "core_ablation_metrics.csv", index=False, encoding="utf-8-sig")
    return combined


def grocery_scores() -> tuple[pd.DataFrame, pd.Series, dict[str, float]]:
    merged = read_csv(NEW_GROCERY / "processed" / "grocery_new_strategy_merged_scores.csv")
    scores = common.calibrate(common.normalize_scores(merged))
    scores["route"] = merged["applied_route"].to_numpy()
    popularity = read_csv(NEW_GROCERY / "processed" / "history_item_popularity.csv")
    if "item_id" not in popularity.columns:
        popularity = popularity.rename(columns={popularity.columns[0]: "item_id"})
    popularity["item_id"] = popularity["item_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    pop = popularity.set_index("item_id")["popularity"]
    selected = json.loads((NEW_GROCERY / "results" / "selected_weights_and_protocol.json").read_text(encoding="utf-8"))["A4_route_weights"]
    return scores, pop, {key: float(value) for key, value in selected.items()}


def build_parameter() -> pd.DataFrame:
    old = read_csv(OLD_ROOT / "results" / "route_weight_sensitivity_metrics.csv")
    old = old[old["dataset"].isin(["automotive", "toys"])].copy()
    old["source_split"] = "existing evaluation population"

    scores, popularity, selected = grocery_scores()
    scopes = {
        "all_users": scores,
        "low_interaction_users": scores[scores["route"].eq("LOW_INTERACTION")],
        "kg_only_users": scores[scores["route"].eq("KG_ONLY")],
    }
    records = []
    grid = np.round(np.arange(0.0, 1.01, 0.1), 1)
    ks = [5, 10, 15, 20, 25]
    for population, subset in scopes.items():
        if population == "low_interaction_users":
            routes = ["LOW_INTERACTION"]
        elif population == "kg_only_users":
            routes = ["KG_ONLY"]
        else:
            routes = ["KG_ONLY", "LOW_INTERACTION"]
        for route in routes:
            for weight in grid:
                weights = dict(selected)
                weights[route] = float(weight)
                values = grocery.score_corrected_variant(subset, "A4_full_route_aware", weights, 0.0)
                metrics, _ = common.evaluate(subset, values, ks, popularity)
                for row in metrics.to_dict("records"):
                    records.append({
                        "dataset": "grocery",
                        "population": population,
                        "varied_route": route,
                        "lambda": float(weight),
                        "selected_weight": selected.get(route, 0.0),
                        **row,
                        "source_split": "new Grocery full population",
                    })
    new = pd.DataFrame(records)
    columns = ["dataset", "population", "varied_route", "lambda", "selected_weight", "k", "users", *METRICS, "source_split"]
    combined = pd.concat([old[columns], new[columns]], ignore_index=True)
    combined["k"] = combined["k"].astype(int)
    combined["users"] = combined["users"].astype(int)
    combined = combined.sort_values(["dataset", "population", "varied_route", "lambda", "k"])
    combined.to_csv(PARAMETER / "route_weight_sensitivity_metrics.csv", index=False, encoding="utf-8-sig")
    new.to_csv(PARAMETER / "grocery_route_weight_sensitivity_metrics.csv", index=False, encoding="utf-8-sig")
    return combined


def build_selected_weights() -> pd.DataFrame:
    old_weights = json.loads((OLD_ROOT / "results" / "selected_route_weights.json").read_text(encoding="utf-8"))
    new_weights = json.loads((NEW_GROCERY / "results" / "selected_weights_and_protocol.json").read_text(encoding="utf-8"))["A4_route_weights"]
    rows = []
    for dataset in ("automotive", "toys"):
        for route, weight in old_weights[dataset].items():
            rows.append({"dataset": dataset, "route": route, "selected_weight": weight, "source": "20260827 existing result"})
    for route, weight in new_weights.items():
        rows.append({"dataset": "grocery", "route": route, "selected_weight": weight, "source": "20260909 new Grocery result"})
    result = pd.DataFrame(rows).sort_values(["dataset", "route"])
    result.to_csv(PARAMETER / "selected_route_weights.csv", index=False, encoding="utf-8-sig")
    return result


def build_subgroups(ablation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    a4 = ablation[ablation["variant"].eq("A4_full_route_aware")].copy()
    grocery_detail = a4[(a4["dataset"].eq("grocery")) & (a4["population"].isin(["all_users", "low_interaction_users", "kg_only_users"]))]
    grocery_detail.to_csv(SUBGROUP / "grocery_user_subgroup_metrics.csv", index=False, encoding="utf-8-sig")

    target = {"automotive": "llm_enhancement_users", "grocery": "low_interaction_users", "toys": "llm_enhancement_users"}
    rows = []
    for dataset, subgroup in target.items():
        for k in sorted(a4[a4["dataset"].eq(dataset)]["k"].unique()):
            all_row = a4[(a4["dataset"].eq(dataset)) & (a4["population"].eq("all_users")) & (a4["k"].eq(k))]
            sub_row = a4[(a4["dataset"].eq(dataset)) & (a4["population"].eq(subgroup)) & (a4["k"].eq(k))]
            if all_row.empty or sub_row.empty:
                continue
            all_row, sub_row = all_row.iloc[0], sub_row.iloc[0]
            record = {"dataset": dataset, "subgroup": subgroup, "k": int(k), "all_users": int(all_row["users"]), "subgroup_users": int(sub_row["users"])}
            for metric in METRICS:
                record[f"all_{metric}"] = float(all_row[metric])
                record[f"subgroup_{metric}"] = float(sub_row[metric])
                record[f"delta_{metric}"] = float(sub_row[metric] - all_row[metric])
            rows.append(record)
    comparison = pd.DataFrame(rows).sort_values(["dataset", "k"])
    comparison.to_csv(SUBGROUP / "three_dataset_user_subgroup_comparison.csv", index=False, encoding="utf-8-sig")
    return grocery_detail, comparison


def plot_subgroups(comparison: pd.DataFrame) -> None:
    data = comparison[comparison["k"].eq(10)].copy().set_index("dataset").loc[["automotive", "grocery", "toys"]].reset_index()
    labels = ["Automotive", "Grocery", "Toys"]
    titles = ["Recall@10", "NDCG@10", "Category Coverage@10", "ILD@10", "Novelty@10"]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), dpi=160)
    axes = axes.ravel()
    x = np.arange(3)
    width = 0.36
    colors = ("#4C78A8", "#F58518")
    for ax, metric, title in zip(axes[:5], METRICS, titles):
        all_values = data[f"all_{metric}"].to_numpy(float)
        subgroup_values = data[f"subgroup_{metric}"].to_numpy(float)
        ax.bar(x - width / 2, all_values, width, label="All users", color=colors[0])
        ax.bar(x + width / 2, subgroup_values, width, label="Target subgroup", color=colors[1])
        ax.set_title(title, fontsize=11, fontweight="semibold")
        ax.set_xticks(x, labels, rotation=15)
        ax.grid(axis="y", alpha=0.22, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        if metric != "novelty":
            ax.set_ylim(bottom=0)
    axes[5].axis("off")
    axes[5].text(0.04, 0.82, "Target subgroups", fontsize=12, fontweight="semibold", va="top")
    axes[5].text(0.04, 0.63, "Automotive: LLM-enhancement users\nGrocery: low-interaction users\nToys: LLM-enhancement users", fontsize=10.5, linespacing=1.6, va="top")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("A4 performance for all users and target subgroups (K=10)", y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(SUBGROUP / "three_dataset_user_subgroup_comparison_k10.png", dpi=300, bbox_inches="tight")
    fig.savefig(SUBGROUP / "three_dataset_user_subgroup_comparison_k10.svg", bbox_inches="tight")
    plt.close(fig)


def plot_grocery_parameter(parameter: pd.DataFrame) -> None:
    data = parameter[(parameter["dataset"].eq("grocery")) & (parameter["population"].eq("all_users")) & (parameter["k"].eq(15))]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), dpi=160)
    axes = axes.ravel()
    palette = {"KG_ONLY": "#4C78A8", "LOW_INTERACTION": "#F58518"}
    for ax, metric, title in zip(axes[:5], METRICS, ["Recall@15", "NDCG@15", "Category Coverage@15", "ILD@15", "Novelty@15"]):
        for route, group in data.groupby("varied_route"):
            ax.plot(group["lambda"], group[metric], marker="o", label=route, color=palette[route], linewidth=2)
        ax.set_title(title, fontsize=11, fontweight="semibold")
        ax.set_xlabel("Exploration weight")
        ax.grid(alpha=0.22, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[5].axis("off")
    axes[5].text(0.04, 0.82, "Selected weights", fontsize=12, fontweight="semibold", va="top")
    axes[5].text(0.04, 0.63, "KG_ONLY: 0.0\nLOW_INTERACTION: 0.1\nSelection objective:\n0.6 NDCG@15 + 0.4 Recall@15", fontsize=10.5, linespacing=1.55, va="top")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("Grocery route-weight sensitivity (all users, K=15)", y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(PARAMETER / "grocery_parameter_sensitivity_k15.png", dpi=300, bbox_inches="tight")
    fig.savefig(PARAMETER / "grocery_parameter_sensitivity_k15.svg", bbox_inches="tight")
    plt.close(fig)


def write_analysis(comparison: pd.DataFrame, grocery_detail: pd.DataFrame) -> None:
    k10 = comparison[comparison["k"].eq(10)].set_index("dataset")
    lines = [
        "# 用户子组实验简要分析",
        "",
        "图中比较完整方法 A4 在全体用户与目标子组上的 K=10 表现。Automotive 和 Toys 的目标子组为需要 LLM 增强的用户，Grocery 的目标子组为低交互用户。不同数据集的子组含义并不完全相同，因此应分别解释，不能把橙色柱统一称为低交互用户。",
        "",
    ]
    for dataset, name in [("automotive", "Automotive"), ("grocery", "Grocery"), ("toys", "Toys")]:
        row = k10.loc[dataset]
        lines.append(
            f"- {name}：子组相对全体用户的 Recall 变化为 {row['delta_recall']:+.6f}，NDCG 变化为 {row['delta_ndcg']:+.6f}，"
            f"Coverage 变化为 {row['delta_category_coverage']:+.6f}，ILD 变化为 {row['delta_ild']:+.6f}，Novelty 变化为 {row['delta_novelty']:+.6f}。"
        )
    lines += [
        "",
        "总体上，三个目标子组的 Recall 均高于各自全体用户。Grocery 低交互用户的 Recall 和 NDCG 提升最明显，说明新候选级 LLM 路由的准确性收益主要集中在低交互人群；但其 Coverage 和 ILD 低于全体用户，不能据此声称子组在所有多样性指标上占优。Automotive 的 LLM 增强子组表现为 Recall 和 Novelty 更高、NDCG 略低；Toys 子组则同时提高 Recall、NDCG 和 Novelty，但 Coverage 与 ILD 略低。",
        "",
        "论文中更稳妥的结论是：路由机制识别出了响应模式不同的用户群体，子组收益主要体现在准确性与新颖性，而类别覆盖和列表内多样性仍存在数据集差异。",
    ]
    (SUBGROUP / "USER_SUBGROUP_ANALYSIS_CN.md").write_text("\n".join(lines), encoding="utf-8")


def write_parameter_analysis(parameter: pd.DataFrame) -> None:
    data = parameter[(parameter["dataset"].eq("grocery")) & (parameter["population"].eq("all_users")) & (parameter["k"].eq(15))]
    kg0 = data[(data["varied_route"].eq("KG_ONLY")) & (data["lambda"].eq(0.0))].iloc[0]
    kg1 = data[(data["varied_route"].eq("KG_ONLY")) & (data["lambda"].eq(1.0))].iloc[0]
    low0 = data[(data["varied_route"].eq("LOW_INTERACTION")) & (data["lambda"].eq(0.0))].iloc[0]
    low1 = data[(data["varied_route"].eq("LOW_INTERACTION")) & (data["lambda"].eq(0.1))].iloc[0]
    text = f"""# Grocery 参数实验简要分析

参数实验分别改变 `KG_ONLY` 与 `LOW_INTERACTION` 路由的探索权重，其他路由保持已选择权重，报告全体用户及对应子组在 K=5/10/15/20/25 上的结果。

- `KG_ONLY` 的验证选择为 0.0。在全体用户 K=15 上，权重由 0.0 增至 1.0 时，Recall 从 {kg0.recall:.6f} 降至 {kg1.recall:.6f}，NDCG 从 {kg0.ndcg:.6f} 降至 {kg1.ndcg:.6f}；Coverage 与 Novelty 上升。这说明当前 Grocery KG 分数具有明显的准确性—多样性权衡，验证目标不支持在该路由加入探索权重。
- `LOW_INTERACTION` 的验证选择为 0.1。相较权重 0.0，权重 0.1 在全体用户 K=15 上将 Recall 从 {low0.recall:.6f} 提高至 {low1.recall:.6f}，NDCG 从 {low0.ndcg:.6f} 提高至 {low1.ndcg:.6f}，说明低强度 LLM 探索能够改善低交互用户，并转化为小幅整体收益。
- 当低交互权重继续增大时，准确性整体下降、Coverage 和 Novelty 上升。因此论文中应将 0.1 表述为验证集准确性目标下的折中点，而不是普遍最优的多样性权重。
"""
    (PARAMETER / "GROCERY_PARAMETER_ANALYSIS_CN.md").write_text(text, encoding="utf-8")


def write_readme() -> None:
    text = """# 三数据集补充实验结果

本目录统一保存 Grocery 新结果以及 Automotive、Toys 的既有结果。

- `01_ablation/core_ablation_metrics.csv`：三个数据集的核心消融结果。
- `02_parameter/grocery_route_weight_sensitivity_metrics.csv`：Grocery 新结果的参数实验。
- `02_parameter/route_weight_sensitivity_metrics.csv`：三个数据集的统一参数结果。
- `02_parameter/selected_route_weights.csv`：各数据集、各路由的最终权重。
- `03_user_subgroup/grocery_user_subgroup_metrics.csv`：Grocery 的全体、低交互和 KG-only 用户结果。
- `03_user_subgroup/three_dataset_user_subgroup_comparison.csv`：三个数据集的目标子组对比数据。
- `03_user_subgroup/three_dataset_user_subgroup_comparison_k10.png`：用户子组可视化。
- `03_user_subgroup/USER_SUBGROUP_ANALYSIS_CN.md`：简短文字分析。

注意：Grocery 使用 2026-09-09 重新处理后的候选级 LLM/KG 路由结果；Automotive 和 Toys 使用 2026-08-27 的既有结果。Grocery 没有触发 `MIXED_ENHANCEMENT` 路由。
"""
    (ROOT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    ablation = build_ablation()
    parameter = build_parameter()
    build_selected_weights()
    grocery_detail, comparison = build_subgroups(ablation)
    plot_subgroups(comparison)
    plot_grocery_parameter(parameter)
    write_analysis(comparison, grocery_detail)
    write_parameter_analysis(parameter)
    write_readme()
    print(json.dumps({
        "ablation_rows": len(ablation),
        "parameter_rows": len(parameter),
        "subgroup_comparison_rows": len(comparison),
        "output": str(ROOT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
