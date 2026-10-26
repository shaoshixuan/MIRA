from pathlib import Path
import json

import pandas as pd

import run_all_experiments as exp


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CONFIG = json.loads(exp.CONFIG_PATH.read_text(encoding="utf-8"))


def category_set(values: pd.Series) -> set[str]:
    result: set[str] = set()
    for value in values.fillna("").astype(str):
        result.update(part.strip() for part in value.split("|") if part.strip())
    return result


def build_same_user_comparison() -> pd.DataFrame:
    ablation = pd.read_csv(RESULTS / "core_ablation_metrics.csv")
    rows = []
    primary_k = int(CONFIG["primary_k"])
    for dataset, paths in CONFIG["datasets"].items():
        scores = exp.calibrate(exp.normalize_scores(exp.read_csv(Path(paths["scores"]))))
        populations = {
            "all_users": scores,
            paths["subgroup_label"]: scores[scores["route"].eq(paths["subgroup_route"])],
        }
        for population, universe in populations.items():
            ours = ablation[
                (ablation.dataset == dataset)
                & (ablation.population == population)
                & (ablation.variant == "A4_full_route_aware")
                & (ablation.k == primary_k)
            ].iloc[0]
            rows.append({
                "dataset": dataset, "population": population,
                "method": "A4_full_route_aware", "users": int(ours.users),
                "recall": ours.recall, "ndcg": ours.ndcg,
                "category_coverage": ours.category_coverage,
                "ild": ours.ild, "novelty": ours.novelty,
            })
            total_categories = category_set(universe["category"])
            population_users = set(universe["user_id"])
            labels = universe[["user_id", "item_id", "category"]]
            for method, rec_path in paths["baselines"].items():
                filename = f"{dataset}_{population}_{method.replace('-', '_')}_k{primary_k}.csv"
                user_metrics = pd.read_csv(RESULTS / "per_user" / filename)
                rec = exp.read_csv(Path(rec_path))
                rec = rec[rec["k"].eq(primary_k)].copy()
                rec["user_id"] = rec["user_id"].astype(str).str.strip()
                rec["item_id"] = rec["item_id"].astype(str).str.strip()
                rec = rec[rec["user_id"].isin(population_users)].merge(
                    labels, on=["user_id", "item_id"], how="left"
                )
                covered = category_set(rec["category"])
                rows.append({
                    "dataset": dataset, "population": population, "method": method,
                    "users": len(user_metrics), "recall": user_metrics.recall.mean(),
                    "ndcg": user_metrics.ndcg.mean(),
                    "category_coverage": len(covered) / len(total_categories) if total_categories else 0,
                    "ild": user_metrics.ild.mean(), "novelty": user_metrics.novelty.mean(),
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "paper_comparison_k10.csv", index=False)
    return frame


def split_two_versions(comparison: pd.DataFrame) -> None:
    ablation = pd.read_csv(RESULTS / "core_ablation_metrics.csv")
    sensitivity = pd.read_csv(RESULTS / "route_weight_sensitivity_metrics.csv")

    ablation[ablation.population.eq("all_users")].to_csv(
        RESULTS / "all_users_core_ablation.csv", index=False
    )
    ablation[~ablation.population.eq("all_users")].to_csv(
        RESULTS / "target_subgroup_core_ablation.csv", index=False
    )
    sensitivity[sensitivity.population.eq("all_users")].to_csv(
        RESULTS / "all_users_route_weight_sensitivity.csv", index=False
    )
    sensitivity[~sensitivity.population.eq("all_users")].to_csv(
        RESULTS / "target_subgroup_route_weight_sensitivity.csv", index=False
    )
    comparison[comparison.population.eq("all_users")].to_csv(
        RESULTS / "all_users_same_user_comparison_k10.csv", index=False
    )
    comparison[~comparison.population.eq("all_users")].to_csv(
        RESULTS / "target_subgroup_same_user_comparison_k10.csv", index=False
    )


def build_scope_analysis() -> None:
    ablation = pd.read_csv(RESULTS / "core_ablation_metrics.csv")
    ours = ablation[
        (ablation.variant == "A4_full_route_aware")
        & (ablation.k == int(CONFIG["primary_k"]))
    ].copy()
    metrics = ["recall", "ndcg", "category_coverage", "ild", "novelty"]
    rows = []
    for dataset, group in ours.groupby("dataset"):
        all_row = group[group.population.eq("all_users")].iloc[0]
        subgroup_row = group[~group.population.eq("all_users")].iloc[0]
        row = {
            "dataset": dataset,
            "subgroup": subgroup_row.population,
            "all_users": int(all_row.users),
            "subgroup_users": int(subgroup_row.users),
        }
        for metric in metrics:
            row[f"all_{metric}"] = all_row[metric]
            row[f"subgroup_{metric}"] = subgroup_row[metric]
            row[f"delta_{metric}"] = subgroup_row[metric] - all_row[metric]
        rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS / "user_scope_analysis_k10.csv", index=False)


if __name__ == "__main__":
    comparison_frame = build_same_user_comparison()
    split_two_versions(comparison_frame)
    build_scope_analysis()
    print("paper outputs created")
