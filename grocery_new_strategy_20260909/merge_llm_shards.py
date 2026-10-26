from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SHARDS = ROOT / "llm_shards"
OUTPUT = ROOT / "llm_output" / "llm_exploration_scores_complete.csv"


def main() -> None:
    files = []
    for shard in sorted(SHARDS.glob("shard_*")):
        candidates = sorted(shard.glob("llm_exploration_scores_*.csv"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No completed result in {shard}")
        files.append(candidates[-1])

    frames = [pd.read_csv(path, encoding="utf-8-sig", low_memory=False) for path in files]
    result = pd.concat(frames, ignore_index=True)
    for column in ("user_id", "item_id"):
        result[column] = result[column].astype(str).str.replace(r"\.0$", "", regex=True)

    duplicate_count = int(result.duplicated(["user_id", "item_id"]).sum())
    failed = ~result["scoring_method"].fillna("").astype(str).str.startswith(("LLM_direct_", "LLM_enhance_"))
    stats = {
        "files": [str(path) for path in files],
        "rows": int(len(result)),
        "users": int(result["user_id"].nunique()),
        "duplicate_keys": duplicate_count,
        "failed_or_fallback_rows": int(failed.sum()),
    }
    if stats["rows"] != 32386 or stats["users"] != 624 or duplicate_count or failed.any():
        raise RuntimeError(f"LLM shard validation failed: {stats}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(stats)
    print(OUTPUT)


if __name__ == "__main__":
    main()
