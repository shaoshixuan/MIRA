from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "llm_input" / "llm_input_data.csv"
HISTORY = ROOT / "llm_input" / "llm_user_history.json"
SHARDS = ROOT / "llm_shards"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()
    SHARDS.mkdir(exist_ok=True)
    data = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)
    data["user_id"] = data["user_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    histories = json.loads(HISTORY.read_text(encoding="utf-8"))
    users = data["user_id"].drop_duplicates().tolist()
    manifest = []
    for shard in range(args.shards):
        selected = set(users[shard::args.shards])
        shard_dir = SHARDS / f"shard_{shard:02d}"
        shard_dir.mkdir(exist_ok=True)
        frame = data[data["user_id"].isin(selected)].copy()
        input_path = shard_dir / "llm_input_data.csv"
        history_path = shard_dir / "llm_user_history.json"
        frame.to_csv(input_path, index=False, encoding="utf-8-sig")
        history_path.write_text(
            json.dumps({user: histories[user] for user in selected}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest.append({"shard": shard, "users": len(selected), "rows": len(frame), "input": str(input_path), "history": str(history_path)})
    (SHARDS / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
