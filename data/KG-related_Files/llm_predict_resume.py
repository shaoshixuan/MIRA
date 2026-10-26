"""Safely resume the Toys LLM scoring run without modifying the original file.

The previous full CSV is treated as immutable input. Successful user-item
scores are loaded, failed/fallback users are retried, and each newly completed
user is appended to a compact journal. Re-running this script loads both the
previous full CSV and the journal, so successful API calls are never repeated.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from llm_predict_new import LLMExplorationScorer


DEFAULT_INPUT = (
    "D:/graduation_recsys/llm_datadeal_for_historyitem_descri/"
    "data_for_llm_rec/llm_input_data.csv"
)
DEFAULT_HISTORY = (
    "D:/graduation_recsys/llm_datadeal_for_historyitem_descri/"
    "data_for_llm_rec/llm_user_history.json"
)
DEFAULT_OUTPUT_DIR = "D:/graduation_recsys/data/KG-related_Files/llm_interest_scores"
DEFAULT_RESUME = (
    "D:/graduation_recsys/data/KG-related_Files/llm_interest_scores/"
    "llm_exploration_scores_20260807_212313.csv"
)
DEFAULT_JOURNAL = (
    "D:/graduation_recsys/data/KG-related_Files/llm_interest_scores/"
    "llm_resume_checkpoint_20260807.csv"
)


class BillingQuotaError(BaseException):
    """Escapes the original broad ``except Exception`` fallback handlers."""


class ResumableLLMExplorationScorer(LLMExplorationScorer):
    def __init__(self, *args, resume_from: str, checkpoint_path: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.resume_from = Path(resume_from)
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def normalize_id(value) -> str:
        text = str(value).strip()
        if re.fullmatch(r"-?\d+\.0", text):
            return text[:-2]
        return text

    @staticmethod
    def successful_method(method: str) -> bool:
        method = str(method)
        return method.startswith("LLM_direct_") or method.startswith("LLM_enhance_")

    def call_llm(self, prompt: str, max_tokens: int = 4096) -> str:
        try:
            return super().call_llm(prompt, max_tokens=max_tokens)
        except RuntimeError as error:
            message = str(error)
            if "30001" in message or "insufficient" in message.lower() or "计费或额度" in message:
                raise BillingQuotaError(message) from error
            raise

    def _load_successful_rows(self, path: Path) -> tuple[int, set[str]]:
        frame = pd.read_csv(
            path,
            usecols=["user_id", "item_id", "llm_exploration_score", "scoring_method"],
            encoding="utf-8-sig",
            on_bad_lines="skip",
        )
        loaded_rows = 0
        loaded_users: set[str] = set()
        for row in frame.itertuples(index=False):
            method = str(row.scoring_method)
            if not self.successful_method(method):
                continue
            user_id = self.normalize_id(row.user_id)
            item_id = self.normalize_id(row.item_id)
            self.exploration_scores[(user_id, item_id)] = float(row.llm_exploration_score)
            self.scoring_methods[(user_id, item_id)] = method
            loaded_rows += 1
            loaded_users.add(user_id)
        return loaded_rows, loaded_users

    def load_resume_scores(self) -> None:
        if not self.resume_from.exists():
            raise FileNotFoundError(f"找不到上次结果文件: {self.resume_from}")

        total_rows = 0
        total_users: set[str] = set()
        for path in [self.resume_from, self.checkpoint_path]:
            if not path.exists():
                continue
            rows, users = self._load_successful_rows(path)
            total_rows += rows
            total_users.update(users)
            print(f"已读取断点来源: {path}，成功行={rows}，成功用户={len(users)}")

        print(f"断点合并完成: 成功用户={len(total_users)}，成功行={total_rows}")

    def append_user_checkpoint(self, user_id: str, candidates: List[Dict]) -> None:
        records = []
        for candidate in candidates:
            item_id = candidate["item_id"]
            key = (user_id, item_id)
            records.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "llm_exploration_score": self.exploration_scores.get(key, 0.0),
                    "scoring_method": self.scoring_methods.get(key, "unknown"),
                }
            )
        write_header = not self.checkpoint_path.exists()
        pd.DataFrame(records).to_csv(
            self.checkpoint_path,
            mode="a",
            header=write_header,
            index=False,
            encoding="utf-8-sig" if write_header else "utf-8",
        )

    def calculate_all_scores(self) -> None:
        print("\n" + "=" * 60)
        print(f"续跑 LLM 探索分数 ({self.embedding_method})")
        print("=" * 60)

        started = time.time()
        users = list(self.user_candidates.keys())
        pending_users = []
        original_positions = {}
        for position, user_id in enumerate(users, start=1):
            candidates = self.user_candidates[user_id]
            completed = all(
                self.successful_method(
                    self.scoring_methods.get((user_id, candidate["item_id"]), "")
                )
                for candidate in candidates
            )
            if not completed:
                pending_users.append(user_id)
                original_positions[user_id] = position

        print(f"原始用户数: {len(users)}")
        print(f"跳过已成功用户: {len(users) - len(pending_users)}")
        print(f"本次需重试用户: {len(pending_users)}")
        if pending_users:
            preview = [
                f"#{original_positions[user]} user={user}" for user in pending_users[:5]
            ]
            print("前几个待处理用户: " + ", ".join(preview))

        for user_id in tqdm(pending_users, desc="续跑计算分数"):
            candidates = self.user_candidates[user_id]
            kg_score = candidates[0].get("kg_interest_score", 0) if candidates else 0
            is_low = kg_score == -10

            if is_low:
                recommendations = self.get_llm_recommendations(user_id)
                for candidate in candidates:
                    item_id = candidate["item_id"]
                    if recommendations and user_id in self.llm_embeddings_cache:
                        text = f"{candidate.get('title', '')} {candidate.get('description', '')}"
                        score = self.calc_similarity(
                            text, self.llm_embeddings_cache[user_id]
                        ) * 0.3
                        method = f"LLM_direct_{self.embedding_method}"
                    else:
                        score = 0.0
                        method = "LLM_failed"
                    self.exploration_scores[(user_id, item_id)] = score
                    self.scoring_methods[(user_id, item_id)] = method
                    self.stats["llm_direct_scored"] += 1
            else:
                descriptions = [candidate.get("description", "") for candidate in candidates]
                keywords = self.get_llm_enhancement(user_id, descriptions)
                cache_key = f"{user_id}_enh"
                for candidate in candidates:
                    item_id = candidate["item_id"]
                    if keywords and cache_key in self.llm_embeddings_cache:
                        text = f"{candidate.get('title', '')} {candidate.get('description', '')}"
                        score = self.calc_similarity(
                            text, self.llm_embeddings_cache[cache_key]
                        ) * 0.1
                        method = f"LLM_enhance_{self.embedding_method}"
                    else:
                        score = candidate.get("original_pctr_score", 0.1) * 0.1
                        method = "LLM_fallback"
                    self.exploration_scores[(user_id, item_id)] = score
                    self.scoring_methods[(user_id, item_id)] = method
                    self.stats["llm_enhanced_scored"] += 1

            # One API call/user is now durable before moving to the next user.
            self.append_user_checkpoint(user_id, candidates)

        self.stats["processing_time"] = time.time() - started
        print(f"续跑完成，耗时: {self.stats['processing_time']:.1f}s")
        print(f"本次 API 调用: {self.stats['api_calls']}")

    def run(self) -> None:
        try:
            self.load_data()
            self.load_resume_scores()
            self.calculate_all_scores()
            output = self.save_results()
            print(f"\n全部完成，合并结果: {output}")
            print(f"断点日志保留在: {self.checkpoint_path}")
        except BillingQuotaError as error:
            print("\n检测到 SiliconFlow 余额/额度不足，已安全停止。")
            print(f"错误: {error}")
            print(f"已完成用户保存在: {self.checkpoint_path}")
            print("充值后重新运行同一条命令即可继续，不会重复已成功用户。")
            raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume the interrupted Toys LLM run")
    parser.add_argument("--resume-from", default=DEFAULT_RESUME)
    parser.add_argument("--checkpoint-path", default=DEFAULT_JOURNAL)
    parser.add_argument("--llm-input-csv", default=DEFAULT_INPUT)
    parser.add_argument("--user-history-json", default=DEFAULT_HISTORY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scorer = ResumableLLMExplorationScorer(
        llm_input_csv=args.llm_input_csv,
        user_history_json=args.user_history_json,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        checkpoint_path=args.checkpoint_path,
    )
    scorer.run()


if __name__ == "__main__":
    main()
