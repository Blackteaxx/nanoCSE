#!/usr/bin/env python3
"""
将 Open-AgentRL parquet 中的 MATH 样本转换为 nanoCSE 单实例 JSON 文件。

用法:
    python scripts/prepare_math_instances.py \
        --parquet data/Gen-Verse/Open-AgentRL-30K/Open-AgentRL-30K-cleaned-math.parquet \
        --output nanoCSE/instances/math/ \
        --limit 50

每个输出 JSON 格式:
    {
        "instance_id": "math_dapo_00042",
        "problem": "...",
        "ground_truth": "42",
        "data_source": "math_dapo",
        "ability": "MATH"
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _to_python(obj):
    """Convert numpy/parquet types to plain Python."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return [_to_python(x) for x in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(x) for x in obj]
    return obj


def extract_problem(prompt_field) -> str:
    """从 prompt 字段提取问题文本。"""
    prompt_field = _to_python(prompt_field)
    if isinstance(prompt_field, list) and prompt_field:
        first = prompt_field[0]
        if isinstance(first, dict):
            return first.get("content", "")
        return str(first)
    if isinstance(prompt_field, str):
        return prompt_field
    return ""


def extract_ground_truth(reward_model_field) -> str:
    """从 reward_model 字段提取 ground_truth。"""
    reward_model_field = _to_python(reward_model_field)
    if isinstance(reward_model_field, dict):
        return str(reward_model_field.get("ground_truth", ""))
    if isinstance(reward_model_field, str):
        try:
            rm = json.loads(reward_model_field)
            return str(rm.get("ground_truth", ""))
        except (json.JSONDecodeError, TypeError):
            return reward_model_field
    return ""


def main():
    parser = argparse.ArgumentParser(description="将 Open-AgentRL MATH 数据转为 nanoCSE 实例 JSON")
    parser.add_argument("--parquet", required=True, help="输入 parquet 文件路径")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--limit", type=int, default=None, help="最多转换 N 个样本")
    parser.add_argument("--data-sources", nargs="*", default=None,
                        help="只包含指定 data_source（默认全部 MATH 类）")
    parser.add_argument("--shuffle", action="store_true", help="随机打乱后再截取")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.parquet)
    print(f"读入 {len(df)} 行, columns={df.columns.tolist()}")

    if args.data_sources:
        df = df[df["data_source"].isin(args.data_sources)]
        print(f"过滤 data_source={args.data_sources} 后剩 {len(df)} 行")

    if args.shuffle:
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    if args.limit is not None and args.limit > 0:
        df = df.head(args.limit)
        print(f"截取前 {args.limit} 行")

    count = 0
    for idx, row in df.iterrows():
        ds = row.get("data_source", "unknown")
        problem = extract_problem(row.get("prompt"))
        gt = extract_ground_truth(row.get("reward_model"))
        ability = row.get("ability", "MATH")

        if not problem or not gt:
            continue

        ei = _to_python(row.get("extra_info", {}))
        if isinstance(ei, str):
            try:
                ei = json.loads(ei)
            except (json.JSONDecodeError, TypeError):
                ei = {}
        raw_idx = ei.get("index", idx) if isinstance(ei, dict) else idx
        instance_id = f"{ds}_{int(raw_idx):05d}"

        instance = {
            "instance_id": instance_id,
            "problem": problem,
            "ground_truth": gt,
            "data_source": ds,
            "ability": ability,
        }

        out_path = out_dir / f"{instance_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(instance, f, ensure_ascii=False, indent=2)

        count += 1

    print(f"完成: 写入 {count} 个实例到 {out_dir}")


if __name__ == "__main__":
    main()
