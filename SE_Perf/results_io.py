"""
结果 I/O 与汇总模块

负责将 AgentResult 写入文件、汇总所有迭代预测结果、
生成最终 final.json，以及 Token 使用统计。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from run_models import PredictionEntry

from perfagent.protocols import AgentResult


# ---------------------------------------------------------------------------
# 单迭代写入
# ---------------------------------------------------------------------------


def write_iteration_preds_from_result(
    result: AgentResult,
    iteration_dir: Path,
    logger,
) -> Path | None:
    """从 AgentResult 直接生成 preds.json（单实例）。"""
    try:
        metric_val = result.metric
        is_finite = False
        try:
            is_finite = metric_val is not None and math.isfinite(float(metric_val))
        except (ValueError, TypeError):
            is_finite = False
        is_success = bool(result.success) and result.error is None
        entry = PredictionEntry(
            solution=result.solution,
            metric=metric_val,
            success=is_success and is_finite,
            artifacts=result.artifacts or {},
        )

        predictions = {str(result.instance_id): entry.to_dict()}

        preds_path = iteration_dir / "preds.json"
        with open(preds_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)

        logger.info(f"已生成迭代预测汇总: {preds_path}")
        return preds_path
    except Exception as e:
        logger.warning(f"生成 preds.json 失败: {e}")
        return None


def write_result_json(result: AgentResult, output_dir: Path, logger) -> Path | None:
    """将 AgentResult 写入 result.json（供 TrajExtractor 等下游工具读取）。"""
    try:
        result_path = output_dir / "result.json"
        result.to_json(result_path)
        logger.info(f"已写入 result.json: {result_path}")
        return result_path
    except Exception as e:
        logger.warning(f"写入 result.json 失败: {e}")
        return None


# ---------------------------------------------------------------------------
# 全局汇总
# ---------------------------------------------------------------------------


def aggregate_all_iterations_preds(root_output_dir: Path, logger) -> Path | None:
    """汇总所有 iteration_* 目录下的 preds.json 到根目录。"""
    aggregated_data: dict[str, list[dict]] = {}

    try:
        iteration_dirs = sorted(root_output_dir.glob("iteration_*"), key=lambda p: p.name)

        for iter_dir in iteration_dirs:
            if not iter_dir.is_dir():
                continue
            try:
                iter_num = int(iter_dir.name.split("_")[-1])
            except ValueError:
                continue

            preds_file = iter_dir / "preds.json"
            if not preds_file.exists():
                continue

            try:
                with open(preds_file, encoding="utf-8") as f:
                    preds = json.load(f)
            except Exception:
                continue

            for instance_id, info in preds.items():
                try:
                    entry = {
                        "iteration": iter_num,
                        "solution": info.get("solution", ""),
                        "metric": info.get("metric"),
                        "success": info.get("success", False),
                        "artifacts": info.get("artifacts"),
                    }
                    aggregated_data.setdefault(str(instance_id), []).append(entry)
                except Exception:
                    continue

        agg_path = root_output_dir / "preds.json"
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(aggregated_data, f, indent=2, ensure_ascii=False)

        if logger:
            logger.info(f"汇总所有迭代预测结果: {agg_path}")
        return agg_path
    except Exception as e:
        if logger:
            logger.warning(f"汇总 preds.json 失败: {e}")
        return None


def write_final_json_from_preds(
    aggregated_preds_path: Path,
    root_output_dir: Path,
    logger,
    *,
    higher_is_better: bool = False,
) -> Path | None:
    """从汇总的 preds.json 中选择最佳结果写入 final.json。

    Args:
        higher_is_better: True 时取 metric 最大的条目，False 时取最小的。
    """
    try:
        with open(aggregated_preds_path, encoding="utf-8") as f:
            aggregated_data = json.load(f)
    except Exception as e:
        logger.warning(f"读取汇总 preds.json 失败: {e}")
        return None

    # 无效/缺失 metric 的哨兵值：始终为"最差"
    _sentinel = -float("inf") if higher_is_better else float("inf")

    def _parse_metric(val: Any) -> float:
        try:
            if val is None:
                return _sentinel
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                lowered = val.strip().lower()
                if lowered in ("inf", "infinity", "nan"):
                    return _sentinel
                return float(val)
            return _sentinel
        except Exception:
            return _sentinel

    _select_best = max if higher_is_better else min

    final_result_map: dict[str, str] = {}
    try:
        for instance_id, entries in aggregated_data.items():
            if not isinstance(entries, list) or not entries:
                continue
            try:
                best_entry = _select_best(entries, key=lambda e: _parse_metric(e.get("metric")))
            except ValueError:
                continue
            final_result_map[str(instance_id)] = best_entry.get("solution", "") or ""

        final_path = root_output_dir / "final.json"
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(final_result_map, f, indent=2, ensure_ascii=False)

        if logger:
            logger.info(f"生成最终结果 final.json: {final_path}")
        return final_path
    except Exception as e:
        if logger:
            logger.warning(f"生成 final.json 失败: {e}")
        return None


# ---------------------------------------------------------------------------
# Token 统计与最终摘要
# ---------------------------------------------------------------------------


def log_token_usage(output_dir, logger):
    """统计并记录 Token 使用情况。"""
    token_log_file = Path(output_dir) / "token_usage.jsonl"
    if not token_log_file.exists():
        return

    total_prompt = 0
    total_completion = 0
    total = 0
    by_context: dict[str, dict[str, int]] = {}

    try:
        with open(token_log_file, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    pt = int(rec.get("prompt_tokens") or 0)
                    ct = int(rec.get("completion_tokens") or 0)
                    tt = int(rec.get("total_tokens") or (pt + ct))
                    ctx = str(rec.get("context") or "unknown")

                    total_prompt += pt
                    total_completion += ct
                    total += tt

                    agg = by_context.setdefault(ctx, {"prompt": 0, "completion": 0, "total": 0})
                    agg["prompt"] += pt
                    agg["completion"] += ct
                    agg["total"] += tt
                except Exception:
                    continue

        print("\nToken 使用统计:")
        print(f"  Total: {total} (Prompt: {total_prompt}, Completion: {total_completion})")
        if by_context:
            print("  按上下文分类:")
            for ctx, vals in by_context.items():
                print(f"    - {ctx}: prompt={vals['prompt']}, completion={vals['completion']}, total={vals['total']}")

        logger.info(
            json.dumps(
                {
                    "token_usage_total": {"prompt": total_prompt, "completion": total_completion, "total": total},
                    "by_context": by_context,
                    "token_log_file": str(token_log_file),
                },
                ensure_ascii=False,
            )
        )
    except Exception:
        pass


def print_final_summary(timestamp, log_file, output_dir, traj_pool_manager, logger,
                        *, higher_is_better: bool = False):
    """打印和记录最终执行摘要。"""
    logger.info("所有任务执行完成")
    print("\n执行完成")
    print(f"  日志: {log_file}")
    print(f"  输出: {output_dir}")

    try:
        root_dir = Path(output_dir)
        agg_path = aggregate_all_iterations_preds(root_dir, logger)
        if agg_path:
            write_final_json_from_preds(agg_path, root_dir, logger, higher_is_better=higher_is_better)
    except Exception as e:
        logger.warning(f"生成最终结果文件失败: {e}")

    log_token_usage(output_dir, logger)
