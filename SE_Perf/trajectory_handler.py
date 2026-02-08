"""
轨迹池处理模块

负责从 AgentResult 构建轨迹数据、更新轨迹池、
以及迭代后处理（.tra 生成 + 轨迹池汇总）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.utils.se_logger import get_se_logger
from core.utils.traj_pool_manager import TrajPoolManager
from core.utils.trajectory_processor import TrajectoryProcessor
from perf_config import SEPerfRunSEConfig, StepConfig
from run_models import TrajectoryData

from perfagent.protocols import AgentResult

logger = get_se_logger("trajectory_handler", emoji="📊")


# ---------------------------------------------------------------------------
# 轨迹构建
# ---------------------------------------------------------------------------


def build_trajectory_from_result(
    result: AgentResult,
    instance_name: str,
    problem_description: str,
    iteration_index: int,
    label: str,
    source_labels: list[str],
    operator_name: str | None,
    output_dir: Path,
) -> TrajectoryData:
    """直接从 AgentResult 构建轨迹数据，避免文件 I/O 中转。
    优先使用已生成的 .tra 文件内容作为 trajectory_content，与磁盘一致且为 JSON 可解析；
    仅当 .tra 不存在或读取失败时，才从 optimization_history 拼摘要。
    """
    tra_path = Path(output_dir) / f"{instance_name}.tra"
    trajectory_content = ""
    if tra_path.exists():
        try:
            trajectory_content = tra_path.read_text(encoding="utf-8").strip()
        except Exception:
            trajectory_content = ""

    return TrajectoryData(
        label=label,
        instance_name=instance_name,
        problem_description=problem_description,
        trajectory_content=trajectory_content,
        solution=result.solution or "",
        metric=result.metric,
        artifacts=result.artifacts or {},
        iteration=iteration_index,
        source_dir=str(output_dir),
        source_entry_labels=list(source_labels or []),
        operator_name=str(operator_name) if operator_name else None,
    )


# ---------------------------------------------------------------------------
# 轨迹池更新
# ---------------------------------------------------------------------------


def update_pool_from_result(
    result: AgentResult,
    instance_name: str,
    problem_description: str,
    iteration_index: int,
    traj_pool_manager: TrajPoolManager,
    se_cfg: SEPerfRunSEConfig,
    run_logger,
    label_prefix: str | None = None,
    source_labels: list[str] | None = None,
    operator_name: str | None = None,
    output_dir: Path | None = None,
) -> None:
    """直接从 AgentResult 更新轨迹池，绕过文件 I/O。"""
    try:
        traj_pool_manager.prompt_config = se_cfg.prompt_config.to_dict()

        label = str(label_prefix) if label_prefix else f"iter{iteration_index}"
        problem_text = problem_description or ""
        traj_data = build_trajectory_from_result(
            result=result,
            instance_name=instance_name,
            problem_description=problem_text,
            iteration_index=iteration_index,
            label=label,
            source_labels=source_labels or [],
            operator_name=operator_name,
            output_dir=output_dir or Path("."),
        )

        traj_pool_manager.summarize_and_add_trajectory(traj_data.to_dict())

        pool_stats = traj_pool_manager.get_pool_stats()
        run_logger.info(f"轨迹池更新完毕（直接模式）: 当前共 {pool_stats.get('total_trajectories', 'unknown')} 条轨迹")
    except Exception as e:
        run_logger.error(f"直接模式轨迹池更新失败: {e}")


# ---------------------------------------------------------------------------
# 后处理入口
# ---------------------------------------------------------------------------


def process_and_summarize(
    iter_dir: Path,
    iter_idx: int,
    step: StepConfig,
    se_cfg: SEPerfRunSEConfig,
    pool_manager: TrajPoolManager,
    run_logger,
    label_prefix: str | None = None,
    source_labels: list[str] | None = None,
    operator_name: str | None = None,
    result: AgentResult | None = None,
    instance_name: str = "",
    problem_description: str | None = None,
):
    """后处理：生成 .tra 文件并从 AgentResult 更新轨迹池。"""
    try:
        # 始终生成 .tra 文件（用于持久化和调试）
        processor = TrajectoryProcessor()
        tra_stats = processor.process_iteration_directory(iter_dir)

        if not tra_stats or tra_stats.get("total_tra_files", 0) <= 0:
            run_logger.warning(f"迭代 {iter_idx} 未生成 .tra 文件")

        # 直接从 AgentResult 更新轨迹池
        if result is not None:
            update_pool_from_result(
                result=result,
                instance_name=instance_name,
                problem_description=problem_description or "",
                iteration_index=iter_idx,
                traj_pool_manager=pool_manager,
                se_cfg=se_cfg,
                run_logger=run_logger,
                label_prefix=label_prefix,
                source_labels=source_labels,
                operator_name=operator_name,
                # 单实例模式：文件直接在 iter_dir 下，不再嵌套 instance_name 子目录
                output_dir=iter_dir,
            )

        # 保存记忆快照
        try:
            mm = getattr(pool_manager, "memory_manager", None)
            if mm is not None:
                mem = mm.load()
                ckpt_path = Path(iter_dir) / f"memory_iter_{iter_idx}.json"
                with open(ckpt_path, "w", encoding="utf-8") as f:
                    json.dump(mem, f, ensure_ascii=False, indent=2)
                run_logger.info(f"已保存迭代 {iter_idx} 的记忆快照: {ckpt_path}")
        except Exception as e:
            run_logger.warning(f"保存迭代 {iter_idx} 记忆快照失败: {e}")
    except Exception as e:
        run_logger.error(f"迭代 {iter_idx} 后处理失败: {e}")
