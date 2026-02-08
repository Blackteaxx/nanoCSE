"""
迭代执行模块

负责 PerfAgent 的单次运行、算子执行以及迭代步骤的统一调度。
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

from core.utils.global_memory_manager import GlobalMemoryManager
from core.utils.local_memory_manager import LocalMemoryManager
from core.utils.traj_pool_manager import TrajPoolManager
from operators import create_operator
from operators.base import InstanceTrajectories, OperatorContext, OperatorResult
from perf_config import SEPerfRunSEConfig, StepConfig
from results_io import write_iteration_preds_from_result, write_result_json
from run_helpers import build_operator_context, build_perf_agent_config, retrieve_global_memory
from trajectory_handler import process_and_summarize

from perfagent.agent import PerfAgent
from perfagent.config import PerfAgentConfig
from perfagent.protocols import AgentRequest, AgentResult
from perfagent.task_runner import BaseTaskRunner


# ---------------------------------------------------------------------------
# PerfAgent 单次运行
# ---------------------------------------------------------------------------


def run_single_perfagent(
    task_data_path: Path,
    instance_id: str,
    perf_config: PerfAgentConfig,
    operator_result: OperatorResult,
    local_memory_text: str | None,
    global_memory_text: str | None,
    output_dir: Path,
    logger,
    problem_description: str | None = None,
    task_runner: BaseTaskRunner | None = None,
) -> AgentResult:
    """构建 AgentRequest 并直接调用 PerfAgent.run_with_request()。

    若调用方已创建 TaskRunner（如 perf_run 按 task_type 创建），应传入
    task_runner，以便 PerfAgent 使用正确的任务类型（effibench / livecodebench / aime 等）。
    未传入时 PerfAgent 内部 _ensure_task_runner 会创建 EffiBenchRunner（向后兼容）。
    """

    # 构建配置副本
    config = copy.deepcopy(perf_config)
    config.logging.trajectory_dir = output_dir
    config.logging.log_dir = output_dir

    request = AgentRequest(
        task_data_path=task_data_path,
        config=config,
        additional_requirements=operator_result.additional_requirements,
        local_memory=local_memory_text,
        global_memory=global_memory_text,
        output_dir=output_dir,
    )

    logger.info(
        f"构建 AgentRequest: instance={instance_id}, "
        f"has_additional_requirements={bool(operator_result.additional_requirements)}, "
        f"has_initial_solution={bool(operator_result.initial_solution)}, "
        f"has_local_memory={bool(local_memory_text)}, "
        f"has_global_memory={bool(global_memory_text)}"
    )

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        agent = PerfAgent(config, task_runner=task_runner)
        result = agent.run_with_request(request)
        logger.info(
            f"PerfAgent 执行完成: instance={instance_id}, "
            f"success={result.success}, "
            f"metric={result.metric}"
        )
        return result
    except Exception as e:
        logger.error(f"PerfAgent 执行异常: {e}", exc_info=True)
        return AgentResult.from_error(
            instance_id=instance_id,
            error=str(e),
        )


# ---------------------------------------------------------------------------
# 算子执行
# ---------------------------------------------------------------------------


def run_operator(
    step: StepConfig,
    instance_name: str,
    instance_entry: InstanceTrajectories,
    op_context: OperatorContext,
    traj_pool_manager: TrajPoolManager,
    logger,
    problem_description: str = "",
) -> list[OperatorResult]:
    """执行算子并返回 OperatorResult 列表。

    - Plan 算子返回多个 OperatorResult。
    - Filter 算子直接修改轨迹池后返回空列表（不触发 PerfAgent）。
    - 普通算子返回单元素列表。
    - 无算子时返回默认空 OperatorResult 的单元素列表。
    """
    if not step.operator:
        return [OperatorResult()]

    logger.info(f"执行算子: {step.operator}")
    op_instance = create_operator(step.operator, op_context)
    if not op_instance:
        logger.error(f"无法创建算子实例: {step.operator}")
        return [OperatorResult()]

    try:
        if step.is_filter:
            op_instance.run_for_instance(
                step, instance_name, instance_entry,
                problem_description=problem_description,
                traj_pool_manager=traj_pool_manager,
            )
            logger.info("Filter 算子执行完毕，跳过后续 PerfAgent 运行")
            return []

        result = op_instance.run_for_instance(
            step, instance_name, instance_entry,
            problem_description=problem_description,
        )
        
        # 记录算子返回结果的详细信息
        if isinstance(result, list):
            logger.info(f"算子 {step.operator} 返回了 {len(result)} 个结果")
            for i, r in enumerate(result):
                logger.info(f"  结果{i}: has_additional_requirements={bool(r.additional_requirements)}, source_labels={r.source_labels}")
            return result  # Plan 算子返回 list[OperatorResult]
        else:
            logger.info(f"算子 {step.operator} 返回了单个结果: has_additional_requirements={bool(result.additional_requirements)}, source_labels={result.source_labels}")
            return [result]
    except Exception as e:
        logger.error(f"算子 '{step.operator}' 执行失败: {e}", exc_info=True)
        return [OperatorResult()]


def resolve_labels(step: StepConfig, num_results: int) -> list[str]:
    """为算子结果列表生成对应的轨迹标签。"""
    if step.is_plan:
        labels = list(step.trajectory_labels or [])
        if not labels:
            labels = [f"sol{i}" for i in range(1, num_results + 1)]
        # 补齐不足的标签
        while len(labels) < num_results:
            labels.append(f"sol{len(labels) + 1}")
        return labels[:num_results]
    else:
        # 普通算子：使用 trajectory_label 或默认标签
        label = step.trajectory_label
        return [label] * num_results if label else [None] * num_results  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# 单次执行（PerfAgent + 后处理）
# ---------------------------------------------------------------------------


def execute_single_run(
    op_result: OperatorResult,
    label: str | None,
    task_data_path: Path,
    instance_name: str,
    problem_description: str,
    se_cfg: SEPerfRunSEConfig,
    step: StepConfig,
    traj_pool_manager: TrajPoolManager,
    local_memory_text: str | None,
    global_memory: GlobalMemoryManager | None,
    output_dir: str,
    iteration_idx: int,
    mode: str,
    logger,
    task_runner: BaseTaskRunner | None = None,
) -> None:
    """执行单次 PerfAgent 运行并后处理。"""
    iter_dir = Path(output_dir) / f"iteration_{iteration_idx}"
    # 单实例模式：文件直接放在 iteration 目录下，不再嵌套 instance_name 子目录
    instance_output_dir = iter_dir

    # 检索 Global Memory
    global_memory_text = retrieve_global_memory(
        global_memory=global_memory,
        instance_name=instance_name,
        problem_description=problem_description or "",
        additional_requirements=op_result.additional_requirements,
        local_memory_text=local_memory_text,
    )

    # 构建 PerfAgent 配置
    perf_config = build_perf_agent_config(
        base_config_path=step.perf_base_config or se_cfg.base_config,
        se_model_config=se_cfg.model.to_dict(),
        max_iterations=se_cfg.max_iterations,
        output_dir=instance_output_dir,
    )

    label_info = f" ({step.operator}: {label})" if step.operator and label else ""
    print(f"\n=== 迭代 {iteration_idx}{label_info} ===")
    os.environ["SE_ITERATION_INDEX"] = str(iteration_idx)

    if mode == "demo":
        logger.info("演示模式：跳过实际执行")
        return

    # 执行 PerfAgent
    result = run_single_perfagent(
        task_data_path=task_data_path,
        instance_id=instance_name,
        perf_config=perf_config,
        operator_result=op_result,
        local_memory_text=local_memory_text,
        global_memory_text=global_memory_text,
        output_dir=instance_output_dir,
        logger=logger,
        problem_description=problem_description,
        task_runner=task_runner,
    )

    # 写入文件（用于持久化和调试）
    write_result_json(result, instance_output_dir, logger)
    write_iteration_preds_from_result(result, iter_dir, logger)

    # 后处理：更新轨迹池
    if result.success or result.error is None:
        process_and_summarize(
            iter_dir,
            iteration_idx,
            step,
            se_cfg,
            traj_pool_manager,
            logger,
            label_prefix=label,
            source_labels=op_result.source_labels or None,
            operator_name=step.operator,
            result=result,
            instance_name=instance_name,
            problem_description=problem_description or "",
        )

    # 始终记录算子执行结果（无论 source_labels 是否为空）
    logger.info(
        f"算子 {step.operator} 执行完成: "
        f"has_additional_requirements={bool(op_result.additional_requirements)}, "
        f"has_initial_solution={bool(op_result.initial_solution)}, "
        f"source_labels={op_result.source_labels}"
    )


# ---------------------------------------------------------------------------
# 迭代步骤入口
# ---------------------------------------------------------------------------


def execute_iteration(
    step: StepConfig,
    task_data_path: Path,
    instance_id: str,
    problem_description: str,
    se_cfg: SEPerfRunSEConfig,
    traj_pool_manager: TrajPoolManager,
    local_memory: LocalMemoryManager | None,
    global_memory: GlobalMemoryManager | None,
    output_dir: str,
    iteration_idx: int,
    mode: str,
    logger,
    task_runner: BaseTaskRunner | None = None,
) -> int:
    """执行单个迭代步骤，返回下一个迭代索引。

    统一了 Plan / Filter / Regular 三个分支的执行逻辑。
    """
    # 刷新 local_memory_text
    local_memory_text = None
    try:
        if local_memory is not None:
            _mem_latest = local_memory.load()
            local_memory_text = local_memory.render_as_markdown(_mem_latest)
    except Exception:
        pass

    # 获取当前轨迹池数据（单实例，扁平结构）
    raw_entry = traj_pool_manager.load_pool()
    instance_entry = InstanceTrajectories.from_dict(raw_entry)

    # 构建算子上下文
    op_context = build_operator_context(se_cfg, step)

    # 执行算子
    op_results = run_operator(
        step, instance_id, instance_entry, op_context, traj_pool_manager, logger,
        problem_description=problem_description,
    )

    # Filter 算子不触发 PerfAgent，直接返回
    if step.is_filter or not op_results:
        return iteration_idx

    # 为每个结果生成对应标签
    labels = resolve_labels(step, len(op_results))

    # 对每个 OperatorResult 执行 PerfAgent + 后处理
    for op_result, label in zip(op_results, labels):
        execute_single_run(
            op_result=op_result,
            label=label,
            task_data_path=task_data_path,
            instance_name=instance_id,
            problem_description=problem_description,
            se_cfg=se_cfg,
            step=step,
            traj_pool_manager=traj_pool_manager,
            local_memory_text=local_memory_text,
            global_memory=global_memory,
            output_dir=output_dir,
            iteration_idx=iteration_idx,
            mode=mode,
            logger=logger,
            task_runner=task_runner,
        )
        iteration_idx += 1

    return iteration_idx
