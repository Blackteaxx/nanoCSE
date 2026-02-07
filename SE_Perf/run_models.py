"""
perf_run 数据模型

定义 perf_run 流水线中各模块间传递的结构化数据类型，
替代原先通过裸 dict 传递的非类型安全方式。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrajectoryData:
    """轨迹池条目数据。

    用于 ``traj_pool_manager.summarize_and_add_trajectories()`` 的输入，
    替代原先手动构建的 dict。

    Attributes:
        label: 轨迹标签（如 "sol1", "iter3"）。
        instance_name: 实例名称。
        problem_description: 问题描述文本。
        trajectory_content: 优化轨迹内容（.tra 格式文本）。
        solution: 优化后的解（代码或答案）。
        metric: 标量指标（越低越好）。
        artifacts: 任务特定上下文（可选）。
        iteration: 迭代编号。
        source_dir: 源输出目录。
        source_entry_labels: 来源轨迹标签列表。
        operator_name: 产生此轨迹的算子名称。
    """

    label: str = ""
    instance_name: str = ""
    problem_description: str = ""
    trajectory_content: str = ""
    solution: str = ""
    metric: float | str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    iteration: int = 0
    source_dir: str = ""
    source_entry_labels: list[str] = field(default_factory=list)
    operator_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为 dict，供 traj_pool_manager 接口使用。"""
        return {
            "label": self.label,
            "instance_name": self.instance_name,
            "problem_description": self.problem_description,
            "trajectory_content": self.trajectory_content,
            "solution": self.solution,
            "metric": self.metric,
            "artifacts": dict(self.artifacts),
            "iteration": self.iteration,
            "source_dir": self.source_dir,
            "source_entry_labels": list(self.source_entry_labels),
            "operator_name": self.operator_name,
        }


@dataclass
class GlobalMemoryContext:
    """Global Memory 检索上下文。

    替代 ``_retrieve_global_memory()`` 中手动构建的 context dict。

    Attributes:
        language: 编程语言（如 "python3"）。
        optimization_target: 优化目标（如 "runtime"）。
        problem_description: 问题描述。
        additional_requirements: 额外要求文本。
        local_memory: 本地记忆内容。
    """

    language: str = "python3"
    optimization_target: str = "runtime"
    problem_description: str = ""
    additional_requirements: str = ""
    local_memory: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转换为 dict，兼容 GlobalMemoryManager 接口。"""
        return {
            "language": self.language,
            "optimization_target": self.optimization_target,
            "problem_description": self.problem_description,
            "additional_requirements": self.additional_requirements,
            "local_memory": self.local_memory,
        }


@dataclass
class PredictionEntry:
    """单实例预测结果条目。

    替代 ``write_iteration_preds_from_result()`` 中手动构建的预测 dict。

    Attributes:
        solution: 优化后的解（代码或答案）。
        metric: 标量指标（越低越好）。
        success: 是否成功（由 TaskRunner 定义）。
        artifacts: 任务特定上下文。
    """

    solution: str = ""
    metric: float | str | None = None
    success: bool = False
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为 dict，用于 JSON 序列化。"""
        return {
            "solution": self.solution,
            "metric": self.metric,
            "success": self.success,
            "artifacts": dict(self.artifacts),
        }
