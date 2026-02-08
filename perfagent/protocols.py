"""
PerfAgent 协议定义

定义 SE_Perf 与 PerfAgent 之间的数据传输接口，
使两个模块之间的通信标准化、清晰化。

使用 AgentRequest / AgentResult 进行通信。
旧协议（PerfAgentRequest / PerfAgentResult）已移除。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _safe_float(value: Any) -> float:
    """安全转换为浮点数"""
    if value is None:
        return float("inf")
    if isinstance(value, str):
        if value.lower() in ("inf", "infinity"):
            return float("inf")
        if value.lower() in ("-inf", "-infinity"):
            return float("-inf")
        if value.lower() == "nan":
            return float("nan")
        try:
            return float(value)
        except ValueError:
            return float("inf")
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("inf")


# ===========================================================================
# 任务元数据
# ===========================================================================


@dataclass
class TaskMetadata:
    """任务的最小元数据

    供 SE_Perf 在首次调用 Agent 之前使用（例如用于全局记忆检索、
    轨迹池匹配等），无需加载完整的任务数据。

    Attributes:
        instance_id: 任务实例唯一标识
        problem_description: 任务/问题的自然语言描述
    """

    instance_id: str
    problem_description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "problem_description": self.problem_description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskMetadata":
        return cls(
            instance_id=data.get("instance_id", "unknown"),
            problem_description=data.get("problem_description", ""),
        )


# ===========================================================================
# AgentRequest / AgentResult —— SE_Perf <-> Agent 通信接口
# ===========================================================================


@dataclass
class AgentRequest:
    """通用 Agent 请求（SE_Perf -> Agent）

    任务无关的标准化请求对象。SE_Perf 通过此对象向 Agent 传递
    任务数据路径、配置以及可选的覆盖参数，而不包含任何任务特定的数据结构。

    Attributes:
        task_data_path: 任务数据文件路径（由 TaskRunner 负责加载和解析）
        config: Agent 配置（任意配置对象，当前为 PerfAgentConfig）
        additional_requirements: 算子策略文本（可选）
        local_memory: 本地记忆内容（可选）
        global_memory: 全局记忆内容（可选）
        output_dir: 输出目录（用于保存轨迹等文件）
    """

    task_data_path: Path
    config: Any  # 通常为 PerfAgentConfig，保持通用以解耦
    # 可选覆盖 / 上下文参数
    additional_requirements: str | None = None
    local_memory: str | None = None
    global_memory: str | None = None
    output_dir: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典（config 需自行提供 to_dict）"""
        config_dict: Any
        if hasattr(self.config, "to_dict"):
            config_dict = self.config.to_dict()
        else:
            config_dict = self.config

        return {
            "task_data_path": str(self.task_data_path),
            "config": config_dict,
            "additional_requirements": self.additional_requirements,
            "local_memory": self.local_memory,
            "global_memory": self.global_memory,
            "output_dir": str(self.output_dir) if self.output_dir else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], config_cls: type | None = None) -> "AgentRequest":
        """从字典创建请求对象

        Args:
            data: 字典数据
            config_cls: 配置类（需实现 from_dict 类方法）；
                        为 None 时原样保留 config 字典
        """
        config_raw = data.get("config", {})
        if config_cls is not None and hasattr(config_cls, "from_dict"):
            config = config_cls.from_dict(config_raw)
        else:
            config = config_raw

        output_dir = Path(data["output_dir"]) if data.get("output_dir") else None

        return cls(
            task_data_path=Path(data["task_data_path"]),
            config=config,
            additional_requirements=data.get("additional_requirements"),
            local_memory=data.get("local_memory"),
            global_memory=data.get("global_memory"),
            output_dir=output_dir,
        )


@dataclass
class AgentResult:
    """通用 Agent 结果（Agent -> SE_Perf）

    任务无关的标准化结果对象。Agent 通过此对象向 SE_Perf 返回
    解、标量指标以及任务特定的上下文信息（artifacts）。

    设计约定:
    - ``metric`` 语义统一为 **越低越好**（lower is better）。
      若某任务的原始指标是越高越好（如 pass rate），应在 TaskRunner
      中取反后再写入此字段。
    - ``problem_description`` 由 ``TaskMetadata`` 提供，不存储在此对象中。

    Attributes:
        instance_id: 任务实例唯一标识
        success: 是否成功（由 TaskRunner 定义成功标准）
        solution: 解（代码文本、数学答案等）
        metric: 标量性能指标（越低越好）
        artifacts: 任务特定的上下文字典（评估元信息等）
        total_iterations: 总迭代次数
        trajectory_file: 轨迹文件路径（可选）
        error: 错误信息（如果执行失败）
    """

    instance_id: str
    success: bool
    solution: str
    metric: float
    artifacts: dict[str, Any] = field(default_factory=dict)
    total_iterations: int = 0
    trajectory_file: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "success": self.success,
            "solution": self.solution,
            "metric": self.metric,
            "artifacts": self.artifacts,
            "total_iterations": self.total_iterations,
            "trajectory_file": self.trajectory_file,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentResult":
        artifacts = dict(data.get("artifacts", {}))
        artifacts.pop("problem_description", None)  # 清理残留
        return cls(
            instance_id=data.get("instance_id", "unknown"),
            success=data.get("success", False),
            solution=data.get("solution", ""),
            metric=_safe_float(data.get("metric", float("inf"))),
            artifacts=artifacts,
            total_iterations=data.get("total_iterations", 0),
            trajectory_file=data.get("trajectory_file"),
            error=data.get("error"),
        )

    @classmethod
    def from_error(
        cls,
        instance_id: str,
        error: str,
    ) -> "AgentResult":
        """从错误创建失败结果"""
        return cls(
            instance_id=instance_id,
            success=False,
            solution="",
            metric=float("inf"),
            artifacts={},
            error=error,
        )

    def to_json(self, path: Path | str) -> None:
        """保存为 JSON 文件"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, path: Path | str) -> "AgentResult":
        """从 JSON 文件加载"""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
