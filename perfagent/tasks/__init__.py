"""
perfagent.tasks — 任务特定 TaskRunner 实现

每个子模块对应一种任务类型（如 effibench、livecodebench、aime），
实现 BaseTaskRunner 接口，使 Agent 核心循环保持任务无关。
"""

from .effibench import EffiBenchRunner, EffiBenchXInstance

__all__ = [
    "EffiBenchRunner",
    "EffiBenchXInstance",
]
