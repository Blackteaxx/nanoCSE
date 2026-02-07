#!/usr/bin/env python3
"""
Crossover Operator

当轨迹池中有效条数大于等于2时，结合两条轨迹的特性生成新的策略。
当有效条数不足时，记录错误并跳过处理。
"""

import textwrap
from typing import Any

from perf_config import StepConfig

from operators.base import BaseOperator, OperatorResult


class CrossoverOperator(BaseOperator):
    """交叉算子：综合两条轨迹的优点，生成新的初始代码"""

    def get_name(self) -> str:
        return "crossover"

    def run_for_instance(
        self,
        step_config: StepConfig,
        instance_name: str,
        instance_entry: dict[str, Any],
        *,
        problem_description: str = "",
    ) -> OperatorResult:
        """处理单个实例的交叉操作。

        从 instance_entry 中选择两条轨迹，生成交叉策略的 additional_requirements。
        """
        if not isinstance(instance_entry, dict):
            return OperatorResult()

        # 自适应选择两个源轨迹（不重复）
        chosen = self._select_source_labels(instance_entry, step_config, required_n=2)
        pick1 = chosen[0] if len(chosen) >= 1 else None
        pick2 = chosen[1] if len(chosen) >= 2 else None

        if pick1 and pick2 and pick1 == pick2:
            # 保障不重复，若重复则尝试再选一个不同的
            extra = [l for l in self._weighted_select_labels(instance_entry, k=3) if l != pick1]
            if extra:
                pick2 = extra[0]

        ref1 = instance_entry.get(pick1) if pick1 else None
        ref2 = instance_entry.get(pick2) if pick2 else None
        used = [s for s in [pick1, pick2] if isinstance(s, str) and s]

        if not isinstance(ref1, dict) or not isinstance(ref2, dict):
            return OperatorResult(source_labels=used)

        summary1 = self._format_entry({str(pick1 or "iter1"): ref1})
        summary2 = self._format_entry({str(pick2 or "iter2"): ref2})

        if not problem_description or not summary1 or not summary2:
            return OperatorResult(source_labels=used)

        content = self._build_additional_requirements(summary1, summary2)
        if not content:
            return OperatorResult(source_labels=used)

        return OperatorResult(
            additional_requirements=content,
            source_labels=used,
        )

    def _build_additional_requirements(self, trajectory1: str, trajectory2: str) -> str:
        t1 = textwrap.indent(trajectory1.strip(), "  ")
        t2 = textwrap.indent(trajectory2.strip(), "  ")
        pcfg = self.context.prompt_config or {}
        opcfg = pcfg.get("crossover", {}) if isinstance(pcfg, dict) else {}
        header = (
            opcfg.get("header")
            or pcfg.get("crossover_header")
            or "### STRATEGY MODE: CROSSOVER STRATEGY\nYou are tasked with synthesizing a SUPERIOR hybrid solution by intelligently combining the best elements of two prior optimization trajectories described below."
        )
        guidelines = (
            opcfg.get("guidelines")
            or pcfg.get("crossover_guidelines")
            or (
                """
### SYNTHESIS GUIDELINES
1. **Complementary Combination**: Actively combine specific strengths.
- Example: If T1 has a better Core Algorithm but slow I/O, and T2 has fast I/O but a naive algorithm, implement T1's algorithm using T2's I/O technique.
- Example: If T1 used a correct Stack logic but slow List, and T2 used a fast Array but had logic bugs, implement T1's logic using T2's structure.
2. **Avoid Shared Weaknesses**: If both trajectories failed at a specific sub-task, you must introduce a novel fix for that specific part.
3. **Seamless Integration**: Do not just concatenate code. The resulting logic must be a single, cohesive implementation.
            """
            )
        )
        parts = []
        if isinstance(header, str) and header.strip():
            parts.append(header.strip())
        parts.append("\n### TRAJECTORY 1 SUMMARY\n" + t1)
        parts.append("\n### TRAJECTORY 2 SUMMARY\n" + t2)
        if isinstance(guidelines, str) and guidelines.strip():
            parts.append("\n" + guidelines.strip())
        return "\n".join(parts)


# 注册算子
from .registry import register_operator

register_operator("crossover", CrossoverOperator)
