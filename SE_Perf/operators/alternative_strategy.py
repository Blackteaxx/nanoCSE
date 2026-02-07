#!/usr/bin/env python3
"""
Alternative Strategy Operator

基于指定的输入轨迹，生成一个全新的、策略上截然不同的解决方案。
此算子旨在跳出局部最优，从不同维度（例如，算法、数据结构、I/O模式）探索解空间。
"""

import textwrap
from typing import Any

from perf_config import StepConfig

from operators.base import BaseOperator, OperatorResult


class AlternativeStrategyOperator(BaseOperator):
    """
    替代策略算子：
    根据 step_config 中指定的单个输入轨迹（input），
    生成一个策略迥异的新轨迹（output）。
    """

    def get_name(self) -> str:
        return "alternative_strategy"

    def run_for_instance(
        self,
        step_config: StepConfig,
        instance_name: str,
        instance_entry: dict[str, Any],
        *,
        problem_description: str = "",
    ) -> OperatorResult:
        """处理单个实例的替代策略生成。"""
        if not isinstance(instance_entry, dict):
            return OperatorResult()

        previous_approach_summary = None
        used_labels: list[str] = []

        # 使用统一方法选择源标签（required_n=1）；若无则回退到全体加权采样
        chosen = self._select_source_labels(instance_entry, step_config, required_n=1)
        if chosen:
            sub = instance_entry.get(chosen[0])
            if isinstance(sub, dict):
                previous_approach_summary = self._format_entry({str(chosen[0]): sub})
                used_labels = [str(chosen[0])]
        else:
            src_keys = self._weighted_select_labels(instance_entry, k=1)
            if src_keys:
                sub = instance_entry.get(src_keys[0])
                if isinstance(sub, dict):
                    previous_approach_summary = self._format_entry({str(src_keys[0]): sub})
                    used_labels = [str(src_keys[0])]

        if not previous_approach_summary:
            previous_approach_summary = self._format_entry(instance_entry)

        if not problem_description or not previous_approach_summary:
            return OperatorResult(source_labels=used_labels)

        content = self._build_additional_requirements(previous_approach_summary)
        if not content:
            return OperatorResult(source_labels=used_labels)

        return OperatorResult(
            additional_requirements=content,
            source_labels=used_labels,
        )

    def _build_additional_requirements(self, previous_approach: str) -> str:
        prev = textwrap.indent(previous_approach.strip(), "  ")
        pcfg = self.context.prompt_config or {}
        opcfg = pcfg.get("alternative_strategy", {}) if isinstance(pcfg, dict) else {}
        header = (
            opcfg.get("header")
            or pcfg.get("alternative_header")
            or "### STRATEGY MODE: ALTERNATIVE SOLUTION STRATEGY\nYou are explicitly instructed to abandon the current optimization trajectory and implement a FUNDAMENTALLY DIFFERENT approach."
        )
        guidelines = (
            opcfg.get("guidelines")
            or pcfg.get("alternative_guidelines")
            or (
                """
### EXECUTION GUIDELINES
1. **Qualitative Shift**: You must NOT provide incremental refinements, micro-optimizations, or simple bugfixes to the code above.
2. **New Paradigm**: Switch the algorithmic paradigm or data structure entirely (e.g., if Greedy -> try DP; if List -> try Heap/Deque; if Iterative -> try Recursive).
3. **Shift Bottleneck Focus**: If the previous attempt focused heavily on Core Algorithmics, consider an I/O-centric technique (or vice versa).
4. **Target**: Aim for a better Big-O complexity (e.g., O(N) over O(N log N)) where feasible.
            """
            )
        )
        parts = []
        if isinstance(header, str) and header.strip():
            parts.append(header.strip())
        parts.append("\n### PREVIOUS APPROACH SUMMARY\n" + prev)
        if isinstance(guidelines, str) and guidelines.strip():
            parts.append("\n" + guidelines.strip())
        return "\n".join(parts)


# 注册算子
from .registry import register_operator

register_operator("alternative_strategy", AlternativeStrategyOperator)
