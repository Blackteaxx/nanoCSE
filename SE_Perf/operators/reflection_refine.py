#!/usr/bin/env python3
"""
Reflection and Refine Operator

根据给定的源轨迹（source trajectory）进行反思与改进，生成更优的实现策略要求，
用于在下一次 PerfAgent 迭代中指导代码优化。
"""

import textwrap
from typing import Any

from perf_config import StepConfig

from operators.base import BaseOperator, OperatorResult


class ReflectionRefineOperator(BaseOperator):
    """
    反思与改进算子：
    输入：step_config.inputs 中给定的单个源轨迹标签，如 {"label": "sol1"}
    输出：带有反思与具体改进指令的 additional_requirements 文本。
    """

    def get_name(self) -> str:
        return "reflection_refine"

    def run_for_instance(
        self,
        step_config: StepConfig,
        instance_name: str,
        instance_entry: dict[str, Any],
        *,
        problem_description: str = "",
    ) -> OperatorResult:
        """处理单个实例的反思与改进。"""
        if not isinstance(instance_entry, dict):
            return OperatorResult()

        src_summary = None
        used_labels: list[str] = []

        # 若未提供输入标签，进行线性加权采样选择源轨迹
        chosen = self._select_source_labels(instance_entry, step_config, required_n=1)
        if chosen:
            sub = instance_entry.get(chosen[0])
            if isinstance(sub, dict):
                src_summary = self._format_entry({str(chosen[0]): sub})
                used_labels = [str(chosen[0])]
        else:
            keys = self._weighted_select_labels(instance_entry, k=1)
            if keys:
                sub = instance_entry.get(keys[0])
                if isinstance(sub, dict):
                    src_summary = self._format_entry({str(keys[0]): sub})
                    used_labels = [str(keys[0])]

        # 最后回退：使用最新条目摘要
        if not src_summary:
            src_summary = self._format_entry(instance_entry)

        if not problem_description or not src_summary:
            return OperatorResult(source_labels=used_labels)

        content = self._build_additional_requirements(src_summary)
        if not content:
            return OperatorResult(source_labels=used_labels)

        return OperatorResult(
            additional_requirements=content,
            source_labels=used_labels,
        )

    def _build_additional_requirements(self, source_summary: str) -> str:
        """
        构造带有反思与改进要求的 additional_requirements 文本。
        """
        src = textwrap.indent((source_summary or "").strip(), "  ")
        pcfg = self.context.prompt_config or {}
        opcfg = pcfg.get("reflection_refine", {}) if isinstance(pcfg, dict) else {}
        header = (
            opcfg.get("header")
            or pcfg.get("reflection_header")
            or "### STRATEGY MODE: REFLECTION AND REFINE STRATEGY\nYou must explicitly reflect on the previous trajectory and implement concrete improvements."
        )
        guidelines = (
            opcfg.get("guidelines")
            or pcfg.get("reflection_guidelines")
            or (
                """
### REFINEMENT GUIDELINES
1. **Diagnose**: Identify the main shortcomings (correctness risks, bottlenecks, redundant work, I/O overhead).
2. **Fixes**: Propose targeted code-level changes (algorithmic upgrade, data structure replacement, caching/precomputation, I/O batching).
3. **Maintain Correctness**: Prioritize correctness; add guards/tests if necessary before optimizing runtime.
4. **Performance Goal**: Aim for measurable runtime improvement. Prefer asymptotic gains over micro-optimizations.
            """
            )
        )
        parts = []
        if isinstance(header, str) and header.strip():
            parts.append(header.strip())
        parts.append("\n### SOURCE TRAJECTORY SUMMARY\n" + src)
        if isinstance(guidelines, str) and guidelines.strip():
            parts.append("\n" + guidelines.strip())
        return "\n".join(parts)


# 注册算子
from .registry import register_operator

register_operator("reflection_refine", ReflectionRefineOperator)
