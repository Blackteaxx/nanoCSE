#!/usr/bin/env python3

"""
Trajectory Analyzer Operator

直接分析轨迹池中的条目，提取问题与轨迹快照，产出附加需求文本。
"""

import textwrap

from perf_config import StepConfig

from operators.base import InstanceTrajectories, OperatorResult, TemplateOperator


class TrajectoryAnalyzerOperator(TemplateOperator):
    def get_name(self) -> str:
        return "trajectory_analyzer"

    def get_strategy_prefix(self) -> str:
        pcfg = self.context.prompt_config or {}
        opcfg = pcfg.get("trajectory_analyzer", {}) if isinstance(pcfg, dict) else {}
        return str(opcfg.get("prefix") or pcfg.get("trajectory_analyzer_prefix") or "SOLUTION STRATEGY")

    def run_for_instance(
        self,
        step_config: StepConfig,
        instance_name: str,
        instance_entry: InstanceTrajectories,
        *,
        problem_description: str = "",
    ) -> OperatorResult:
        """处理单个实例的轨迹分析。"""
        if not instance_entry.trajectories:
            return OperatorResult()

        problem_statement = problem_description
        snapshot = self._format_entry(instance_entry)

        if not problem_statement or not snapshot:
            return OperatorResult()

        content = self._build_additional_requirements(str(problem_statement), snapshot)
        if not content:
            return OperatorResult()

        return OperatorResult(additional_requirements=content)

    def _build_additional_requirements(self, problem_statement: str, trajectory_snapshot: str) -> str:
        pcfg = self.context.prompt_config or {}
        opcfg = pcfg.get("trajectory_analyzer", {}) if isinstance(pcfg, dict) else {}
        header = str(opcfg.get("header") or pcfg.get("trajectory_analyzer_header") or "SOLUTION STRATEGY")
        prob = textwrap.indent(str(problem_statement).strip(), "  ")
        snap = textwrap.indent(str(trajectory_snapshot).strip(), "  ")
        body = (
            opcfg.get("guidance")
            or pcfg.get("trajectory_analyzer_guidance")
            or (
                "Guidance:\n"
                "1. Begin from an alternative entry point: runtime tracing, I/O profiling, or component isolation.\n"
                "2. Use a non-linear reasoning sequence with hypothesis→micro-test loops.\n"
                "3. Integrate unconventional techniques: targeted benchmarks, memory profiling, fuzzing.\n"
                "4. Prioritize overlooked aspects: performance metrics, boundary conditions, integration constraints.\n"
                "5. Keep changes minimal and testable; modify one module at a time with assertions.\n"
                "6. Explicitly validate assumptions to avoid repeating prior patterns.\n"
            )
        )
        return f"{header}\n\nPROBLEM:\n{prob}\n\nTRAJECTORY SNAPSHOT:\n{snap}\n\n{body}".strip()


from .registry import register_operator

register_operator("trajectory_analyzer", TrajectoryAnalyzerOperator)
