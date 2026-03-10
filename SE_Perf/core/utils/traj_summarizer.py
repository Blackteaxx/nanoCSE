#!/usr/bin/env python3
"""
轨迹总结器
为trajectory pool生成轨迹总结的专用prompt系统
"""

from typing import Any  # noqa: UP035

from core.utils.json_utils import robust_json_loads
from core.utils.se_logger import get_se_logger


class TrajSummarizer:
    """轨迹总结器，生成轨迹分析prompt并解析响应"""

    def __init__(self, config: dict[str, Any] | None = None):
        self.logger = get_se_logger("traj_summarizer", emoji="📊")
        self.config = config or {}

    def get_system_prompt(self) -> str:
        """
        获取 PerfAgent 轨迹总结的系统提示词

        Returns:
            系统提示词字符串
        """
        cfg = {}
        try:
            cfg = self.config.get("summarizer", {}) if isinstance(self.config, dict) else {}
        except Exception:
            cfg = {}
        override = cfg.get("system_prompt")

        if isinstance(override, str) and override.strip():
            base = override
        else:
            base = """You are an AI assistant specialized in analyzing iterative optimization trajectories.

Your task is to analyze the provided agent execution data and provide a structured summary of the agent's problem-solving journey.

The agent's guiding principle is "CORRECTNESS FIRST, THEN PERFORMANCE". Your goal is to capture this iterative process, including its successes, failures, and analytical insights.

You will be provided with:
1. A problem description (optional)
2. A trajectory file (.tra) in JSON format containing the agent's step-by-step execution and chat history.
3. A prediction/solution file containing the final solution (this file might be redundant if the trajectory already contains the final solution, but should be used as the definitive "final_solution" if present).

Return your analysis in JSON format with the following fields:

- "solution_name": The nick name of the final solution (e.g., "monotonic_stack", "greedy_v2").

- "approach_summary": A concise high-level narrative describing the agent's complete journey and final approach (replaces 'overall_summary').

- "evolution_steps": A list of objects, one for each iteration (i.e., each `assistant` turn) found in the trajectory file. This chronologically tracks the agent's journey.
    - "iteration": The iteration number (e.g., 1, 2, 3...).
    - "thinking_summary": A summary of the agent's reasoning for this step (from its "Thinking" section).
    - "change_type": The *type* of change implemented. (e.g., "initial_implementation", "bugfix", "algorithm", "data-structure", "I/O_optimization", "micro-optimization", "reasoning_refinement").
    - "change_description": The specific technique or change implemented.
    - "metrics": The resulting metrics from the *next* 'user' feedback message (i.e., the feedback *after* this change was applied).
    - "status": A concise summary of this iteration's outcome (e.g., "Success: metric improved", "Failed: Correctness regression", "Failed: Error", "Failed: Metric regression").

- "analysis": An object containing high-level insights derived from the entire trajectory.
    - "best_strategy": An object describing the *best correct solution* achieved during the trajectory (if any). If no solution was ever correct, this can be null.
        - "high_level": "Abstract plan (algorithmic viewpoint)."
        - "algorithmic_choices": ["e.g., monotonic stack", "two-pointers"]
        - "data_structures": ["e.g., stack", "heap"]
    - "root_causes_of_failures": A list of objects detailing *why* iterations failed.
        - "iteration": The iteration number that failed.
        - "cause": "The root cause of the failure."
    - "key_learnings": A list of generalizable insights or patterns observed.
"""
        return base

    def get_user_prompt_template(self) -> str:
        """
        获取用户提示词模板

        Returns:
            用户提示词模板字符串
        """
        cfg = {}
        try:
            cfg = self.config.get("summarizer", {}) if isinstance(self.config, dict) else {}
        except Exception:
            cfg = {}
        override = cfg.get("user_prompt_template")
        if isinstance(override, str) and override.strip():
            return override
        return """Please analyze the following optimization trajectory and provide insights about the solution approach.

The trajectory tried to iteratively improve a solution for the problem described below.

Problem Description:
{problem_description}

Trajectory Data (.tra file):
{trajectory_content}

Solution:
{solution_content}

Please provide your analysis in the JSON format specified in the system prompt."""

    def format_user_prompt(
        self,
        trajectory_content: str,
        solution_content: str,
        problem_description: str | None = None,
        best_solution: str | None = None,
        target_solution: str | None = None,
    ) -> str:
        """
        格式化用户提示词

        Args:
            trajectory_content: 轨迹文件内容
            solution_content: 解/代码文本
            problem_description: 问题描述（可选）

        Returns:
            格式化后的用户提示词
        """
        template = self.get_user_prompt_template()
        mapping = {
            "problem_description": problem_description or "N/A",
            "trajectory_content": trajectory_content,
            "solution_content": solution_content or "",
            "best_solution": best_solution or "",
            "target_solution": target_solution or (solution_content or ""),
        }
        return template.format(**mapping)

    def parse_response(self, response_content: str) -> dict[str, Any]:
        """
        将LLM响应字符串严格转换为JSON对象。

        仅执行字符串到JSON的解析；若格式不正确或无法解析，抛出异常。

        Args:
            response_content: LLM响应的原始内容

        Returns:
            解析后的JSON数据

        Raises:
            ValueError: 当响应为空或未找到可解析的JSON片段时
            json.JSONDecodeError: 当JSON解析失败时
        """
        content = (response_content or "").strip()
        if not content:
            raise ValueError("空响应内容，无法解析为JSON")

        # 尝试直接解析完整JSON
        if content.startswith("{"):
            return robust_json_loads(content)

        # 尝试提取JSON片段进行解析
        start_idx = content.find("{")
        end_idx = content.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            json_content = content[start_idx:end_idx]
            return robust_json_loads(json_content)

        # 未找到可解析的JSON片段
        raise ValueError("响应中未找到可解析的JSON内容")

    def validate_response_format(self, response_data: dict[str, Any]) -> bool:
        if not isinstance(response_data, dict):
            raise ValueError("响应数据必须为字典")
        required_keys = ["solution_name", "approach_summary"]
        missing = [k for k in required_keys if k not in response_data]
        if missing:
            raise ValueError(f"响应格式缺少键: {', '.join(missing)}")
        return True

    def create_fallback_summary(self, trajectory_content: str, solution_content: str, iteration: int) -> dict[str, Any]:
        """
        创建备用总结（当LLM调用失败时使用）

        Args:
            trajectory_content: 轨迹内容
            solution_content: 解/代码文本
            iteration: 迭代次数

        Returns:
            备用总结数据
        """
        # 简单的备用分析
        trajectory_length = len(trajectory_content.split("\n")) if trajectory_content else 0
        patch_length = len(solution_content) if solution_content else 0

        return {
            "approach_summary": f"Iteration {iteration} execution with {trajectory_length} trajectory steps",
            "modified_files": ["unknown"],
            "key_changes": "Unable to analyze - LLM summarization failed",
            "strategy": f"iteration_{iteration}_strategy",
            "specific_techniques": ["automated_execution"],
            "tools_used": ["swe_agent"],
            "reasoning_pattern": "step_by_step_execution",
            "assumptions_made": ["standard_swe_agent_assumptions"],
            "components_touched": ["unknown_components"],
            "meta": {
                "is_fallback": True,
                "trajectory_length": trajectory_length,
                "patch_length": patch_length,
                "iteration": iteration,
            },
        }
