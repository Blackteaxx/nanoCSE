"""
MATH TaskRunner 实现

支持 Open-AgentRL 中 MATH 类任务（math_dapo、AIME、Skywork train-math-* 等）
的进化搜索。评估方式：从 LLM 响应中提取 \\boxed{} 答案，与 ground_truth 比较。

metric：1.0（正确）或 0.0（错误），metric_higher_is_better=True。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from perfagent.protocols import TaskMetadata
from perfagent.task_runner import BaseTaskRunner


@dataclass
class MathTaskConfig:
    """MATH 任务特定配置"""

    use_math_verify: bool = True
    strict_box_verify: bool = True
    timeout: int = 10

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any] | None) -> "MathTaskConfig":
        if config_dict is None:
            return cls()
        return cls(
            use_math_verify=config_dict.get("use_math_verify", True),
            strict_box_verify=config_dict.get("strict_box_verify", True),
            timeout=config_dict.get("timeout", 10),
        )


@dataclass
class MathInstance:
    """MATH 实例数据"""

    id: str
    problem: str
    ground_truth: str
    data_source: str = ""
    ability: str = "MATH"
    extra_info: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict, file_path: Path | None = None) -> "MathInstance":
        instance_id = data.get("instance_id", "")
        if not instance_id and file_path:
            instance_id = file_path.stem

        problem = data.get("problem", "")
        if not problem:
            prompt = data.get("prompt")
            if isinstance(prompt, list) and prompt:
                problem = prompt[0].get("content", "")
            elif isinstance(prompt, str):
                problem = prompt

        ground_truth = data.get("ground_truth", "")
        if not ground_truth:
            rm = data.get("reward_model")
            if isinstance(rm, dict):
                ground_truth = str(rm.get("ground_truth", ""))
            elif isinstance(rm, str):
                ground_truth = rm

        return cls(
            id=instance_id,
            problem=problem,
            ground_truth=ground_truth,
            data_source=data.get("data_source", ""),
            ability=data.get("ability", "MATH"),
            extra_info=data.get("extra_info", {}),
        )


def _last_boxed_only_string(string: str) -> str | None:
    """Extract the last \\boxed{} expression from a string."""
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    num_left_braces_open = 0
    right_brace_idx = None
    for i in range(idx + 4, len(string)):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def _remove_boxed(s: str) -> str:
    """Remove the \\boxed{} wrapper."""
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left) : -1]
    return s


def _normalize_answer(answer: str) -> str:
    """Basic normalization for answer comparison."""
    answer = answer.strip()
    answer = re.sub(r"\s+", " ", answer)
    answer = re.sub(r"\\text\{(.*?)\}", r"\1", answer)
    answer = re.sub(r"\\textbf\{(.*?)\}", r"\1", answer)
    answer = re.sub(r"\\mathrm\{(.*?)\}", r"\1", answer)
    answer = answer.replace("\\$", "").replace("$", "")
    answer = answer.replace("\\%", "").replace("%", "")
    answer = answer.replace(",", "")
    answer = answer.strip()
    return answer


def _evaluate_math_answer(
    solution_text: str,
    ground_truth: str,
    *,
    use_math_verify: bool = True,
    strict_box_verify: bool = True,
) -> tuple[bool, str | None]:
    """Evaluate a math solution by comparing its \\boxed{} answer with ground truth.

    Returns (correct: bool, extracted_answer: str | None).
    """
    boxed = _last_boxed_only_string(solution_text)
    if boxed is None:
        return False, None

    extracted = _remove_boxed(boxed)
    norm_extracted = _normalize_answer(extracted)
    norm_gt = _normalize_answer(ground_truth)

    if norm_extracted == norm_gt:
        return True, extracted

    if use_math_verify:
        try:
            from math_verify import parse, verify

            parsed_solution = parse(solution_text, parsing_timeout=5)
            if len(parsed_solution) >= 2:
                if parsed_solution[1] == ground_truth:
                    return True, str(parsed_solution[1])

                parsed_gt = parse(f"\\boxed{{{ground_truth}}}", parsing_timeout=5)
                if verify(parsed_gt, parsed_solution, timeout_seconds=5):
                    return True, str(parsed_solution[1])
        except Exception:
            pass

    return False, extracted


class MathRunner(BaseTaskRunner):
    """MATH 任务的 TaskRunner 实现

    支持 Open-AgentRL 中的数学推理任务。通过 \\boxed{} 答案匹配评估正确性。
    """

    def __init__(
        self,
        *,
        task_config: dict[str, Any] | None = None,
        _logger: logging.Logger | None = None,
    ):
        self._logger = _logger or logging.getLogger(__name__)
        self._task_config = MathTaskConfig.from_dict(task_config)

    @classmethod
    def load_metadata(cls, path: Path) -> TaskMetadata:
        data = json.loads(path.read_text(encoding="utf-8"))
        instance = MathInstance.from_dict(data, path)
        return TaskMetadata(
            instance_id=instance.id,
            problem_description=instance.problem,
        )

    def load_instance(self, path: Path) -> MathInstance:
        data = json.loads(path.read_text(encoding="utf-8"))
        return MathInstance.from_dict(data, path)

    def get_initial_solution(self, instance_data: Any, config: Any) -> str:
        return ""

    def evaluate(
        self,
        solution: str,
        instance_data: Any,
        config: Any,
    ) -> tuple[float, dict[str, Any]]:
        instance: MathInstance = instance_data

        correct, extracted = _evaluate_math_answer(
            solution,
            instance.ground_truth,
            use_math_verify=self._task_config.use_math_verify,
            strict_box_verify=self._task_config.strict_box_verify,
        )

        metric = 1.0 if correct else 0.0

        artifacts: dict[str, Any] = {
            "correct": correct,
            "extracted_answer": extracted,
            "ground_truth": instance.ground_truth,
        }

        self._logger.info(
            f"实例 {instance.id}: {'正确' if correct else '错误'} "
            f"(extracted={extracted}, gt={instance.ground_truth})"
        )

        return metric, artifacts

    def build_system_prompt(self, instance_data: Any, **context: Any) -> str:
        instance: MathInstance = instance_data
        config = context.get("config")

        if config and hasattr(config, "prompts"):
            tmpl = getattr(config.prompts, "system_template", "")
            if tmpl:
                additional_requirements = (
                    context.get("additional_requirements")
                    or getattr(
                        getattr(config, "prompts", None),
                        "additional_requirements",
                        None,
                    )
                    or ""
                )
                local_memory = (
                    context.get("local_memory")
                    or getattr(
                        getattr(config, "prompts", None), "local_memory", None
                    )
                    or ""
                )
                global_memory = (
                    context.get("global_memory")
                    or getattr(
                        getattr(config, "prompts", None), "global_memory", None
                    )
                    or ""
                )
                try:
                    return tmpl.format(
                        problem=instance.problem,
                        ground_truth_hint="",
                        additional_requirements=additional_requirements,
                        local_memory=local_memory,
                        global_memory=global_memory,
                    )
                except KeyError:
                    pass

        return (
            "You are an expert mathematician. Solve the following math problem step by step.\n"
            "Show your complete reasoning process and put your final answer in \\boxed{}.\n\n"
            f"## Problem\n{instance.problem}\n\n"
            "## Instructions\n"
            "1. Analyze the problem carefully\n"
            "2. Show your step-by-step reasoning\n"
            "3. Put your final numerical answer in \\boxed{}\n"
        )

    def build_optimization_prompt(
        self,
        solution: str,
        metric: float,
        artifacts: dict[str, Any],
        **context: Any,
    ) -> str:
        config = context.get("config")

        if config and hasattr(config, "prompts"):
            tmpl = getattr(config.prompts, "optimization_template", "")
            if tmpl:
                correct = artifacts.get("correct", False)
                extracted = artifacts.get("extracted_answer", "N/A")
                ground_truth = artifacts.get("ground_truth", "")
                try:
                    return tmpl.format(
                        current_solution=solution,
                        correct="correct" if correct else "incorrect",
                        extracted_answer=extracted,
                        metric=metric,
                        ground_truth=ground_truth,
                    )
                except KeyError:
                    pass

        correct = artifacts.get("correct", False)
        extracted = artifacts.get("extracted_answer", "N/A")

        if correct:
            return (
                "Your previous solution was CORRECT. "
                "Try to find an alternative approach or verify your reasoning.\n\n"
                f"Previous solution:\n{solution}\n"
            )

        return (
            "Your previous solution was INCORRECT.\n"
            f"Your extracted answer: {extracted}\n\n"
            f"Previous solution:\n{solution}\n\n"
            "Please carefully re-analyze the problem and provide a corrected solution. "
            "Make sure to put your final answer in \\boxed{}."
        )

    def extract_solution(self, llm_response: str, current_solution: str) -> str:
        """Extract math solution from LLM response.

        For MATH tasks the entire response *is* the solution (including
        reasoning chain and \\boxed{} answer). We only fall back to
        current_solution when the response is empty.
        """
        response = llm_response.strip()
        if not response:
            self._logger.warning("LLM 响应为空，返回当前解")
            return current_solution
        return response
