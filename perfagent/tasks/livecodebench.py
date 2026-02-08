"""
LiveCodeBench TaskRunner 实现

支持 LiveCodeBench 格式的代码生成任务，使用 LiveCodeBench 的评测系统进行评估。
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from perfagent.task_runner import BaseTaskRunner
from perfagent.protocols import TaskMetadata

from perfagent.lcb_eval.testing_util import run_test


@dataclass
class LCBTaskConfig:
    """LiveCodeBench 任务特定配置"""
    timeout: int = 6
    use_only_public_tests: bool = False  # 仅使用公开测试用例（用于快速测试）
    enable_detailed_feedback: bool = True  # 在 artifacts 中包含详细的失败信息
    
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any] | None) -> "LCBTaskConfig":
        if config_dict is None:
            return cls()
        return cls(
            timeout=config_dict.get("timeout", 6),
            use_only_public_tests=config_dict.get("use_only_public_tests", False),
            enable_detailed_feedback=config_dict.get("enable_detailed_feedback", True),
        )


@dataclass
class LCBInstance:
    """LiveCodeBench 实例数据"""
    id: str  # question_id 或文件名
    question_title: str
    question_content: str
    platform: str  # leetcode, codeforces, atcoder
    question_id: str
    contest_id: str | None
    contest_date: str | None
    starter_code: str
    difficulty: str | None
    public_test_cases: list[dict]
    private_test_cases: list[dict]
    metadata: dict
    
    @classmethod
    def from_dict(cls, data: dict, file_path: Path | None = None) -> "LCBInstance":
        """从字典创建实例，支持从文件名推断 ID"""
        instance_id = data.get("question_id", "unknown")
        if file_path and instance_id == "unknown":
            instance_id = file_path.stem
        
        return cls(
            id=instance_id,
            question_title=data.get("question_title", ""),
            question_content=data.get("question_content", ""),
            platform=data.get("platform", "unknown"),
            question_id=data.get("question_id", ""),
            contest_id=data.get("contest_id"),
            contest_date=data.get("contest_date"),
            starter_code=data.get("starter_code", ""),
            difficulty=data.get("difficulty"),
            public_test_cases=data.get("public_test_cases", []),
            private_test_cases=data.get("private_test_cases", []),
            metadata=data.get("metadata", {}),
        )


class LiveCodeBenchRunner(BaseTaskRunner):
    """LiveCodeBench 代码生成任务的 TaskRunner 实现"""
    
    def __init__(
        self,
        *,
        task_config: dict[str, Any] | None = None,
        _logger: logging.Logger | None = None,
    ):
        self._logger = _logger or logging.getLogger(__name__)
        self._task_config = LCBTaskConfig.from_dict(task_config)
    
    # ------------------------------------------------------------------
    # 元数据提取（类方法，无需实例化）
    # ------------------------------------------------------------------
    
    @classmethod
    def load_metadata(cls, path: Path) -> TaskMetadata:
        """从任务数据文件中提取最小元数据"""
        data = json.loads(path.read_text(encoding="utf-8"))
        instance = LCBInstance.from_dict(data, path)
        return TaskMetadata(
            instance_id=instance.id,
            problem_description=instance.question_content,
        )
    
    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------
    
    def load_instance(self, path: Path) -> LCBInstance:
        """加载完整的任务数据"""
        data = json.loads(path.read_text(encoding="utf-8"))
        return LCBInstance.from_dict(data, path)
    
    # ------------------------------------------------------------------
    
    # 初始解
    # ------------------------------------------------------------------
    
    def get_initial_solution(self, instance_data: Any, config: Any) -> str:
        """提取或生成初始解
        
        对于 LiveCodeBench，返回 starter_code（可能为空字符串）。
        如果有 starter_code，模型可以在此基础上完成；
        如果为空，模型需要从零开始生成完整代码。
        """
        instance: LCBInstance = instance_data
        return instance.starter_code or ""
    
    # ------------------------------------------------------------------
    # 评估
    # ------------------------------------------------------------------
    
    def evaluate(
        self,
        solution: str,
        instance_data: Any,
        config: Any,
    ) -> tuple[float, dict[str, Any]]:
        """评估解的质量
        
        使用本地 perfagent/lcb_eval/testing_util.run_test() 进行评测。
        
        注意：
        - 返回原始通过率：1.0（所有测试通过）或 0.0（有测试失败）
        - Agent 层的 PerfAgentConfig.metric_higher_is_better 配置决定比较方向
        - problem_description 由 TaskMetadata 提供，不包含在 artifacts 中
        
        test_results 格式说明：
        - True：测试通过
        - -1：全局超时
        - -2：错误答案
        - -3：超时（TLE）
        - -4：运行时错误
        - -5：测试框架错误
        """
        instance: LCBInstance = instance_data
        
        # 1. 构建测试用例
        all_tests = []
        all_tests.extend(instance.public_test_cases)
        if not self._task_config.use_only_public_tests:
            all_tests.extend(instance.private_test_cases)
        
        if not all_tests:
            self._logger.warning(f"实例 {instance.id} 没有测试用例")
            return 1.0, {
                "passed": False,
                "test_results": [],
                "evaluation_metadata": {"error": "No test cases"},
            }
        
        inputs = [t.get("input", "") for t in all_tests]
        outputs = [t.get("output", "") for t in all_tests]
        
        # 2. 构建 testing_util 所需的样本格式
        fn_name = instance.metadata.get("func_name", None)
        sample = {
            "input_output": json.dumps({
                "inputs": inputs,
                "outputs": outputs,
                "fn_name": fn_name,
            })
        }
        
        # 3. 调用 testing_util.run_test()
        timeout = self._task_config.timeout
        try:
            test_results, eval_metadata = run_test(
                sample, 
                test=solution, 
                timeout=timeout
            )
        except Exception as e:
            self._logger.error(f"评测失败: {e}")
            return 0.0, {
                "passed": False,
                "test_results": [],
                "evaluation_metadata": {"error": str(e)},
            }
        
        # 4. 判断是否通过（所有测试用例都必须通过）
        # test_results 是一个列表，每个元素是测试用例的结果
        # True 表示通过，负数表示错误（-1=超时, -2=错误答案, -3=TLE, -4=运行时错误, -5=测试框架错误）
        # 使用 r > 0 来判断：True > 0 为 True，负数 > 0 为 False
        passed = all(r > 0 for r in test_results)
        
        # 5. 计算 metric：返回原始通过率（1.0=通过，0.0=不通过）
        # Agent 层的 metric_higher_is_better 配置决定比较方向
        metric = 1.0 if passed else 0.0
        
        # 6. 构建 artifacts
        # 注意：artifacts 只包含评估相关的补充信息
        # problem_description 已在顶层提供，不应包含在 artifacts 中
        artifacts = {
            "evaluation_metadata": eval_metadata,
        }
        
        # 7. 添加详细的失败信息（如果启用）
        if self._task_config.enable_detailed_feedback and not passed:
            failure_details = self._format_failure_details(test_results, eval_metadata)
            artifacts["failure_details"] = failure_details
        
        self._logger.info(
            f"实例 {instance.id}: {'通过' if passed else '未通过'}"
        )
        
        return metric, artifacts
    
    def _format_failure_details(self, results: list, metadata: dict) -> str:
        """格式化失败测试用例的详细信息"""
        details = ["## Failed Test Cases\n"]
        
        for i, result in enumerate(results):
            if result is not True:
                error_msg = "Unknown error"
                if isinstance(result, int):
                    error_map = {
                        -1: "Global timeout",
                        -2: "Wrong Answer",
                        -3: "Time Limit Exceeded",
                        -4: "Runtime Error",
                        -5: "Test framework error",
                    }
                    error_msg = error_map.get(result, f"Error code: {result}")
                elif isinstance(result, str):
                    error_msg = result
                
                details.append(f"- Test case {i+1}: {error_msg}")
        
        return "\n".join(details)
    
    # ------------------------------------------------------------------
    # Prompt 构建
    # ------------------------------------------------------------------
    
    def build_system_prompt(self, instance_data: Any, **context: Any) -> str:
        """构建系统 prompt
        
        Expected context keys:
            config: PerfAgentConfig 对象
            additional_requirements: str（可选，优先于 config）
            local_memory: str（可选）
            global_memory: str（可选）
        """
        instance: LCBInstance = instance_data
        config = context.get("config")
        
        # 获取 prompt 模板（必须有配置）
        if not config or not hasattr(config, "prompts"):
            raise ValueError("config.prompts 未配置，无法构建 system prompt")
        
        tmpl = getattr(config.prompts, "system_template", "")
        if not tmpl:
            raise ValueError("config.prompts.system_template 为空，无法构建 system prompt")
        
        # 构建 starter_code 代码块
        starter_code = instance.starter_code or ""
        
        
        # 从 context 或 config 获取 additional_requirements, local_memory, global_memory
        additional_requirements = (
            context.get("additional_requirements")
            or getattr(getattr(config, "prompts", None), "additional_requirements", None)
            or ""
        )
        local_memory = (
            context.get("local_memory")
            or getattr(getattr(config, "prompts", None), "local_memory", None)
            or ""
        )
        global_memory = (
            context.get("global_memory")
            or getattr(getattr(config, "prompts", None), "global_memory", None)
            or ""
        )

        # 填充模板
        try:
            return tmpl.format(
                question_content=instance.question_content,
                starter_code=starter_code,
                additional_requirements=additional_requirements,
                local_memory=local_memory,
                global_memory=global_memory,
            )
        except KeyError as e:
            raise ValueError(f"system_prompt 模板填充失败，缺少占位符: {e}")
    
    def build_optimization_prompt(
        self,
        solution: str,
        metric: float,
        artifacts: dict[str, Any],
        **context: Any,
    ) -> str:
        """构建优化 prompt"""
        config = context.get("config")
        
        # 获取 prompt 模板（必须有配置）
        if not config or not hasattr(config, "prompts"):
            raise ValueError("config.prompts 未配置，无法构建 optimization prompt")
        
        tmpl = getattr(config.prompts, "optimization_template", "")
        if not tmpl:
            raise ValueError("config.prompts.optimization_template 为空，无法构建 optimization prompt")
        
        # 准备模板变量
        passed = artifacts.get("passed", False)
        passed_count = sum(1 for r in artifacts.get("test_results", []) if r > 0)
        total_count = len(artifacts.get("test_results", []))
        failure_details = artifacts.get("failure_details", "")
        
        # 填充模板
        try:
            return tmpl.format(
                current_solution=solution,
                passed="通过" if passed else "未通过",
                passed_count=passed_count,
                total_count=total_count,
                failure_details=failure_details,
                metric=metric,
            )
        except KeyError as e:
            raise ValueError(f"optimization_prompt 模板填充失败，缺少占位符: {e}")
    
    # ------------------------------------------------------------------
    # 解提取
    # ------------------------------------------------------------------
    
    def extract_solution(self, llm_response: str, current_solution: str) -> str:
        """从 LLM 响应中提取新的解
        
        支持：
        1. ```python ... ``` 代码块（Python 语言标识符）
        2. ``` ... ``` 无语言标识符的代码块
        3. 直接返回代码（如果没有代码块）
        
        优先提取最后一个代码块（通常是最完整的版本）
        """
        # 尝试提取带 python 语言标识符的代码块
        python_blocks = re.findall(r"```python\n(.*?)```", llm_response, re.DOTALL)
        if python_blocks:
            return python_blocks[-1].strip()
        
        # 尝试提取带 python 语言标识符（大写）的代码块
        python_blocks_upper = re.findall(r"```PYTHON\n(.*?)```", llm_response, re.DOTALL)
        if python_blocks_upper:
            return python_blocks_upper[-1].strip()
        
        # 尝试提取无语言标识符的代码块
        code_blocks = re.findall(r"```\n(.*?)```", llm_response, re.DOTALL)
        if code_blocks:
            return code_blocks[-1].strip()
        
        # 如果没有代码块，检查是否是纯代码（没有 markdown 标记）
        # 简单的启发式：如果包含 import, def, class 等关键字
        if re.search(r"\b(import|def|class|if __name__)\b", llm_response):
            return llm_response.strip()
        
        # 都不匹配，返回当前解（保持不变）
        self._logger.warning("无法从响应中提取代码，返回当前解")
        return current_solution
