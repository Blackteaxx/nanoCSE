"""
EffiBench TaskRunner 实现

从 perfagent/agent.py 中提取的 EffiBench 代码性能优化任务特定逻辑。
负责加载 EffiBench 实例数据、评估代码性能、构建 prompt、从 LLM 响应提取解等。

设计要点:
- EffiBenchXInstance 数据类从 agent.py 迁移至此，agent.py 通过 re-export 保持向后兼容。
- EffiBenchRunner 实现 BaseTaskRunner 的全部抽象方法。
- evaluate() 返回 (metric, artifacts)，其中 metric 为优化目标值（越低越好），
  artifacts 包含评估结果详情（problem_description 由 TaskMetadata 提供）。
- EffiBenchTaskConfig 封装所有 EffiBench 特定配置，从 PerfAgentConfig.task_config 解析。
"""

from __future__ import annotations

import json
import logging
import re
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..diff_applier import DiffApplier
from ..effibench.benchmark import run_performance_benchmark
from ..effibench.utils import EFFIBENCH_REGISTRY
from ..protocols import TaskMetadata
from ..task_runner import BaseTaskRunner

# ---------------------------------------------------------------------------
# EffiBenchTaskConfig —— EffiBench 任务特定配置
# ---------------------------------------------------------------------------


@dataclass
class EffiBenchTaskConfig:
    """EffiBench 任务特定配置

    从 PerfAgentConfig.task_config 字典中解析，封装所有 EffiBench 专属参数。
    原先散布在 OptimizationConfig、RuntimeConfig、LanguageConfig 中的字段
    统一收拢于此。
    """

    # 优化方向
    target: str = "runtime"  # runtime | memory | integral
    code_generation_mode: str = "diff"  # diff | direct
    enable_memory_checks: bool = True
    enable_runtime_checks: bool = True
    include_other_metrics_in_summary: bool = True

    # 语言
    language: str = "python3"

    # 运行时资源限制
    time_limit: int = 20  # 秒
    memory_limit: int = 1024  # MB
    max_workers: int = 4
    num_runs: int = 10
    trim_ratio: float = 0.1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EffiBenchTaskConfig:
        """从字典创建，忽略未知键"""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# EffiBenchXInstance 数据类（从 perfagent/agent.py 迁移）
# ---------------------------------------------------------------------------


@dataclass
class EffiBenchXInstance:
    """EffiBench 扩展实例数据

    表示一个 EffiBench 代码性能优化问题实例，包含问题描述、
    测试用例、解决方案等信息。
    """

    id: str
    title: str
    title_slug: str
    description: str
    description_md: str
    source: str
    url: str
    type: str
    starter_code: str | None = None
    solutions: dict[str, dict[str, str]] = field(default_factory=dict)
    language: str | None = None
    generated_tests: list[dict[str, Any]] = field(default_factory=list)
    evaluator: str | None = None
    test_runners: dict[str, str] = field(default_factory=dict)
    # 任务名（来源于实例文件名，不含扩展名）
    task_name: str | None = None

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "EffiBenchXInstance":
        # Robustly parse generated_tests when it can be a list or a JSON string
        gt_raw = data.get("generated_tests", [])
        if isinstance(gt_raw, str):
            try:
                gt_parsed = json.loads(gt_raw)
            except Exception:
                gt_parsed = []
        elif isinstance(gt_raw, list):
            gt_parsed = gt_raw
        else:
            gt_parsed = []

        # Robustly parse test_runners when it can be a dict or a JSON string
        tr_raw = data.get("test_runners", {})
        if isinstance(tr_raw, str):
            try:
                tr_parsed = json.loads(tr_raw)
            except Exception:
                tr_parsed = {}
        elif isinstance(tr_raw, dict):
            tr_parsed = tr_raw
        else:
            tr_parsed = {}

        return EffiBenchXInstance(
            id=str(data.get("id", "unknown")),
            title=data.get("title", ""),
            title_slug=data.get("title_slug", ""),
            description=data.get("description", ""),
            description_md=data.get("description_md", ""),
            source=data.get("source", ""),
            url=data.get("url", ""),
            type=data.get("type", ""),
            starter_code=data.get("starter_code"),
            solutions=data.get("solutions", {}),
            language=data.get("language"),
            generated_tests=gt_parsed,
            evaluator=data.get("evaluator"),
            test_runners=tr_parsed,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "title_slug": self.title_slug,
            "description": self.description,
            "description_md": self.description_md,
            "source": self.source,
            "url": self.url,
            "type": self.type,
            "starter_code": self.starter_code,
            "solutions": self.solutions,
            "language": self.language,
            "generated_tests": self.generated_tests,
            "evaluator": self.evaluator,
            "test_runners": self.test_runners,
            "task_name": self.task_name,
        }


# ---------------------------------------------------------------------------
# EffiBenchRunner
# ---------------------------------------------------------------------------


class EffiBenchRunner(BaseTaskRunner):
    """EffiBench 代码性能优化任务的 TaskRunner 实现

    从 PerfAgent 核心类中提取的 EffiBench 任务特定逻辑，
    使 Agent 核心循环保持任务无关。

    职责:
    - 加载 EffiBench 实例数据（JSON → EffiBenchXInstance）
    - 提取或生成初始代码
    - 调用 benchmark 后端评估代码性能
    - 构建 system prompt 和 optimization prompt
    - 从 LLM 响应中提取代码（全量替换或 SEARCH/REPLACE diff）
    """

    def __init__(
        self,
        *,
        task_config: dict[str, Any] | None = None,
        code_generation_mode: str | None = None,
        _logger: logging.Logger | None = None,
    ):
        """
        Args:
            task_config: 任务特定配置字典（来自 PerfAgentConfig.task_config）
            code_generation_mode: 已弃用，使用 task_config 代替（向后兼容）
            _logger: 可选的 logger 实例
        """
        self._logger = _logger or logging.getLogger(__name__)

        # 解析任务特定配置
        tc = dict(task_config or {})
        # 向后兼容：如果传入了旧的 code_generation_mode 参数
        if code_generation_mode is not None and "code_generation_mode" not in tc:
            tc["code_generation_mode"] = code_generation_mode
        self._task_config = EffiBenchTaskConfig.from_dict(tc)

        self._code_generation_mode = self._task_config.code_generation_mode
        self._diff_applier = DiffApplier()

    # ==================================================================
    # BaseTaskRunner 实现
    # ==================================================================

    # ------------------------------------------------------------------
    # 元数据提取（类方法，无需实例化）
    # ------------------------------------------------------------------

    @classmethod
    def load_metadata(cls, path: Path) -> TaskMetadata:
        """从 EffiBench 实例 JSON 文件提取最小元数据"""
        data = json.loads(path.read_text(encoding="utf-8"))
        instance_id = path.stem or str(data.get("id", "unknown"))
        problem_description = data.get("description_md", "") or data.get("description", "")
        return TaskMetadata(
            instance_id=instance_id,
            problem_description=problem_description,
        )

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    def load_instance(self, path: Path) -> EffiBenchXInstance:
        """加载完整的 EffiBench 实例数据"""
        data = json.loads(path.read_text(encoding="utf-8"))
        instance = EffiBenchXInstance.from_dict(data)
        # 将文件名（不含扩展名）作为 task_name
        if not instance.task_name:
            instance.task_name = path.stem
        return instance

    # ------------------------------------------------------------------
    # 初始解
    # ------------------------------------------------------------------

    def get_initial_solution(self, instance_data: Any, config: Any) -> str:
        """提取或生成初始代码

        返回默认占位符代码（根据语言配置）。
        """
        language = self._task_config.language
        return self._get_default_placeholder(language)

    # ------------------------------------------------------------------
    # 评估
    # ------------------------------------------------------------------

    def evaluate(
        self,
        solution: str,
        instance_data: Any,
        config: Any,
    ) -> tuple[float, dict[str, Any]]:
        """评估代码性能

        执行级联评估（单次正确性检查 → 多次性能评估），
        返回标量指标（越低越好）和 artifacts 字典。

        Returns:
            (metric, artifacts) 元组:
            - metric: 优化目标值（runtime/memory/integral），越低越好
            - artifacts: 包含 benchmark_results 等评估详情
        """
        instance: EffiBenchXInstance = instance_data
        language = self._task_config.language
        target = self._task_config.target
        test_cases = instance.generated_tests or []

        # 执行性能评估
        benchmark_results = self._run_benchmark_cascade(
            language, solution, test_cases, instance
        )

        # 提取标量指标
        perf = benchmark_results.get("performance_analysis", {})
        raw_metric = perf.get(target, float("inf"))
        metric = self._clean_performance_value(raw_metric)
        pass_rate = self._extract_pass_rate(benchmark_results)

        # 关键: 若测试未全部通过, metric 强制为 inf（越低越好语义下不应采纳）
        if pass_rate < 1.0:
            metric = float("inf")

        unit = (
            "s"
            if target == "runtime"
            else ("MB" if target == "memory" else "MB*s")
        )

        artifacts: dict[str, Any] = {
            "language": language,
            "optimization_target": target,
            "performance_unit": unit,
            "pass_rate": pass_rate,
            "benchmark_results": benchmark_results,
            "final_metrics": {
                "runtime": perf.get("runtime", float("inf")),
                "memory": perf.get("memory", float("inf")),
                "integral": perf.get("integral", float("inf")),
            },
        }

        return metric, artifacts

    # ------------------------------------------------------------------
    # Prompt 构建
    # ------------------------------------------------------------------

    def build_system_prompt(self, instance_data: Any, **context: Any) -> str:
        """构建系统 prompt

        Expected context keys:
            config: PerfAgentConfig 对象
            language: str（可选，优先于 task_config）
            optimization_target: str（可选，优先于 task_config）
            additional_requirements: str（可选，优先于 config）
            local_memory: str（可选）
            global_memory: str（可选）
        """
        instance: EffiBenchXInstance = instance_data
        config = context.get("config")

        language = context.get("language") or self._task_config.language
        optimization_target = context.get("optimization_target") or self._task_config.target
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

        task_description = instance.description_md
        task_type = getattr(instance, "type", None)
        starter_code = self._resolve_starter_code(instance, language)
        allowed_imports_scope = EFFIBENCH_REGISTRY.get(language, {}).get("imports", "")
        is_functional = (task_type or "").lower() == "functional"

        # 从 config.prompts.system_template 获取模板，必须配置
        tmpl = (
            getattr(getattr(config, "prompts", None), "system_template", "")
            if config
            else ""
        )
        if not tmpl:
            raise ValueError(
                "config.prompts.system_template 未配置。"
                "请在 YAML 配置文件的 prompts.system_template 中提供系统模板。"
            )

        try:
            base = tmpl.format(
                language=language,
                optimization_target=optimization_target,
                task_description=task_description,
                additional_requirements=additional_requirements,
                local_memory=local_memory,
                global_memory=global_memory,
                allowed_imports_scope=allowed_imports_scope,
            )
        except KeyError as e:
            raise ValueError(
                f"system_template 格式化失败：缺少占位符 {e}。"
                f"可用变量: language, optimization_target, task_description, "
                f"additional_requirements, local_memory, global_memory, allowed_imports_scope"
            ) from e

        if is_functional and starter_code:
            starter_section = (
                "\n\n## Starter Code\n"
                "Use the following starter code as the exact framework for your solution.\n\n"
                f"```{language}\n"
                f"{starter_code}\n"
                "```\n\n"
                "- Implement the function with the exact signature (name, parameters, etc.) "
                "specified in the starter code.\n"
            )
            return base + starter_section
        return base

    def build_optimization_prompt(
        self,
        solution: str,
        metric: float,
        artifacts: dict[str, Any],
        **context: Any,
    ) -> str:
        """构建优化 prompt

        Expected context keys:
            config: PerfAgentConfig 对象
            language: str（可选，默认从 task_config 获取）
        """
        config = context.get("config")
        language = context.get("language") or artifacts.get("language") or self._task_config.language
        benchmark_results = artifacts.get("benchmark_results", {})

        code_gen_mode = self._task_config.code_generation_mode

        # 从 config.prompts.optimization_template 获取模板，必须配置
        tmpl = (
            getattr(getattr(config, "prompts", None), "optimization_template", "")
            if config
            else ""
        )
        if not tmpl:
            raise ValueError(
                "config.prompts.optimization_template 未配置。"
                "请在 YAML 配置文件的 prompts.optimization_template 中提供优化模板。"
            )

        if code_gen_mode == "direct":
            return tmpl

        # diff-based prompt 构建：先准备格式化变量，再填充模板
        target = artifacts.get("optimization_target") or self._task_config.target
        include_other = self._task_config.include_other_metrics_in_summary
        metrics_dict, artifacts_dict = self._build_metrics_and_artifacts(
            benchmark_results, target=target, include_other_metrics=include_other
        )
        metrics_md = self._format_metrics_md(metrics_dict)
        artifacts_md = self._format_artifacts_md(artifacts_dict)
        current_program_md = f"```\n{solution}\n```"

        try:
            return tmpl.format(
                current_program=current_program_md,
                current_metrics=metrics_md,
                current_artifacts_section=artifacts_md,
                language=language,
            )
        except KeyError as e:
            raise ValueError(
                f"optimization_template 格式化失败：缺少占位符 {e}。"
                f"可用变量: current_program, current_metrics, current_artifacts_section, language"
            ) from e

    # ------------------------------------------------------------------
    # 解提取
    # ------------------------------------------------------------------

    def extract_solution(self, llm_response: str, current_solution: str) -> str:
        """从 LLM 响应中提取新的代码解

        根据 code_generation_mode:
        - "direct": 提取 Markdown 代码块中的完整代码
        - "diff": 提取 SEARCH/REPLACE 块并应用到当前代码

        如果提取失败，返回 current_solution（不变）。
        """
        if not llm_response:
            return current_solution

        if self._code_generation_mode == "direct":
            code = self._extract_full_code_from_response(llm_response)
            return code if code else current_solution
        else:
            diff_text = self._extract_diff_from_response(llm_response)
            if diff_text:
                try:
                    return self._diff_applier.apply_diff(current_solution, diff_text)
                except Exception as e:
                    self._logger.warning(f"Diff 应用失败: {e}")
                    return current_solution
            return current_solution

    # ==================================================================
    # 公开辅助方法（供 Agent 在过渡期直接调用）
    # ==================================================================

    def build_metrics_and_artifacts(
        self,
        benchmark_results: dict[str, Any],
        target: str = "runtime",
        include_other_metrics: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """公开版本的 metrics/artifacts 构建（供外部调用）"""
        return self._build_metrics_and_artifacts(
            benchmark_results, target=target, include_other_metrics=include_other_metrics
        )

    def format_metrics_md(self, metrics: dict[str, Any]) -> str:
        """公开版本的 metrics Markdown 格式化"""
        return self._format_metrics_md(metrics)

    def format_artifacts_md(self, artifacts: dict[str, Any]) -> str:
        """公开版本的 artifacts Markdown 格式化"""
        return self._format_artifacts_md(artifacts)

    def extract_pass_rate(self, results: dict[str, Any]) -> float:
        """公开版本的通过率提取"""
        return self._extract_pass_rate(results)

    def clean_performance_value(self, val: Any) -> float:
        """公开版本的性能值清理"""
        return self._clean_performance_value(val)

    # ==================================================================
    # 私有辅助方法
    # ==================================================================

    # --- 语言 & 代码相关 ---

    @staticmethod
    def _normalize_language(lang: str | None) -> str:
        """标准化语言名称"""
        if not lang:
            return "python3"
        low = lang.lower()
        if low in ("python", "py", "python3"):
            return "python3"
        if low in ("cpp", "c++", "cxx"):
            return "cpp"
        if low in ("javascript", "js"):
            return "javascript"
        if low in ("java",):
            return "java"
        return low

    @staticmethod
    def _get_default_placeholder(language: str | None = None) -> str:
        """获取默认占位符代码（根据语言）"""
        lang = EffiBenchRunner._normalize_language(language)
        placeholder_map = {
            "python3": "# Start your code here\n",
            "cpp": "// Start your code here\n",
            "java": "// Start your code here\n",
            "javascript": "// Start your code here\n",
            "golang": "// Start your code here\n",
        }
        return placeholder_map.get(lang, "# Start your code here\n")

    @staticmethod
    def _resolve_starter_code(instance: EffiBenchXInstance, language: str) -> str | None:
        """解析 starter code"""
        sc = getattr(instance, "starter_code", None)
        if isinstance(sc, dict):
            try:
                return sc.get(language)
            except Exception:
                return None
        if isinstance(sc, str):
            return sc
        return None

    @staticmethod
    def _resolve_test_runner(instance: EffiBenchXInstance, language: str) -> str | None:
        """解析 test runner 代码"""
        trs = getattr(instance, "test_runners", None)
        lang_norm = EffiBenchRunner._normalize_language(language)
        if isinstance(trs, dict):
            try:
                val = trs.get(lang_norm)
                if isinstance(val, str) and val.strip():
                    return val
            except Exception:
                return None
        if isinstance(trs, str):
            try:
                parsed = json.loads(trs)
                if isinstance(parsed, dict):
                    val = parsed.get(lang_norm)
                    if isinstance(val, str) and val.strip():
                        return val
            except Exception:
                pass
        return None

    # --- 性能评估 ---

    def _run_benchmark_cascade(
        self,
        language: str,
        code: str,
        test_cases: list[dict[str, Any]],
        instance: EffiBenchXInstance,
    ) -> dict[str, Any]:
        """级联性能评估（单次正确性 → 多次性能）

        使用 self._task_config 中的运行时参数。
        返回原始的 benchmark 结果字典。
        """
        time_limit = self._task_config.time_limit
        memory_limit = self._task_config.memory_limit
        trim_ratio = self._task_config.trim_ratio
        max_workers = self._task_config.max_workers
        num_runs = self._task_config.num_runs

        eval_start_time = time.time()
        self._logger.info(
            f"[性能评估开始] 时间: {datetime.now().strftime('%H:%M:%S')}, "
            f"测试用例数: {len(test_cases)}, 代码长度: {len(code)} 字符"
        )

        # 占位符代码跳过
        if code == self._get_default_placeholder(language):
            self._logger.info("[性能评估跳过] 代码为占位符，返回默认失败结构")
            return self._create_default_performance_result(consistent=True)

        # evaluator / test_cases 校验
        evaluator = getattr(instance, "evaluator", None)
        tc_valid = (
            bool(test_cases)
            and isinstance(test_cases, list)
            and isinstance(test_cases[0], dict)
        )
        if not evaluator or not tc_valid:
            self._logger.warning(
                f"[性能评估跳过] 缺少必要组件 - evaluator: {bool(evaluator)}, "
                f"test_cases有效: {tc_valid}"
            )
            return self._create_default_performance_result(consistent=True)

        test_runner = self._resolve_test_runner(instance, language)

        # --- 单次预运行 ---
        single_run_start = time.time()
        self._logger.info("[单次预运行开始] 验证代码正确性...")
        try:
            single_run_summary = run_performance_benchmark(
                lang=language,
                solution=code,
                test_cases=test_cases,
                evaluator=evaluator,
                test_runner=test_runner,
                num_runs=1,
                time_limit=time_limit,
                memory_limit=memory_limit,
                trim_ratio=trim_ratio,
                max_workers=max_workers,
            )
            single_run_elapsed = time.time() - single_run_start
            pass_rate = single_run_summary.get("performance_analysis", {}).get(
                "pass_rate", 0
            )
            self._logger.info(
                f"[单次预运行完成] 耗时: {single_run_elapsed:.2f}s, "
                f"通过率: {pass_rate:.2%}"
            )
        except Exception as e:
            single_run_elapsed = time.time() - single_run_start
            self._logger.warning(
                f"[单次预运行失败] 耗时: {single_run_elapsed:.2f}s, "
                f"错误类型: {type(e).__name__}, 错误信息: {e}\n"
                f"{traceback.format_exc()}"
            )
            return self._create_default_performance_result(consistent=True)

        # 检查是否全部通过
        passed = single_run_summary.get("performance_analysis", {}).get(
            "passed", False
        )
        if not passed:
            total_elapsed = time.time() - eval_start_time
            failed_count = len(single_run_summary.get("failed_test_details", []))
            self._logger.info(
                f"[性能评估提前结束] 代码未全部通过测试，"
                f"失败用例数: {failed_count}, 总耗时: {total_elapsed:.2f}s"
            )
            return single_run_summary

        # 若重复运行次数为 1，直接返回
        if num_runs == 1:
            total_elapsed = time.time() - eval_start_time
            perf_analysis = single_run_summary.get("performance_analysis", {})
            self._logger.info(
                f"[性能评估完成] 单次运行模式, 总耗时: {total_elapsed:.2f}s, "
                f"runtime: {perf_analysis.get('runtime', 'N/A')}s, "
                f"memory: {perf_analysis.get('memory', 'N/A')}MB"
            )
            return single_run_summary

        # --- 多次性能评估 ---
        multi_run_start = time.time()
        self._logger.info(
            f"[多次评估开始] 运行次数: {num_runs}, "
            f"time_limit: {time_limit}s, memory_limit: {memory_limit}MB"
        )
        try:
            result = run_performance_benchmark(
                lang=language,
                solution=code,
                test_cases=test_cases,
                evaluator=evaluator,
                test_runner=test_runner,
                num_runs=num_runs,
                time_limit=time_limit,
                memory_limit=memory_limit,
                trim_ratio=trim_ratio,
                max_workers=max_workers,
            )
            multi_run_elapsed = time.time() - multi_run_start
            total_elapsed = time.time() - eval_start_time
            perf_analysis = result.get("performance_analysis", {})
            self._logger.info(
                f"[多次评估完成] 多次运行耗时: {multi_run_elapsed:.2f}s, "
                f"总评估耗时: {total_elapsed:.2f}s\n"
                f"  - runtime: {perf_analysis.get('runtime', 'N/A')}s (trimmed_mean)\n"
                f"  - memory: {perf_analysis.get('memory', 'N/A')}MB (trimmed_mean)\n"
                f"  - integral: {perf_analysis.get('integral', 'N/A')}MB*s\n"
                f"  - pass_rate: {perf_analysis.get('pass_rate', 0):.2%}"
            )
            return result
        except Exception as e:
            multi_run_elapsed = time.time() - multi_run_start
            total_elapsed = time.time() - eval_start_time
            self._logger.error(
                f"[多次评估失败] 多次运行耗时: {multi_run_elapsed:.2f}s, "
                f"总耗时: {total_elapsed:.2f}s, "
                f"错误类型: {type(e).__name__}, 错误信息: {e}\n"
                f"{traceback.format_exc()}"
            )
            return self._create_default_performance_result(consistent=False)

    @staticmethod
    def _create_empty_metric_analysis() -> dict[str, Any]:
        """创建空的单项指标分析结构"""
        return {
            "original_n": 0,
            "n": 0,
            "mean": float("inf"),
            "std": float("inf"),
            "min": float("inf"),
            "max": float("inf"),
            "max_diff": float("inf"),
            "95%_CI": (float("inf"), float("inf")),
            "trimmed_mean": float("inf"),
        }

    @classmethod
    def _create_empty_performance_metrics(cls) -> dict[str, Any]:
        """创建空的性能分析指标结构"""
        return {
            "original_n": 0,
            "n": 0,
            "runtime": float("inf"),
            "memory": float("inf"),
            "integral": float("inf"),
            "pass_rate": 0.0,
            "passed": False,
            "analysis": {
                "runtime": cls._create_empty_metric_analysis(),
                "memory": cls._create_empty_metric_analysis(),
                "integral": cls._create_empty_metric_analysis(),
            },
        }

    @classmethod
    def _create_default_performance_result(cls, consistent: bool = True) -> dict[str, Any]:
        """创建默认的性能评估结果结构"""
        return {
            "performance_analysis": cls._create_empty_performance_metrics(),
            "first_run_details": [],
            "failed_test_details": [],
            "failed_submission_exit_codes": [],
            "pass_rates": [],
            "pass_rate_consistent": consistent,
        }

    @staticmethod
    def _extract_pass_rate(results: dict[str, Any]) -> float:
        """从评估结果中提取通过率"""
        # 1. 尝试直接获取 pass_rate 字段
        pass_rate = results.get("performance_analysis", {}).get("pass_rate")
        try:
            if pass_rate is not None and isinstance(pass_rate, (int, float)):
                return float(pass_rate)
        except Exception:
            pass

        # 2. 尝试从 pass_rates 列表获取（取最小值，保守策略）
        pr_list = results.get("pass_rates")
        try:
            if isinstance(pr_list, list) and pr_list:
                return float(min(float(p) for p in pr_list))
        except Exception:
            pass

        # 3. 尝试从 first_run_details 计算
        try:
            fr = results.get("first_run_details") or []
            total = len(fr)
            passed = sum(1 for tc in fr if tc.get("passed", False))
            return (passed / total) if total > 0 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _clean_performance_value(val: Any) -> float:
        """清理性能指标值，转换为 float，处理 inf/nan"""
        if isinstance(val, (int, float)):
            return float(val)

        # 尝试处理 numpy 类型或 callable
        try:
            item_fn = getattr(val, "item", None)
            if callable(item_fn):
                val = item_fn()
        except Exception:
            pass

        if isinstance(val, str):
            s = val.strip().lower()
            if s in ("inf", "+inf", "infinity", "+infinity"):
                return float("inf")
            elif s in ("-inf", "-infinity"):
                return float("-inf")
            elif s == "nan":
                return float("nan")
            else:
                try:
                    return float(val)
                except Exception:
                    return float("inf")
        return float(val) if isinstance(val, (int, float)) else float("inf")

    # --- Prompt 辅助 ---

    @staticmethod
    def _build_metrics_and_artifacts(
        benchmark_results: dict[str, Any],
        target: str = "runtime",
        include_other_metrics: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """根据基准评估结果构造 metrics 与 artifacts 字典（用于 prompt 格式化）"""
        performance_metrics = benchmark_results.get("performance_analysis", {})
        failed_test_details = benchmark_results.get("failed_test_details", []) or []

        keys_to_include = {"runtime", "memory", "integral"}
        if not include_other_metrics:
            keys_to_include = {target}

        passed = performance_metrics.get("passed", False)
        if not passed:
            num_failed = len(failed_test_details)
            num_total = len(benchmark_results.get("first_run_details", []))
            pass_rate = (num_total - num_failed) / num_total if num_total > 0 else 0

            representative_failures: dict[str, Any] = {}
            for failure in failed_test_details:
                status = failure.get("status", "unknown")
                if status not in representative_failures:
                    representative_failures[status] = failure

            failure_details_summary: list[str] = []
            for status, failure in representative_failures.items():
                text = failure.get("text", "No additional error text.")
                if isinstance(text, str) and len(text) > 300:
                    text = text[-300:] + "..."
                failure_details_summary.append(
                    f"- Status: {status}, Details (last 300 chars of Output): {text}"
                )

            failures_text = "\n".join(failure_details_summary)
            all_statuses = ", ".join(representative_failures.keys())

            error_artifacts = {
                "error_type": f"SolutionFailedTests (statuses: {all_statuses})",
                "error_message": (
                    f"Solution passed {pass_rate:.2%} of test cases. "
                    f"Failure details:\n{failures_text}"
                ),
                "suggestion": (
                    "Review the solution to ensure it correctly handles "
                    "all test cases, including edge cases."
                ),
            }

            metrics: dict[str, Any] = {
                "pass_rate": pass_rate,
                "target": target,
                "error": (
                    f"Solution failed {num_failed} test case(s) with statuses: "
                    f"{all_statuses}. See artifacts for details."
                ),
            }
            for k in keys_to_include:
                metrics[k] = "Infinity"

            return metrics, error_artifacts

        # 成功情况
        metrics = {
            "pass_rate": 1.0,
            "target": target,
        }
        for k in keys_to_include:
            metrics[k] = performance_metrics.get(k, "Infinity")

        artifacts = {"details": "All test cases passed."}
        return metrics, artifacts

    @staticmethod
    def _format_metrics_md(metrics: dict[str, Any]) -> str:
        """将性能指标格式化为 Markdown 文本"""
        lines: list[str] = []

        pr = metrics.get("pass_rate")
        if pr is not None:
            try:
                pr_pct = f"{float(pr) * 100:.2f}%"
            except Exception:
                pr_pct = str(pr)
            lines.append(f"- Pass rate: {pr_pct}")

        def _fmt(val: Any, unit: str) -> str:
            if isinstance(val, (int, float)):
                if val == float("inf"):
                    return "Infinity"
                if val == float("-inf"):
                    return "-Infinity"
                return f"{float(val):.6f} {unit}"
            s = str(val).strip().lower()
            if s in ("inf", "+inf", "infinity", "+infinity"):
                return "Infinity"
            if s in ("-inf", "-infinity"):
                return "-Infinity"
            if s == "nan":
                return "NaN"
            try:
                return f"{float(val):.6f} {unit}"
            except Exception:
                return f"{val} {unit}"

        if "runtime" in metrics:
            lines.append(f"- Runtime: {_fmt(metrics.get('runtime'), 's')}")
        if "memory" in metrics:
            lines.append(f"- Memory: {_fmt(metrics.get('memory'), 'MB')}")
        if "integral" in metrics:
            lines.append(f"- Integral: {_fmt(metrics.get('integral'), 'MB*s')}")

        tgt = metrics.get("target")
        if tgt is not None:
            lines.append(f"- Target: {tgt}")

        err = metrics.get("error")
        if err:
            lines.append(f"- Error: {err}")

        return "\n".join(lines) if lines else "- No metrics available."

    @staticmethod
    def _format_artifacts_md(artifacts: dict[str, Any]) -> str:
        """将构件信息格式化为 Markdown 文本"""
        if not artifacts:
            return "- No artifacts available."
        lines: list[str] = []
        for k, v in artifacts.items():
            if isinstance(v, str) and "\n" in v:
                indented = "\n  ".join(v.split("\n"))
                lines.append(f"- {k}: {indented}")
            else:
                lines.append(f"- {k}: {v}")
        return "\n".join(lines)

    # --- 代码提取 ---

    @staticmethod
    def _extract_full_code_from_response(response: str) -> str:
        """从模型响应中提取完整代码（Markdown 代码块）"""
        if not response:
            return ""
        # 匹配 ```language ... ```
        pattern = r"```(?:\w+)?\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            # 返回最后一个匹配的代码块，通常是最终代码
            return matches[-1].strip()
        return ""

    @staticmethod
    def _extract_diff_from_response(response: str) -> str:
        """从模型响应中提取 SEARCH/REPLACE diff"""
        if not response:
            return ""
        if "<<<<<<< SEARCH" in response and ">>>>>>> REPLACE" in response:
            try:
                start_idx = response.find("<<<<<<< SEARCH")
                end_idx = response.rfind(">>>>>>> REPLACE")
                if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
                    return response[
                        start_idx : end_idx + len(">>>>>>> REPLACE")
                    ].strip()
            except Exception:
                return ""
        return ""
