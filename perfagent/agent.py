"""
PerfAgent 核心类

实现通用的迭代优化循环，通过 TaskRunner 插件支持多种任务类型。
Agent 不直接处理任务特定的数据结构，所有任务特定操作均委托给 TaskRunner。
"""

import logging
import time
import math
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import PerfAgentConfig

if TYPE_CHECKING:
    from .protocols import AgentRequest, AgentResult

from .llm_client import LLMClient
from .task_runner import BaseTaskRunner
from .trajectory import TrajectoryLogger
from .utils.log import get_se_logger

# 向后兼容 re-export：EffiBenchXInstance 已迁移到 perfagent/tasks/effibench.py，
# 此处 re-export 以保持所有现有 import 继续工作。
from .tasks.effibench import EffiBenchXInstance  # noqa: F401


@dataclass
class RunContext:
    """保存单次运行的上下文状态（通用，任务无关）

    Attributes:
        instance_data: 任务实例数据（不透明，由 TaskRunner 解释）
        trajectory: 轨迹记录器
        current_solution: 当前解
        best_solution: 最优解
        best_metric: 最优标量指标
        current_artifacts: 当前评估的 artifacts
        best_artifacts: 最优解对应的 artifacts
        optimization_history: 优化历史记录
        no_improve_count: 连续未改进次数
    
    Note:
        best_metric 的比较方向由 PerfAgentConfig.metric_higher_is_better 控制：
        - False（默认）：越小越好，如运行时间、错误率
        - True：越大越好，如准确率、通过率
    """

    instance_data: Any
    trajectory: TrajectoryLogger
    current_solution: str
    best_solution: str
    best_metric: float
    current_artifacts: dict[str, Any]
    best_artifacts: dict[str, Any]
    optimization_history: list[dict[str, Any]]
    no_improve_count: int = 0


class PerfAgent:
    """通用性能优化 Agent

    通过 TaskRunner 插件实现任务无关的迭代优化循环。
    使用 AgentRequest/AgentResult 协议与 SE_Perf 层通信。
    """

    def __init__(
        self,
        config: PerfAgentConfig,
        task_runner: BaseTaskRunner | None = None,
        token_log_path: str | None = None,
        io_log_path: str | None = None,
        iteration_index: int | None = None,
    ):
        self.config = config
        self.task_runner = task_runner

        # 简化逻辑：凭据存在即初始化 LLMClient，无需 use_llm 标志
        self.llm_client = None
        if self.config.model.api_base and self.config.model.api_key:
            client_cfg = {
                "name": self.config.model.name,
                "api_base": self.config.model.api_base,
                "api_key": self.config.model.api_key,
                "max_output_tokens": self.config.model.max_output_tokens,
                "request_timeout": self.config.model.request_timeout,
                "max_retries": self.config.model.max_retries,
                "retry_delay": self.config.model.retry_delay,
                "retry_backoff_factor": getattr(self.config.model, "retry_backoff_factor", 2.0),
                "retry_jitter": getattr(self.config.model, "retry_jitter", 0.5),
                "log_inputs_outputs": self.config.model.log_inputs_outputs,
                "log_sanitize": self.config.model.log_sanitize,
            }
            # 将 LLM I/O 独立写入 log_dir/llm_io.log
            io_log_file = Path(self.config.logging.log_dir) / "llm_io.log"
            self.llm_client = LLMClient(
                client_cfg,
                io_log_path=io_log_file,
                log_inputs_outputs=self.config.model.log_inputs_outputs,
                log_sanitize=self.config.model.log_sanitize,
                request_timeout=self.config.model.request_timeout,
                token_log_path=token_log_path,
                io_jsonl_path=io_log_path,
            )
            if iteration_index is not None:
                self.llm_client.set_iteration_index(iteration_index)

        # 设置日志：统一绑定到当前 run 的 log_dir 下的 perfagent.log
        # 同一算子多结果时（如 Plan 出 5 个 sol）会多次执行 PerfAgent，每次 log_dir 为 iteration_N/instance_id。
        # 必须用含迭代信息的 logger 名，否则复用同一 logger 会全部写到 iteration_1 的目录。
        log_dir_path = Path(self.config.logging.log_dir)
        parts = log_dir_path.parts
        logger_suffix = "_".join(parts[-2:]) if len(parts) >= 2 else log_dir_path.name
        agent_logger_name = f"perfagent.agent.{logger_suffix}"
        get_se_logger(
            agent_logger_name,
            log_dir_path / "perfagent.log",
            emoji="🔧",
            level=getattr(logging, self.config.logging.log_level.upper()),
            also_stream=False,
        )
        self.logger = logging.getLogger(agent_logger_name)

        # 优化历史
        self.optimization_history: list[dict[str, Any]] = []

    # ==================================================================
    # TaskRunner 管理
    # ==================================================================

    def _ensure_task_runner(self) -> BaseTaskRunner:
        """确保 TaskRunner 已设置。若构造时未提供，自动创建 EffiBenchRunner（向后兼容）。"""
        if self.task_runner is not None:
            return self.task_runner
        from .tasks.effibench import EffiBenchRunner

        self.task_runner = EffiBenchRunner(
            task_config=self.config.task_config,
            _logger=self.logger,
        )
        return self.task_runner

    @staticmethod
    def _get_instance_id(instance_data: Any) -> str:
        """从实例数据中提取 ID（尝试多个属性名）"""
        for attr in ("task_name", "id", "instance_id"):
            val = getattr(instance_data, attr, None)
            if val:
                return str(val)
        if isinstance(instance_data, dict):
            for key in ("task_name", "id", "instance_id"):
                if key in instance_data:
                    return str(instance_data[key])
        return "unknown"

    # ==================================================================
    # 主入口
    # ==================================================================

    def run(self, instance_data: Any) -> dict[str, Any]:
        """运行优化流程（通用入口）

        流程：初始化上下文 -> 优化循环 -> 生成结果
        第一次迭代即生成初始解并评估，不再有独立的"初始评估"步骤。

        Args:
            instance_data: 任务实例数据（不透明对象，由 TaskRunner 解释）。
                           向后兼容：可传入 EffiBenchXInstance。

        Returns:
            结果字典，兼容 AgentResult.from_dict
        """
        self._ensure_task_runner()
        run_start_time = time.time()
        instance_id = self._get_instance_id(instance_data)

        self.logger.info(
            f"\n{'#' * 70}\n"
            f"# [PerfAgent 运行开始]\n"
            f"# 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"# 实例: {instance_id}\n"
            f"# 最大迭代次数: {self.config.max_iterations}\n"
            f"# 模型: {self.config.model.name}\n"
            f"{'#' * 70}"
        )

        try:
            # 1. 初始化上下文
            init_start = time.time()
            ctx = self._init_run_context(instance_data)
            init_elapsed = time.time() - init_start
            self.logger.info(f"[上下文初始化完成] 耗时: {init_elapsed:.3f}s")

            # 2. 优化循环（第一次迭代即生成初始解并评估）
            loop_start = time.time()
            self._process_optimization_loop(ctx)
            loop_elapsed = time.time() - loop_start
            self.logger.info(f"[优化循环完成] 总耗时: {loop_elapsed:.2f}s ({loop_elapsed / 60:.1f}分钟)")

            # 3. 完成并生成结果
            result = self._finalize_run(ctx)

            run_elapsed = time.time() - run_start_time
            self.logger.info(
                f"\n{'#' * 70}\n"
                f"# [PerfAgent 运行成功完成]\n"
                f"# 实例: {instance_id}\n"
                f"# 总耗时: {run_elapsed:.2f}s ({run_elapsed / 60:.1f}分钟)\n"
                f"# 成功: {result.get('success', False)}\n"
                f"{'#' * 70}"
            )
            return result

        except Exception as e:
            run_elapsed = time.time() - run_start_time
            self.logger.error(
                f"\n{'!' * 70}\n"
                f"! [PerfAgent 运行失败]\n"
                f"! 实例: {instance_id}\n"
                f"! 运行耗时: {run_elapsed:.2f}s ({run_elapsed / 60:.1f}分钟)\n"
                f"! 错误类型: {type(e).__name__}\n"
                f"! 错误信息: {e}\n"
                f"! 堆栈跟踪:\n{traceback.format_exc()}\n"
                f"{'!' * 70}"
            )
            # 尝试记录错误轨迹
            try:
                if "ctx" in locals():
                    ctx.trajectory.finalize(
                        success=False, error_message=str(e), final_submission_code=ctx.best_solution
                    )
                    self.logger.info("[轨迹已保存] 错误轨迹记录完成")
            except Exception as traj_error:
                self.logger.warning(f"[轨迹保存失败] {type(traj_error).__name__}: {traj_error}")
            raise

    # ==================================================================
    # AgentRequest / AgentResult API
    # ==================================================================

    def run_with_request(self, request: "AgentRequest") -> "AgentResult":
        """接收 AgentRequest，返回 AgentResult。

        Args:
            request: AgentRequest 对象

        Returns:
            AgentResult
        """
        from .protocols import AgentRequest, AgentResult

        if not isinstance(request, AgentRequest):
            raise TypeError(f"不支持的请求类型: {type(request).__name__}，请使用 AgentRequest")

        runner = self._ensure_task_runner()

        # 应用请求中的覆盖参数
        if request.additional_requirements:
            self.config.prompts.additional_requirements = request.additional_requirements
        if request.local_memory:
            self.config.prompts.local_memory = request.local_memory
        if request.global_memory:
            self.config.prompts.global_memory = request.global_memory
        if request.output_dir:
            self.config.logging.trajectory_dir = str(request.output_dir)
            self.config.logging.log_dir = str(request.output_dir)

        try:
            # 通过 TaskRunner 加载实例
            instance_data = runner.load_instance(request.task_data_path)
            raw_result = self.run(instance_data)

            return AgentResult(
                instance_id=raw_result.get("instance_id", "unknown"),
                success=raw_result.get("success", False),
                solution=raw_result.get("solution", ""),
                metric=raw_result.get("metric", float("inf")),
                artifacts=raw_result.get("artifacts", {}),
                total_iterations=raw_result.get("total_iterations", 0),
                trajectory_file=raw_result.get("trajectory_file"),
                error=raw_result.get("error"),
            )
        except Exception as e:
            self.logger.error(f"[run_with_request 异常] {type(e).__name__}: {e}", exc_info=True)
            return AgentResult.from_error(instance_id="unknown", error=str(e))

    # ==================================================================
    # 初始化
    # ==================================================================

    def _init_run_context(self, instance_data: Any) -> RunContext:
        """初始化运行上下文"""
        runner = self._ensure_task_runner()
        instance_id = self._get_instance_id(instance_data)

        # 初始化轨迹记录器
        trajectory = TrajectoryLogger(
            instance_id,
            self.config.logging.trajectory_dir,
            log_dir=self.config.logging.log_dir,
        )

        # 通过 TaskRunner 构建 System Prompt
        system_prompt = runner.build_system_prompt(
            instance_data,
            config=self.config,
        )
        trajectory.add_history(role="system", content=system_prompt, message_type="system_prompt")

        # 通过 TaskRunner 获取初始解
        initial_solution = runner.get_initial_solution(instance_data, self.config)
        if initial_solution is None:
            raise ValueError("无法获取初始解")

        # 初始化历史
        self.optimization_history = []

        # 根据配置方向初始化 best_metric
        if self.config.metric_higher_is_better:
            # 越大越好：从负无穷开始
            initial_best_metric = -float("inf")
        else:
            # 越小越好：从正无穷开始（默认）
            initial_best_metric = float("inf")

        return RunContext(
            instance_data=instance_data,
            trajectory=trajectory,
            current_solution=initial_solution,
            best_solution=initial_solution,
            best_metric=initial_best_metric,
            current_artifacts={},
            best_artifacts={},
            optimization_history=self.optimization_history,
        )

    # ==================================================================
    # 优化循环
    # ==================================================================

    def _process_optimization_loop(self, ctx: RunContext):
        """执行优化循环"""
        remaining_iterations = self.config.max_iterations

        self.logger.info(
            f"\n[优化循环开始] 计划迭代次数: {remaining_iterations}, "
            f"max_iterations: {self.config.max_iterations}"
        )

        for iteration in range(remaining_iterations):
            current_iter_num = iteration + 1

            should_stop = self._process_single_iteration(ctx, current_iter_num)
            if should_stop:
                self.logger.info(f"[优化循环提前终止] 在第 {current_iter_num} 次迭代后停止")
                break

        self.logger.info(f"[优化循环结束] 共执行 {len(ctx.optimization_history)} 次迭代")

    def _process_single_iteration(self, ctx: RunContext, iteration_num: int) -> bool:
        """处理单次迭代。返回 True 表示应该停止循环。"""
        runner = self._ensure_task_runner()
        iteration_start_time = time.time()
        self.logger.info(
            f"\n{'=' * 60}\n[迭代 {iteration_num} 开始] 时间: {datetime.now().strftime('%H:%M:%S')}\n{'=' * 60}"
        )

        # 1. 通过 TaskRunner 构建优化 Prompt
        prompt_start = time.time()
        opt_prompt = runner.build_optimization_prompt(
            solution=ctx.current_solution,
            metric=ctx.best_metric,
            artifacts=ctx.current_artifacts,
            config=self.config,
        )
        prompt_elapsed = time.time() - prompt_start
        self.logger.debug(f"[Prompt构建] 耗时: {prompt_elapsed:.3f}s, Prompt长度: {len(opt_prompt)} 字符")

        step_id = ctx.trajectory.start_step(
            "generate_optimization",
            query=opt_prompt,
            code_snapshot=ctx.current_solution,
        )

        # 2. 调用 LLM
        llm_phase_start = time.time()
        system_prompt = runner.build_system_prompt(ctx.instance_data, config=self.config)
        optimization_response = self._call_llm(system_prompt, ctx.trajectory.history, opt_prompt)
        llm_phase_elapsed = time.time() - llm_phase_start

        # 3. 通过 TaskRunner 提取新解
        extract_start = time.time()
        new_solution = runner.extract_solution(optimization_response, ctx.current_solution)
        extract_elapsed = time.time() - extract_start

        # 4. 检查解是否变化
        code_changed = new_solution != ctx.current_solution
        self.logger.info(
            f"[解变更检查] 已变更: {code_changed}, "
            f"新解长度: {len(new_solution)} 字符"
        )

        if not code_changed:
            self._handle_no_change(ctx, step_id, optimization_response, iteration_num)
            ctx.no_improve_count += 1
            iteration_elapsed = time.time() - iteration_start_time
            self.logger.info(
                f"[迭代 {iteration_num} 结束] 解未变更, 跳过评估\n"
                f"  - LLM调用耗时: {llm_phase_elapsed:.2f}s ({llm_phase_elapsed / 60:.1f}分钟)\n"
                f"  - 解提取耗时: {extract_elapsed:.3f}s\n"
                f"  - 迭代总耗时: {iteration_elapsed:.2f}s ({iteration_elapsed / 60:.1f}分钟)\n"
                f"  - 连续未改进次数: {ctx.no_improve_count}"
            )
            if self.config.early_stop_no_improve and ctx.no_improve_count >= self.config.early_stop_no_improve:
                self.logger.info(f"[提前停止] 连续未改进达到阈值 {self.config.early_stop_no_improve}")
                return True
            return False

        # 5. 通过 TaskRunner 评估新解
        eval_phase_start = time.time()
        try:
            metric, artifacts = runner.evaluate(new_solution, ctx.instance_data, self.config)
            eval_phase_elapsed = time.time() - eval_phase_start

            # 更新上下文状态
            improved = self._update_run_context_after_eval(
                ctx, new_solution, metric, artifacts, iteration_num
            )

            # 记录步骤
            self._record_iteration_step(
                ctx,
                step_id,
                optimization_response,
                new_solution,
                metric,
                artifacts,
                iteration_num,
                improved,
            )

            if improved:
                ctx.no_improve_count = 0
            else:
                ctx.no_improve_count += 1

            # 输出迭代总结
            iteration_elapsed = time.time() - iteration_start_time
            self.logger.info(
                f"\n[迭代 {iteration_num} 完成] 时间: {datetime.now().strftime('%H:%M:%S')}\n"
                f"  ⏱️  时间分解:\n"
                f"      - LLM调用耗时: {llm_phase_elapsed:.2f}s ({llm_phase_elapsed / 60:.1f}分钟) "
                f"({llm_phase_elapsed / iteration_elapsed * 100:.1f}%)\n"
                f"      - 解提取耗时: {extract_elapsed:.3f}s\n"
                f"      - 评估耗时: {eval_phase_elapsed:.2f}s ({eval_phase_elapsed / 60:.1f}分钟) "
                f"({eval_phase_elapsed / iteration_elapsed * 100:.1f}%)\n"
                f"      - 迭代总耗时: {iteration_elapsed:.2f}s ({iteration_elapsed / 60:.1f}分钟)\n"
                f"  📊 metric: {metric}\n"
                f"  ✅ 结果: {'改进，已采纳' if improved else '未改进'}\n"
                f"  📈 连续未改进次数: {ctx.no_improve_count}"
            )

            if self.config.early_stop_no_improve and ctx.no_improve_count >= self.config.early_stop_no_improve:
                self.logger.info(f"[提前停止] 连续未改进达到阈值 {self.config.early_stop_no_improve}")
                return True

        except Exception as e:
            eval_phase_elapsed = time.time() - eval_phase_start
            iteration_elapsed = time.time() - iteration_start_time
            self.logger.error(
                f"[迭代 {iteration_num} 异常] 评估阶段出错\n"
                f"  - 错误类型: {type(e).__name__}\n"
                f"  - 错误信息: {e}\n"
                f"  - LLM调用耗时: {llm_phase_elapsed:.2f}s\n"
                f"  - 评估耗时(至异常): {eval_phase_elapsed:.2f}s\n"
                f"  - 迭代总耗时: {iteration_elapsed:.2f}s\n"
                f"  - 堆栈跟踪:\n{traceback.format_exc()}"
            )
            self._handle_evaluation_error(ctx, step_id, optimization_response, iteration_num, str(e))

        return False

    # ==================================================================
    # LLM 调用
    # ==================================================================

    def _call_llm(self, system_prompt: str, history: list[dict[str, Any]], user_prompt: str) -> str:
        """调用 LLM 获取响应

        Args:
            system_prompt: 系统 prompt（由 TaskRunner 构建）
            history: 对话历史
            user_prompt: 用户 prompt（优化指令）

        Returns:
            LLM 响应文本
        """
        messages = self._build_messages(system_prompt, history, user_prompt)

        if self.llm_client:
            llm_start_time = time.time()
            self.logger.info(
                f"[LLM调用开始] 时间: {datetime.now().strftime('%H:%M:%S')}, 模型: {self.config.model.name}"
            )
            try:
                response = self.llm_client.call_llm(
                    messages,
                    temperature=self.config.model.temperature,
                    max_tokens=self.config.model.max_output_tokens,
                    usage_context="perfagent.optimize",
                )
                llm_elapsed = time.time() - llm_start_time
                self.logger.info(
                    f"[LLM调用完成] 耗时: {llm_elapsed:.2f}s ({llm_elapsed / 60:.1f}分钟), "
                    f"响应长度: {len(response)} 字符"
                )
                return response
            except Exception as e:
                llm_elapsed = time.time() - llm_start_time
                self.logger.error(
                    f"[LLM调用失败] 耗时: {llm_elapsed:.2f}s, 错误类型: {type(e).__name__}, "
                    f"错误信息: {e}\n{traceback.format_exc()}"
                )
                raise
        self.logger.warning("[LLM未配置] LLM 客户端未初始化，跳过本次优化")
        return "LLM 未配置或不可用，跳过本次优化建议。请检查 API 配置。"

    # ==================================================================
    # 上下文更新 & 步骤记录
    # ==================================================================

    def _update_run_context_after_eval(
        self, ctx: RunContext, new_solution: str, metric: float, artifacts: dict[str, Any], iteration: int
    ) -> bool:
        """更新上下文并判断是否改进

        根据配置的 metric_higher_is_better 决定比较方向：
        - metric_higher_is_better=True: metric 越大越好
        - metric_higher_is_better=False: metric 越小越好（默认）
        """
        # 根据配置方向判断是否改进
        if self.config.metric_higher_is_better:
            improved = metric > ctx.best_metric  # 越大越好
        else:
            improved = metric < ctx.best_metric  # 越小越好（默认）

        # 如果最大迭代次数为 1，强制视为改进（即总是保存生成代码）
        if self.config.max_iterations == 1 and not improved:
            improved = True
            self.logger.info("单次迭代模式：强制采纳生成解作为最佳结果")

        # 记录历史
        # 处理 JSON unsafe 值 (Infinity, -Infinity, NaN)
        def _make_json_safe(val):
            if isinstance(val, float):
                if math.isinf(val) or math.isnan(val):
                    return None  # 将 Infinity/-Infinity/NaN 转换为 null
            return val

        metric_before_safe = _make_json_safe(ctx.best_metric)
        metric_after_safe = _make_json_safe(metric)
        improvement_safe = _make_json_safe(ctx.best_metric - metric) if metric_before_safe is not None and metric_after_safe is not None else None

        ctx.optimization_history.append(
            {
                "iteration": iteration,
                "metric_before": metric_before_safe,
                "metric_after": metric_after_safe,
                "improvement": improvement_safe,
                "success": improved,
            }
        )

        if improved:
            ctx.best_metric = metric
            ctx.best_solution = new_solution
            ctx.best_artifacts = artifacts
            self.logger.info(f"采用更优解: metric {ctx.best_metric}")
        else:
            self.logger.info(f"未改进: metric {metric} vs best {ctx.best_metric}")

        # 决定是否采用解
        if self.config.adopt_only_if_improved:
            if improved:
                ctx.current_solution = new_solution
            else:
                ctx.current_solution = ctx.best_solution
        else:
            ctx.current_solution = new_solution

        ctx.current_artifacts = artifacts
        return improved

    def _record_iteration_step(
        self,
        ctx: RunContext,
        step_id: str,
        response: str,
        new_solution: str,
        metric: float,
        artifacts: dict[str, Any],
        iteration: int,
        improved: bool,
    ):
        """记录迭代步骤到轨迹"""
        adopted = improved if self.config.adopt_only_if_improved else True

        summary_text = self._build_summary_text(
            iteration=iteration,
            code_changed=adopted,
            solution=ctx.current_solution,
            metric=metric,
            artifacts=artifacts,
        )

        ctx.trajectory.end_step(
            step_id,
            response=response,
            thought=("应用优化并完成评估" if adopted else "评估未改进，未采用优化"),
            code_changed=adopted,
            performance_metrics={"metric": metric, **(artifacts or {})},
            code_snapshot=ctx.current_solution,
            summary=summary_text,
        )

    # ==================================================================
    # 错误处理
    # ==================================================================

    def _handle_no_change(self, ctx: RunContext, step_id: str, response: str, iteration: int):
        """处理解未变更的情况"""
        summary = self._build_summary_text(
            iteration=iteration,
            code_changed=False,
            solution=ctx.current_solution,
            metric=ctx.best_metric,
            artifacts=ctx.current_artifacts,
        )
        ctx.trajectory.end_step(
            step_id,
            response=response,
            thought="解提取后未变化，跳过",
            code_changed=False,
            code_snapshot=ctx.current_solution,
            summary=summary,
        )
        self.logger.warning("解未发生变化，跳过此次迭代")

    def _handle_evaluation_error(
        self, ctx: RunContext, step_id: str, response: str, iteration: int, error_msg: str
    ):
        """处理评估异常"""
        summary = self._build_summary_text(
            iteration=iteration,
            code_changed=True,
            solution=ctx.current_solution,
            error_message=f"评估失败: {error_msg}",
        )
        ctx.trajectory.end_step(
            step_id,
            response=response,
            thought="评估阶段发生异常",
            code_changed=True,
            performance_metrics=None,
            error=f"评估失败: {error_msg}",
            code_snapshot=ctx.current_solution,
            summary=summary,
        )

    # ==================================================================
    # 结果汇总
    # ==================================================================

    def _finalize_run(self, ctx: RunContext) -> dict[str, Any]:
        """完成运行并生成最终结果"""
        finalize_start = time.time()
        self.logger.info(f"\n[结果汇总开始] 时间: {datetime.now().strftime('%H:%M:%S')}")

        instance_id = self._get_instance_id(ctx.instance_data)
        best_metric = ctx.best_metric
        executed_iterations = len(ctx.optimization_history)

        # 根据配置方向判断成功
        if self.config.metric_higher_is_better:
            # 越大越好：只要不是 -inf 就算成功
            success = bool(best_metric > -float("inf"))
        else:
            # 越小越好：只要不是 inf 就算成功（默认）
            success = bool(best_metric < float("inf"))

        # 构建最终 artifacts（不含 problem_description，由 TaskMetadata 提供）
        artifacts = dict(ctx.best_artifacts)

        # 记录最终轨迹
        trajectory_file = ctx.trajectory.finalize(
            success=success,
            final_performance={"metric": best_metric},
            final_submission_code=ctx.best_solution,
        )

        # 主结果（AgentResult 格式）
        final_result: dict[str, Any] = {
            "instance_id": instance_id,
            "success": success,
            "solution": ctx.best_solution,
            "metric": best_metric,
            "artifacts": artifacts,
            "total_iterations": executed_iterations,
            "trajectory_file": trajectory_file,
            "error": None,
        }

        # 统计优化历史
        successful_iterations = sum(1 for h in ctx.optimization_history if h.get("success", False))

        finalize_elapsed = time.time() - finalize_start
        self.logger.info(
            f"\n[优化结果总结]\n"
            f"  📋 基本信息:\n"
            f"      - 实例ID: {instance_id}\n"
            f"      - 执行迭代数: {executed_iterations}\n"
            f"      - 成功改进迭代数: {successful_iterations}\n"
            f"\n"
            f"  📈 性能变化:\n"
            f"      - 最终 metric: {best_metric}\n"
            f"      - 优化成功: {'✅ 是' if success else '❌ 否'}\n"
            f"\n"
            f"  📁 轨迹文件: {trajectory_file}\n"
            f"  ⏱️  结果汇总耗时: {finalize_elapsed:.3f}s"
        )

        return final_result

    # ==================================================================
    # 通用辅助方法
    # ==================================================================

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

    @staticmethod
    def _format_artifacts_md(artifacts: dict[str, Any]) -> str:
        """将 artifacts 字典格式化为 Markdown 文本"""
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

    def _build_summary_text(
        self,
        iteration: int,
        code_changed: bool,
        solution: str | None = None,
        metric: float | None = None,
        artifacts: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> str:
        """构建一步迭代的 Markdown 摘要文本"""
        parts: list[str] = [
            "## Program Update",
            f"- Iteration: {iteration}",
            f"- Code changed: {code_changed}",
        ]

        if metric is not None:
            parts.append(f"- Metric: {metric}")

        if error_message:
            parts.append(f"- Error: {error_message}")

        parts.append("")
        parts.append("## Current Solution")
        parts.append(solution or "")

        if artifacts:
            parts.append("")
            parts.append("## Current Artifacts")
            parts.append(self._format_artifacts_md(artifacts))

        return "\n".join(parts)

    def _build_messages(
        self, system_prompt: str, history: list[dict[str, Any]], user_prompt: str, limit: int = 200
    ) -> list[dict[str, str]]:
        """构建 LLM 消息列表"""
        use_all = bool(getattr(self.config.prompts, "include_all_history", False))
        if use_all:
            msgs: list[dict[str, str]] = []
            tail = history[-limit:] if len(history) > limit else history
            for h in tail:
                role = h.get("role")
                content = h.get("content", "")
                if role in ("system", "user", "assistant") and content:
                    msgs.append({"role": role, "content": content})
            return msgs
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
