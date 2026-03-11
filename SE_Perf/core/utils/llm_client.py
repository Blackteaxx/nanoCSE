#!/usr/bin/env python3
"""
LLM客户端模块
为SE框架提供统一的LLM调用接口
"""

import json
import os
import random
import re
import threading
import time
from typing import Any  # noqa: UP035

from openai import OpenAI

try:
    from openai import APIConnectionError, APIError, APITimeoutError, BadRequestError, RateLimitError
except Exception:
    APIError = Exception
    RateLimitError = Exception
    APITimeoutError = Exception
    APIConnectionError = Exception
    BadRequestError = Exception

from core.utils.se_logger import get_se_logger


class LLMClient:
    """LLM客户端，支持多种模型和API端点"""

    def __init__(
        self,
        model_config: dict[str, Any],
        token_log_path: str | None = None,
        io_log_path: str | None = None,
    ):
        """
        初始化LLM客户端

        Args:
            model_config: 模型配置字典，包含name, api_base, api_key等
            token_log_path: token 统计日志路径，优先使用显式参数，回退到环境变量
            io_log_path: LLM I/O 日志路径，优先使用显式参数，回退到环境变量
        """
        self.config = model_config
        self.logger = get_se_logger("llm_client", emoji="🤖")
        self.token_log_path = token_log_path or os.getenv("SE_TOKEN_LOG_PATH")
        self._token_lock = threading.Lock()
        self.io_log_path = io_log_path or os.getenv("SE_LLM_IO_LOG_PATH")
        self._io_lock = threading.Lock()
        self._iteration_index: int | str | None = None

        # 验证必需的配置参数
        required_keys = ["name", "api_base", "api_key"]
        missing_keys = [key for key in required_keys if key not in model_config]
        if missing_keys:
            raise ValueError(f"缺少必需的配置参数: {missing_keys}")

        # 请求控制参数（带默认值）
        self.request_timeout: float = float(self.config.get("request_timeout", 600.0))
        self.max_retries: int = int(self.config.get("max_retries", 10))
        self.retry_delay: float = float(self.config.get("retry_delay", 1.5))
        self.retry_backoff_factor: float = float(self.config.get("retry_backoff_factor", 2.0))
        self.retry_jitter: float = float(self.config.get("retry_jitter", 0.3))

        # 初始化OpenAI客户端，遵循api_test.py的工作模式，并设置超时
        self.client = OpenAI(
            api_key=self.config["api_key"],
            base_url=self.config["api_base"],
            timeout=self.request_timeout,
        )

        self.logger.info(f"初始化LLM客户端: {self.config['name']}")

    def set_iteration_index(self, idx: int | str | None) -> None:
        """设置当前迭代索引（线程安全的替代环境变量方案）。"""
        self._iteration_index = idx

    def _get_iteration_index(self) -> int | str | None:
        """获取迭代索引，优先使用实例属性，回退到环境变量。"""
        if self._iteration_index is not None:
            return self._iteration_index
        env = os.getenv("SE_ITERATION_INDEX")
        if env is not None:
            try:
                return int(env)
            except Exception:
                return env
        return None

    def _is_retryable_error(self, e: Exception) -> bool:
        if isinstance(e, (RateLimitError, APITimeoutError, APIConnectionError)):
            return True
        if isinstance(e, APIError):
            status = getattr(e, "status_code", None) or getattr(e, "status", None)
            if isinstance(status, int) and status in (408, 429, 500, 502, 503, 504):
                return True
        msg = str(e).lower()
        if any(x in msg for x in ("rate limit", "timed out", "timeout", "temporarily unavailable", "connection")):
            return True
        if isinstance(e, BadRequestError):
            return False
        return False

    def _compute_sleep(self, attempt_index: int) -> float:
        base = self.retry_delay * (self.retry_backoff_factor ** max(0, attempt_index - 1))
        jitter = random.uniform(0, self.retry_jitter)
        return base + jitter

    def clean_think_tags(self, text: str) -> str:
        """移除 <think>...</think> 标签及其中内容，并去除首尾空白"""
        try:
            return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        except Exception:
            return text

    def call_llm(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
        usage_context: str | None = None,
    ) -> str:
        """
        调用LLM并返回响应内容

        Args:
            messages: 消息列表，每个消息包含role和content
            temperature: 温度参数，控制输出随机性
            max_tokens: 最大输出token数，None表示使用配置默认值

        Returns:
            LLM响应的文本内容

        Raises:
            Exception: LLM调用失败时抛出异常
        """
        # 使用配置中的 max_output_tokens 作为默认值
        if max_tokens is None:
            max_tokens = self.config.get("max_output_tokens", 4000)

        attempt = 0
        last_err: Exception | None = None
        while attempt < self.max_retries:
            try:
                self.logger.debug(f"调用LLM: {len(messages)} 条消息, temp={temperature}, max_tokens={max_tokens}")

                # 规范化模型名：仅移除 openai/ 前缀，其他保持原样
                raw_name = self.config.get("name", "")
                if isinstance(raw_name, str) and raw_name.startswith("openai/"):
                    model_name = raw_name.split("/", 1)[1]
                else:
                    model_name = raw_name

                # 使用 OpenAI 客户端调用，传入必需的参数
                # 生成可选的 extra_body，用于控制是否启用思考模板
                extra_body: dict[str, Any] = {}
                if enable_thinking is not None:
                    extra_body = {"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}}

                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **({"extra_body": extra_body} if extra_body else {}),
                )

                # 提取响应内容
                content = response.choices[0].message.content

                # 记录使用情况
                if getattr(response, "usage", None):
                    self.logger.debug(
                        f"Token使用: 输入={getattr(response.usage, 'prompt_tokens', '未知')}, "
                        f"输出={getattr(response.usage, 'completion_tokens', '未知')}, "
                        f"总计={getattr(response.usage, 'total_tokens', '未知')}"
                    )
                    try:
                        if self.token_log_path:
                            entry = {
                                "ts": time.time(),
                                "context": usage_context or "se_perf",
                                "model": model_name,
                                "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                                "completion_tokens": getattr(response.usage, "completion_tokens", None),
                                "total_tokens": getattr(response.usage, "total_tokens", None),
                                "messages_chars": sum(
                                    len(str(m.get("content", ""))) for m in messages if isinstance(m, dict)
                                ),
                            }
                            iter_idx = self._get_iteration_index()
                            if iter_idx is not None:
                                entry["iteration_index"] = iter_idx
                            with self._token_lock:
                                with open(self.token_log_path, "a", encoding="utf-8") as f:
                                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    except Exception:
                        pass

                try:
                    if self.io_log_path:
                        io_entry = {
                            "ts": time.time(),
                            "context": usage_context or "se_perf",
                            "model": model_name,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "attempt_index": attempt,
                            "messages": messages,
                            "response": content,
                        }
                        iter_idx = self._get_iteration_index()
                        if iter_idx is not None:
                            io_entry["iteration_index"] = iter_idx
                        if getattr(response, "usage", None):
                            io_entry["usage"] = {
                                "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                                "completion_tokens": getattr(response.usage, "completion_tokens", None),
                                "total_tokens": getattr(response.usage, "total_tokens", None),
                            }
                        with self._io_lock:
                            with open(self.io_log_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(io_entry, ensure_ascii=False) + "\n")
                except Exception:
                    pass

                return content

            except Exception as e:
                last_err = e
                attempt += 1
                should_retry = attempt < self.max_retries and self._is_retryable_error(e)
                self.logger.warning(f"LLM调用失败: {e}; attempt={attempt}/{self.max_retries}; retry={should_retry}")
                if should_retry:
                    time.sleep(self._compute_sleep(attempt))
                else:
                    break

        assert last_err is not None
        raise last_err

    def call_with_system_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        usage_context: str | None = None,
    ) -> str:
        """
        使用系统提示词和用户提示词调用LLM

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            temperature: 温度参数
            max_tokens: 最大输出token数

        Returns:
            LLM响应的文本内容
        """
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        return self.call_llm(messages, temperature, max_tokens, usage_context=usage_context)

    @classmethod
    def from_se_config(cls, se_config: dict[str, Any], use_operator_model: bool = False) -> "LLMClient":
        """
        从SE框架配置创建LLM客户端

        Args:
            se_config: SE框架配置字典
            use_operator_model: 是否使用operator_models配置而不是主模型配置

        Returns:
            LLM客户端实例
        """
        if use_operator_model and "operator_models" in se_config:
            model_config = se_config["operator_models"]
        else:
            model_config = se_config["model"]

        return cls(model_config)


class TrajectorySummarizer:
    """专门用于轨迹总结的LLM客户端包装器"""

    def __init__(self, llm_client: LLMClient, prompt_config: dict[str, Any] | None = None):
        """
        初始化轨迹总结器

        Args:
            llm_client: LLM客户端实例
        """
        self.llm_client = llm_client
        self.logger = get_se_logger("traj_summarizer", emoji="📊")
        self.prompt_config = prompt_config or {}

    def summarize_trajectory(
        self,
        trajectory_content: str,
        solution_content: str,
        iteration: int,
        problem_description: str | None = None,
        best_solution_text: str | None = None,
        target_solution_text: str | None = None,
    ) -> dict[str, Any]:
        """
        使用LLM总结轨迹内容

        Args:
            trajectory_content: .tra文件内容
            solution_content: 解/代码文本
            iteration: 迭代次数
            problem_description: 问题描述（可选，将并入提示词）

        Returns:
            轨迹总结字典
        """
        from .traj_summarizer import TrajSummarizer

        summarizer = TrajSummarizer(self.prompt_config)

        # 获取提示词
        system_prompt = summarizer.get_system_prompt()
        user_prompt = summarizer.format_user_prompt(
            trajectory_content,
            solution_content,
            problem_description,
            best_solution=best_solution_text,
            target_solution=target_solution_text,
        )

        self.logger.info(f"开始LLM轨迹总结 (迭代{iteration})")
        self.logger.debug(f"LLM系统提示词 (迭代{iteration}):\n{system_prompt}")
        self.logger.debug(f"LLM用户提示词 (迭代{iteration}):\n{user_prompt}")

        # 重试机制：解析失败或调用失败时重试，总次数3次
        last_error: str | None = None
        for attempt in range(1, 4):
            try:
                response = self.llm_client.call_with_system_prompt(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.7,
                    max_tokens=None,  # 使用配置中的 max_output_tokens
                    usage_context="traj_summarizer",
                )
                self.logger.debug(f"LLM原始响应 (迭代{iteration}, 第{attempt}次):\n{response}")
                # 去除思考内容
                response = self.llm_client.clean_think_tags(response)
                self.logger.debug(f"LLM清理后响应 (迭代{iteration}, 第{attempt}次):\n{response}")

                summary = summarizer.parse_response(response)
                summarizer.validate_response_format(summary)

                self.logger.info(f"LLM轨迹总结成功 (迭代{iteration}, 第{attempt}次)")
                return summary

            except json.JSONDecodeError as e:
                last_error = "json_decode_error"
                self.logger.warning(f"LLM轨迹总结解析失败: JSON解析错误 (迭代{iteration}, 第{attempt}次): {e}")
            except ValueError as e:
                last_error = "invalid_response_format"
                self.logger.warning(
                    f"LLM轨迹总结解析失败: 响应格式错误或无有效JSON片段 (迭代{iteration}, 第{attempt}次): {e}"
                )
            except Exception as e:
                last_error = "llm_call_failed"
                self.logger.warning(f"LLM轨迹总结调用失败 (迭代{iteration}, 第{attempt}次): {e}")

        # 所有重试失败，返回备用总结
        if last_error:
            self.logger.error(f"LLM轨迹总结最终失败 (迭代{iteration}): {last_error}")
        return summarizer.create_fallback_summary(trajectory_content, solution_content or "", iteration)
