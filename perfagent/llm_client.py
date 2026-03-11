"""
LLM 客户端，支持多种模型和API端点
"""

import json
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

try:
    from openai import APIConnectionError, APIError, APITimeoutError, BadRequestError, RateLimitError
except Exception:
    APIError = Exception
    RateLimitError = Exception
    APITimeoutError = Exception
    APIConnectionError = Exception
    BadRequestError = Exception

from .utils.log import get_se_logger


class LLMClient:
    """LLM客户端，支持多种模型和API端点"""

    def __init__(
        self,
        model_config: dict[str, Any],
        max_retries: int = 10,
        retry_delay: float = 1.5,
        retry_backoff_factor: float = 2.0,
        retry_jitter: float = 0.3,
        io_log_path: str | Path | None = None,
        log_inputs_outputs: bool = True,
        log_sanitize: bool = True,
        request_timeout: float = 60.0,
        token_log_path: str | Path | None = None,
        io_jsonl_path: str | Path | None = None,
    ):
        """
        初始化LLM客户端

        Args:
            model_config: 模型配置字典，包含name, api_base, api_key等
            max_retries: 最大重试次数
            retry_delay: 每次重试的等待秒数
            io_log_path: LLM 输入/输出日志文件路径
            log_inputs_outputs: 是否记录原始输入输出
            token_log_path: token 统计日志路径，优先使用显式参数，回退到环境变量
            io_jsonl_path: LLM I/O jsonl 日志路径，优先使用显式参数，回退到环境变量
        """
        self.config = model_config
        # 统一使用文件日志（带 emoji），与 IO 日志同目录
        self.io_log_path = Path(io_log_path) if io_log_path else Path("./logs/llm_io.log")
        # Logger 名称需含迭代信息（与 agent 一致），否则同一实例多迭代时复用同一 logger，日志全进 iteration_1
        parent_parts = self.io_log_path.parent.parts
        task_suffix = (
            "_".join(parent_parts[-2:]) if len(parent_parts) >= 2 else (self.io_log_path.parent.name or "default")
        )
        logger_name = f"perfagent.llm_client.{task_suffix}"
        get_se_logger(logger_name, self.io_log_path, emoji="🤖", also_stream=False)
        self.logger = logging.getLogger(logger_name)
        self.token_log_path = str(token_log_path) if token_log_path else os.getenv("SE_TOKEN_LOG_PATH")
        self._token_lock = threading.Lock()
        self.io_jsonl_path = str(io_jsonl_path) if io_jsonl_path else os.getenv("SE_LLM_IO_LOG_PATH")
        self._io_lock = threading.Lock()
        self._iteration_index: int | str | None = None

        # 优先使用配置中的增强参数
        self.max_retries = int(model_config.get("max_retries", max_retries))
        self.retry_delay = float(model_config.get("retry_delay", retry_delay))
        self.retry_backoff_factor = float(model_config.get("retry_backoff_factor", retry_backoff_factor))
        self.retry_jitter = float(model_config.get("retry_jitter", retry_jitter))
        self.log_inputs_outputs = bool(model_config.get("log_inputs_outputs", log_inputs_outputs))
        self.log_sanitize = bool(model_config.get("log_sanitize", log_sanitize))
        self.request_timeout = float(model_config.get("request_timeout", request_timeout))

        # 验证必需的配置参数
        required_keys = ["name", "api_base", "api_key"]
        missing_keys = [key for key in required_keys if key not in model_config]
        if missing_keys:
            raise ValueError(f"缺少必需的配置参数: {missing_keys}")

        # 初始化OpenAI客户端，遵循api_test.py的工作模式
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

    def _format_content_for_log(self, content: str | None, indent: int = 2) -> str:
        """将文本内容格式化为多行日志，保留真实换行并缩进。

        Args:
            content: 文本内容
            indent: 缩进空格数量

        Returns:
            友好的多行字符串，带缩进并保留换行
        """
        prefix = " " * indent
        if content is None:
            return f"{prefix}content: (None)"
        text = str(content)
        if text == "":
            return f"{prefix}content: (empty)"
        lines = text.splitlines() or [text]
        formatted = [f"{prefix}content:"]
        formatted.extend(f"{prefix}  {line}" for line in lines)
        return "\n".join(formatted)

    def _format_messages_for_log(self, messages: list[dict[str, str]], indent: int = 0) -> str:
        """将消息列表格式化为多行日志，保留真实换行并缩进内容。"""
        base_prefix = " " * indent
        out_lines = [f"{base_prefix}messages:"]
        for i, m in enumerate(messages, start=1):
            role = m.get("role", "unknown")
            out_lines.append(f"{base_prefix}  [{i}] role: {role}")
            out_lines.append(self._format_content_for_log(m.get("content"), indent=indent + 4))
        return "\n".join(out_lines)

    def call_llm(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int | None = None,
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
        from datetime import datetime

        # 使用配置中的max_output_tokens作为默认值
        if max_tokens is None:
            max_tokens = self.config.get("max_output_tokens", 4000)

        call_start_time = time.time()
        attempt = 0
        total_retry_wait_time = 0.0
        rate_limit_count = 0
        timeout_count = 0
        connection_error_count = 0
        last_err: Exception | None = None

        self.logger.info(
            f"[LLM请求开始] 时间: {datetime.now().strftime('%H:%M:%S')}, "
            f"消息数: {len(messages)}, max_tokens: {max_tokens}, context: {usage_context or 'default'}"
        )

        while attempt < self.max_retries:
            try:
                self.logger.debug(f"调用LLM: {len(messages)} 条消息, temp={temperature}, max_tokens={max_tokens}")

                # 记录原始输入
                if self.log_inputs_outputs:
                    try:
                        # 日志脱敏：移除可能的密钥与端点信息
                        model_name = self.config.get("name", "unknown")
                        api_base = self.config.get("api_base", "")
                        safe_api_base = "<redacted>" if self.log_sanitize and api_base else api_base
                        req_lines = [
                            "LLM Request:",
                            f"model: {model_name}",
                            f"temperature: {temperature}",
                            f"max_tokens: {max_tokens}",
                            f"api_base: {safe_api_base}",
                            self._format_messages_for_log(messages, indent=0),
                        ]
                        self.logger.info("\n".join(req_lines))
                    except Exception as log_e:
                        self.logger.error(f"记录请求失败: {log_e}")

                # 使用基本的OpenAI客户端调用，遵循api_test.py的工作模式
                # 不使用额外参数，避免服务器错误
                self.logger.debug(f"调用模型: {model_name}, max_tokens={max_tokens}")

                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # 检查响应是否有效 - 空 choices 视为可重试错误
                if not response.choices:
                    self.logger.warning(
                        f"API返回空choices (尝试 {attempt + 1}/{self.max_retries}), Response: {response}"
                    )
                    # 空响应时进行重试
                    attempt += 1  # 递增重试计数
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * attempt)
                        continue
                    else:
                        raise ValueError(
                            f"API返回空choices，已重试{self.max_retries}次。Response id: {response.id}, model: {response.model}"
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
                                "context": usage_context or "perfagent",
                                "model": self.config["name"],
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
                    if self.io_jsonl_path:
                        io_entry = {
                            "ts": time.time(),
                            "context": usage_context or "perfagent",
                            "model": self.config["name"],
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
                            with open(self.io_jsonl_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(io_entry, ensure_ascii=False) + "\n")
                except Exception:
                    pass

                # 记录原始输出
                if self.log_inputs_outputs:
                    try:
                        usage = getattr(response, "usage", None)
                        usage_dict = None
                        if usage:
                            usage_dict = {
                                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                                "completion_tokens": getattr(usage, "completion_tokens", None),
                                "total_tokens": getattr(usage, "total_tokens", None),
                            }
                        # 响应日志脱敏：不记录原始头信息
                        resp_lines = [
                            "LLM Response:",
                            self._format_content_for_log(content, indent=0),
                            "usage: " + json.dumps(usage_dict, ensure_ascii=False),
                        ]
                        self.logger.info("\n".join(resp_lines))
                    except Exception as log_e:
                        self.logger.error(f"记录响应失败: {log_e}")

                # 记录成功调用的统计信息
                total_elapsed = time.time() - call_start_time
                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

                self.logger.info(
                    f"[LLM请求成功] 总耗时: {total_elapsed:.2f}s ({total_elapsed / 60:.1f}分钟)\n"
                    f"  - 尝试次数: {attempt + 1}\n"
                    f"  - 重试等待时间: {total_retry_wait_time:.2f}s\n"
                    f"  - 实际API耗时: {total_elapsed - total_retry_wait_time:.2f}s\n"
                    f"  - 限流次数: {rate_limit_count}\n"
                    f"  - 超时次数: {timeout_count}\n"
                    f"  - 连接错误次数: {connection_error_count}\n"
                    f"  - Token使用: prompt={prompt_tokens}, completion={completion_tokens}, "
                    f"total={prompt_tokens + completion_tokens}\n"
                    f"  - 响应长度: {len(content) if content else 0} 字符"
                )

                return content

            except Exception as e:
                last_err = e
                attempt += 1
                should_retry = attempt < self.max_retries and self._is_retryable_error(e)

                # 分类错误类型
                error_type = type(e).__name__
                if isinstance(e, RateLimitError) or "rate limit" in str(e).lower():
                    rate_limit_count += 1
                    error_category = "限流(RateLimit)"
                elif isinstance(e, APITimeoutError) or "timeout" in str(e).lower():
                    timeout_count += 1
                    error_category = "超时(Timeout)"
                elif isinstance(e, APIConnectionError) or "connection" in str(e).lower():
                    connection_error_count += 1
                    error_category = "连接错误(Connection)"
                else:
                    error_category = "其他错误"

                elapsed_so_far = time.time() - call_start_time
                sleep_time = self._compute_sleep(attempt) if should_retry else 0

                try:
                    self.logger.warning(
                        f"[LLM调用失败] 第 {attempt}/{self.max_retries} 次尝试\n"
                        f"  - 错误类别: {error_category}\n"
                        f"  - 错误类型: {error_type}\n"
                        f"  - 错误信息: {e}\n"
                        f"  - 已耗时: {elapsed_so_far:.2f}s\n"
                        f"  - 将重试: {'是' if should_retry else '否'}\n"
                        f"  - 等待时间: {sleep_time:.2f}s\n"
                        f"  - 累计统计: 限流={rate_limit_count}, 超时={timeout_count}, 连接错误={connection_error_count}"
                    )
                except Exception:
                    pass

                if should_retry:
                    try:
                        time.sleep(sleep_time)
                        total_retry_wait_time += sleep_time
                    except Exception:
                        time.sleep(self.retry_delay)
                        total_retry_wait_time += self.retry_delay
                else:
                    break

        # 重试后仍失败，抛出最后一个错误
        total_elapsed = time.time() - call_start_time
        self.logger.error(
            f"[LLM调用最终失败] 总耗时: {total_elapsed:.2f}s\n"
            f"  - 尝试次数: {attempt}\n"
            f"  - 重试等待总时间: {total_retry_wait_time:.2f}s\n"
            f"  - 限流次数: {rate_limit_count}\n"
            f"  - 超时次数: {timeout_count}\n"
            f"  - 连接错误次数: {connection_error_count}\n"
            f"  - 最后错误: {type(last_err).__name__}: {last_err}"
        )
        assert last_err is not None
        raise last_err

    def call_with_system_prompt(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.3, max_tokens: int | None = None
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

        return self.call_llm(messages, temperature, max_tokens)
