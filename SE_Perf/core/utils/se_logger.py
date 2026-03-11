#!/usr/bin/env python3
"""
SE框架日志配置模块

基于SWE-agent现有日志系统，为SE框架提供统一的日志管理。
日志文件保存在每次运行的output_dir下，确保不会重叠覆盖。

在并发场景（如 evolve_server 多线程处理请求）中，通过
线程作用域过滤机制保证每个请求的日志只写入自己的文件。
"""

from pathlib import Path

from .log import add_file_handler, cleanup_file_handler, get_logger


def setup_se_logging(output_dir: str | Path) -> tuple[str, str]:
    """为SE框架设置日志系统。

    Args:
        output_dir: 输出目录路径

    Returns:
        ``(log_file_path, handler_id)`` 二元组。
        *handler_id* 用于事后调用 :func:`cleanup_se_logging` 释放资源。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = output_dir / "se_framework.log"

    handler_id = add_file_handler(
        log_file_path,
        filter="SE",
        level="DEBUG",
    )

    return str(log_file_path), handler_id


def cleanup_se_logging(handler_id: str | None) -> None:
    """清理由 :func:`setup_se_logging` 创建的日志 handler。

    - 将 *handler_id* 从当前线程的活跃作用域中移除
    - 从所有 logger 上卸载该 handler 并关闭文件

    安全地接受 ``None``（直接返回）。
    """
    if handler_id is None:
        return
    cleanup_file_handler(handler_id)


def get_se_logger(module_name: str, emoji: str = "📋") -> object:
    """获取SE框架专用logger。

    Args:
        module_name: 模块名称（如 ``"SE.core.utils"``）
        emoji: 显示用的emoji

    Returns:
        配置好的 logger 对象
    """
    if not module_name.startswith("SE"):
        module_name = f"SE.{module_name}"

    return get_logger(module_name, emoji=emoji)
