from __future__ import annotations

import logging
import os
import threading
import uuid
from collections.abc import Callable
from pathlib import Path, PurePath

from rich.logging import RichHandler
from rich.text import Text

_SET_UP_LOGGERS: set[str] = set()
_ADDITIONAL_HANDLERS: dict[str, logging.Handler] = {}
_LOG_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Thread-scoped handler filtering for concurrent safety
# ---------------------------------------------------------------------------
_thread_local = threading.local()


class _ScopedFileFilter(logging.Filter):
    """Only emit log records when this handler's ID is active in the calling thread.

    When no scoping has been configured on the current thread (i.e.
    ``enter_handler_scope`` was never called), the filter defaults to *allow*
    so that single-threaded / CLI usage is unaffected.
    """

    def __init__(self, handler_id: str):
        super().__init__()
        self.handler_id = handler_id

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        active: set[str] | None = getattr(_thread_local, "active_handler_ids", None)
        if active is None:
            return True
        return self.handler_id in active


def enter_handler_scope(handler_id: str) -> None:
    """Mark *handler_id* as active for the current thread."""
    active: set[str] | None = getattr(_thread_local, "active_handler_ids", None)
    if active is None:
        active = set()
        _thread_local.active_handler_ids = active
    active.add(handler_id)


def exit_handler_scope(handler_id: str) -> None:
    """Remove *handler_id* from the current thread's active set."""
    active: set[str] | None = getattr(_thread_local, "active_handler_ids", None)
    if active is not None:
        active.discard(handler_id)

logging.TRACE = 5  # type: ignore
logging.addLevelName(logging.TRACE, "TRACE")  # type: ignore


def _interpret_level(level: int | str | None, *, default=logging.DEBUG) -> int:
    if not level:
        return default
    if isinstance(level, int):
        return level
    if level.isnumeric():
        return int(level)
    return getattr(logging, level.upper())


_STREAM_LEVEL = _interpret_level(os.environ.get("SWE_AGENT_LOG_STREAM_LEVEL"))
_INCLUDE_LOGGER_NAME_IN_STREAM_HANDLER = False

_THREAD_NAME_TO_LOG_SUFFIX: dict[str, str] = {}
"""Mapping from thread name to suffix to add to the logger name."""


def register_thread_name(name: str) -> None:
    """Register a suffix to add to the logger name for the current thread."""
    thread_name = threading.current_thread().name
    _THREAD_NAME_TO_LOG_SUFFIX[thread_name] = name


class _RichHandlerWithEmoji(RichHandler):
    def __init__(self, emoji: str, *args, **kwargs):
        """Subclass of RichHandler that adds an emoji to the log message."""
        super().__init__(*args, **kwargs)
        if not emoji.endswith(" "):
            emoji += " "
        self.emoji = emoji

    def get_level_text(self, record: logging.LogRecord) -> Text:
        level_name = record.levelname.replace("WARNING", "WARN")
        return Text.styled((self.emoji + level_name).ljust(10), f"logging.level.{level_name.lower()}")


def get_logger(name: str, *, emoji: str = "") -> logging.Logger:
    """Get logger. Use this instead of `logging.getLogger` to ensure
    that the logger is set up with the correct handlers.
    """
    thread_name = threading.current_thread().name
    if thread_name != "MainThread":
        name = name + "-" + _THREAD_NAME_TO_LOG_SUFFIX.get(thread_name, thread_name)
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        # Already set up
        return logger
    handler = _RichHandlerWithEmoji(
        emoji=emoji,
        show_time=bool(os.environ.get("SWE_AGENT_LOG_TIME", False)),
        show_path=False,
    )
    handler.setLevel(_STREAM_LEVEL)
    # Set to lowest level and only use stream handlers to adjust levels
    logger.setLevel(logging.TRACE)  # type: ignore
    logger.addHandler(handler)
    logger.propagate = False
    _SET_UP_LOGGERS.add(name)
    with _LOG_LOCK:
        for handler in _ADDITIONAL_HANDLERS.values():
            my_filter = getattr(handler, "my_filter", None)
            if my_filter is None:
                logger.addHandler(handler)
            elif isinstance(my_filter, str) and my_filter in name:
                logger.addHandler(handler)
            elif callable(my_filter) and my_filter(name):
                logger.addHandler(handler)
    if _INCLUDE_LOGGER_NAME_IN_STREAM_HANDLER:
        _add_logger_name_to_stream_handler(logger)
    return logger


def add_file_handler(
    path: PurePath | str,
    *,
    filter: str | Callable[[str], bool] | None = None,
    level: int | str = logging.TRACE,  # type: ignore[attr-defined]
    id_: str = "",
) -> str:
    """Adds a file handler to all loggers that we have set up
    and all future loggers that will be set up with `get_logger`.

    A ``_ScopedFileFilter`` is attached so that in concurrent (multi-thread)
    scenarios each thread only writes to the file handlers it owns.  Call
    ``enter_handler_scope`` / ``exit_handler_scope`` (or the higher-level
    ``cleanup_file_handler``) to manage scoping.

    Args:
        filter: If str: Check that the logger name contains the filter string.
            If callable: Check that the logger name satisfies the condition returned by the callable.
        level: The level of the handler.
        id_: The id of the handler. If not provided, a random id will be generated.

    Returns:
        The id of the handler. This can be used to remove the handler later.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(_interpret_level(level))
    if not id_:
        id_ = str(uuid.uuid4())
    handler.addFilter(_ScopedFileFilter(id_))
    enter_handler_scope(id_)
    with _LOG_LOCK:
        for name in _SET_UP_LOGGERS:
            if filter is not None:
                if isinstance(filter, str) and filter not in name:
                    continue
                if callable(filter) and not filter(name):
                    continue
            logger = logging.getLogger(name)
            logger.addHandler(handler)
    handler.my_filter = filter  # type: ignore
    _ADDITIONAL_HANDLERS[id_] = handler
    return id_


def remove_file_handler(id_: str) -> None:
    """Remove a file handler by its id."""
    handler = _ADDITIONAL_HANDLERS.pop(id_, None)
    if handler is None:
        return
    with _LOG_LOCK:
        for log_name in _SET_UP_LOGGERS:
            logger = logging.getLogger(log_name)
            logger.removeHandler(handler)
    try:
        handler.close()
    except Exception:
        pass


def cleanup_file_handler(id_: str) -> None:
    """Exit handler scope for this thread and remove the handler globally.

    Safe to call multiple times or with an invalid *id_*.
    """
    exit_handler_scope(id_)
    remove_file_handler(id_)


def _add_logger_name_to_stream_handler(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        if isinstance(handler, _RichHandlerWithEmoji):
            formatter = logging.Formatter("[%(name)s] %(message)s")
            handler.setFormatter(formatter)


def add_logger_names_to_stream_handlers() -> None:
    """Add the logger name to the stream handler for all loggers that we have set up."""
    global _INCLUDE_LOGGER_NAME_IN_STREAM_HANDLER
    _INCLUDE_LOGGER_NAME_IN_STREAM_HANDLER = True
    with _LOG_LOCK:
        for logger in _SET_UP_LOGGERS:
            _add_logger_name_to_stream_handler(logging.getLogger(logger))


def set_stream_handler_levels(level: int) -> None:
    """Set the default stream level and adjust the levels of all stream handlers
    to be at most the given level.

    Note: Can only be used to lower the level, not raise it.
    """
    global _STREAM_LEVEL
    _STREAM_LEVEL = level
    with _LOG_LOCK:
        for name in _SET_UP_LOGGERS:
            logger = logging.getLogger(name)
            for handler in logger.handlers:
                if isinstance(handler, _RichHandlerWithEmoji):
                    current_level = handler.level
                    if current_level < level:
                        handler.setLevel(level)
