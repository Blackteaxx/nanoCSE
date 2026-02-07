#!/usr/bin/env python3
"""
SE Operators Base Classes

定义了所有算子的基类和核心接口。
算子是模块化的、可重用的组件，用于执行特定的轨迹操作，如生成、交叉或过滤。
"""

from __future__ import annotations

import abc
import random
import re
from dataclasses import dataclass, field
from typing import Any

from core.utils.llm_client import LLMClient
from core.utils.se_logger import get_se_logger
from core.utils.traj_pool_manager import TrajPoolManager
from perf_config import StepConfig


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryItem:
    """轨迹池中单条轨迹的结构化表示。

    Attributes:
        label: 轨迹标签（如 "sol1", "iter3"）。
        metric: 标量性能指标（越低越好）。
        solution: 解/代码文本。
        summary: 轨迹摘要（可为 dict 或 str）。
        extras: 其他未列举的字段（保留原始 JSON 中的所有额外字段）。
    """

    label: str = ""
    metric: float | None = None
    solution: str = ""
    summary: Any = None
    extras: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "TrajectoryItem":
        known = {"label", "metric", "solution", "summary"}
        metric_raw = d.get("metric")
        try:
            metric = float(metric_raw) if metric_raw is not None else None
        except (ValueError, TypeError):
            metric = None
        return TrajectoryItem(
            label=str(d.get("label") or ""),
            metric=metric,
            solution=str(d.get("solution") or ""),
            summary=d.get("summary"),
            extras={k: v for k, v in d.items() if k not in known},
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "label": self.label,
            "solution": self.solution,
        }
        if self.metric is not None:
            out["metric"] = self.metric
        if self.summary is not None:
            out["summary"] = self.summary
        out.update(self.extras)
        return out


@dataclass
class InstanceTrajectories:
    """实例在轨迹池中的所有轨迹数据（结构化）。

    既提供类型安全的属性访问，也兼容 dict 风格的 get/items 操作
    以便算子代码平滑迁移。

    Attributes:
        problem: 问题描述文本（来自轨迹池 "problem" 键）。
        trajectories: 按标签索引的轨迹字典。
    """

    problem: str = ""
    trajectories: dict[str, TrajectoryItem] = field(default_factory=dict)

    # --- dict-compatible 接口（便于算子渐进迁移）---

    def get(self, key: str, default: Any = None) -> Any:
        """dict 风格的 get：先检查 'problem'，然后在 trajectories 中查找。"""
        if key == "problem":
            return self.problem
        traj = self.trajectories.get(key)
        if traj is not None:
            return traj.to_dict()
        return default

    def items(self):
        """dict 风格的 items：返回 (key, value) 对，包括 'problem'。"""
        yield "problem", self.problem
        for k, v in self.trajectories.items():
            yield k, v.to_dict()

    def __getitem__(self, key: str) -> Any:
        if key == "problem":
            return self.problem
        traj = self.trajectories.get(key)
        if traj is not None:
            return traj.to_dict()
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key == "problem" or key in self.trajectories

    def __bool__(self) -> bool:
        return bool(self.problem) or bool(self.trajectories)

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> "InstanceTrajectories":
        """从轨迹池的原始 JSON dict 构建。"""
        if not isinstance(d, dict):
            return InstanceTrajectories()
        problem = str(d.get("problem") or "")
        trajectories: dict[str, TrajectoryItem] = {}
        for k, v in d.items():
            if k == "problem" or not isinstance(v, dict):
                continue
            trajectories[k] = TrajectoryItem.from_dict(v)
        return InstanceTrajectories(problem=problem, trajectories=trajectories)

    def to_dict(self) -> dict[str, Any]:
        """转回原始 dict 格式。"""
        out: dict[str, Any] = {}
        if self.problem:
            out["problem"] = self.problem
        for k, traj in self.trajectories.items():
            out[k] = traj.to_dict()
        return out


@dataclass
class OperatorContext:
    """算子执行的共享上下文。

    封装算子所需的模型配置、提示词配置和选择模式，
    替代原先通过 dict 传递的 operator_config。

    Attributes:
        model_config: LLM 模型配置（保留 dict，因为需透传给 LLMClient）。
        prompt_config: 提示词配置。
        selection_mode: 默认轨迹选择模式（"weighted" 或 "random"）。
    """

    model_config: dict[str, Any] = field(default_factory=dict)
    prompt_config: dict[str, Any] = field(default_factory=dict)
    selection_mode: str = "weighted"


@dataclass
class OperatorResult:
    """单实例算子执行结果

    这是 Operator 返回给 perf_run.py 的标准化结果对象。
    包含用于构建 AgentRequest 的全部信息。

    Attributes:
        additional_requirements: 额外的 prompt 要求（来自算子分析）
        initial_solution: 可选的初始解覆盖
        source_labels: 使用的源轨迹标签列表
    """

    additional_requirements: str | None = None
    initial_solution: str | None = None
    source_labels: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 基类
# ---------------------------------------------------------------------------


class BaseOperator(abc.ABC):
    """
    SE算子基类，定义通用功能和新的 `run` 接口。
    所有算子都应继承自此类。
    """

    def __init__(self, context: OperatorContext):
        """
        初始化算子。

        Args:
            context: OperatorContext 实例。
        """
        self.context = context
        self.llm_client: LLMClient | None = None
        self.logger = get_se_logger(f"operator.{self.get_name()}", emoji="🔧")

    def _setup_model(self) -> None:
        """设置LLM客户端实例。"""
        if self.llm_client is not None:
            return
        model_config_data = self.context.model_config
        self.llm_client = LLMClient(model_config_data)
        self.logger.info(f"LLM客户端已初始化: {model_config_data.get('name')}")

    def _call_llm_api(self, prompt: str, system_prompt: str = "") -> str:
        """
        调用LLM API。

        Args:
            prompt: 用户提示。
            system_prompt: 系统提示。

        Returns:
            LLM生成的响应文本。
        """
        self._setup_model()
        history = []
        if system_prompt:
            history.append({"role": "system", "content": system_prompt})
        history.append({"role": "user", "content": prompt})

        try:
            model_cfg = self.context.model_config
            temp = model_cfg.get("temperature", 0.3)
            max_out = model_cfg.get("max_output_tokens")
            self.logger.debug(f"LLM系统提示词:\n{system_prompt}")
            self.logger.debug(f"LLM用户提示词:\n{prompt}")
            message = self.llm_client.call_llm(
                history,
                temperature=temp,
                max_tokens=max_out,
                usage_context=f"operator.{self.get_name()}",
            )
            self.logger.debug(f"LLM原始响应:\n{message}")
            if message:
                message = self.llm_client.clean_think_tags(message)
            self.logger.debug(f"LLM清理后响应:\n{message}")
            return message or ""
        except Exception as e:
            self.logger.error(f"LLM API调用失败: {e}")
            return ""

    def _extract_code_block_py(self, text: str) -> str | None:
        """从LLM输出中提取 ```py ... ``` 代码块内容。"""
        if not isinstance(text, str) or not text:
            return None
        pattern = re.compile(r"```(?:py|python)\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
        m = pattern.search(text)
        if m:
            return m.group(1).strip() or None
        return None

    def _extract_code_text(self, text: str) -> str | None:
        """优先提取代码块，否则返回原始文本并尝试清理。"""
        if not isinstance(text, str) or not text.strip():
            return None
        block = self._extract_code_block_py(text)
        if isinstance(block, str) and block.strip():
            return block.strip()

        raw_code = text.strip()
        if raw_code.startswith("```") and raw_code.endswith("```"):
            try:
                raw_code = re.sub(r"^```(?:py|python)?\s*\n?", "", raw_code, flags=re.IGNORECASE)
                raw_code = re.sub(r"\n?```$", "", raw_code)
            except Exception:
                pass
        return raw_code.strip() or None

    def _require_py_block_with_retry(
        self,
        build_prompt_fn,
        max_retries: int = 2,
        temperature_override: float | None = None,
    ) -> str | None:
        """要求LLM以```py代码块```输出，若未满足则重试。"""
        self._setup_model()
        model_cfg = self.context.model_config
        base_temp = model_cfg.get("temperature", 0.3)
        temp_to_use = base_temp if temperature_override is None else temperature_override

        for attempt in range(max_retries + 1):
            try:
                prompt, system_prompt = build_prompt_fn(attempt)
                pcfg = self.context.prompt_config or {}
                common = pcfg.get("base_operator", {}) if isinstance(pcfg.get("base_operator"), dict) else {}
                enforce_tail = common.get("enforce_tail")
                if enforce_tail is None:
                    enforce_tail = pcfg.get("operator_enforce_tail")
                import_blocks = common.get("imports_block")
                if import_blocks is None:
                    import_blocks = pcfg.get("operator_imports_block")
                optimization_target = common.get("optimization_target")
                if optimization_target is None:
                    optimization_target = pcfg.get("operator_optimization_target")
                system_prompt_use = system_prompt or ""
                if isinstance(enforce_tail, str) and enforce_tail.strip():
                    system_prompt_use += enforce_tail
                if isinstance(import_blocks, str) and import_blocks.strip():
                    system_prompt_use += import_blocks
                if isinstance(optimization_target, str) and optimization_target.strip():
                    system_prompt_use += optimization_target

                history = [{"role": "system", "content": system_prompt_use}, {"role": "user", "content": prompt}]
                max_out = model_cfg.get("max_output_tokens")
                enable_thinking = None if attempt == 0 else False

                self.logger.info(f"第{attempt + 1}次尝试，温度={temp_to_use}")
                self.logger.debug(f"LLM系统提示词(重试第{attempt + 1}次)")
                self.logger.debug(f"LLM用户提示词(重试第{attempt + 1}次)")

                message = self.llm_client.call_llm(
                    history,
                    temperature=temp_to_use,
                    max_tokens=max_out,
                    enable_thinking=enable_thinking,
                    usage_context=f"operator.{self.get_name()}",
                )
                self.logger.debug(f"LLM原始响应(重试第{attempt + 1}次):\n{message}")
                if message:
                    message = self.llm_client.clean_think_tags(message)
                # self.logger.debug(f"LLM清理后响应(重试第{attempt + 1}次):\n{message}")

                code = self._extract_code_block_py(message or "")
                if code:
                    return code

                self.logger.warning("未检测到```py代码块，进行重试")
            except Exception as e:
                self.logger.error(f"格式化代码块生成失败: {e}")
                continue
        return None

    def _format_entry(self, approaches_data: dict[str, Any]) -> str:
        return TrajPoolManager.format_entry(approaches_data)

    def _weighted_select_labels(
        self, entry: dict[str, Any], k: int = 1, allowed_labels: list[str] | None = None
    ) -> list[str]:
        """基于 performance 的线性加权采样选择子标签，performance 越低权重越高。
        若提供 allowed_labels，则仅在该集合中进行采样（忽略不存在的标签）。
        """
        if not isinstance(entry, dict):
            return []
        items: list[tuple[str, float]] = []
        for subkey, subval in entry.items():
            if subkey == "problem" or not isinstance(subval, dict):
                continue
            if allowed_labels is not None:
                lab = str(subkey)
                lab2 = str(subval.get("label")) if isinstance(subval.get("label"), str) else None
                if lab not in allowed_labels and (lab2 is None or lab2 not in allowed_labels):
                    continue
            perf = subval.get("metric")
            try:
                perf_val = float(perf) if perf is not None else 1.0
            except Exception:
                perf_val = 1.0
            items.append((str(subkey), perf_val))
        if not items:
            return []
        eps = 1e-9
        selected: list[str] = []
        remaining = items.copy()
        for _ in range(min(k, len(remaining))):
            weights = [max(0.001, 1.0 / max(eps, perf)) for _, perf in remaining]
            total = sum(weights)
            if total <= 0:
                choice = random.choice(remaining)[0]
            else:
                weights = [w / total for w in weights]
                r = random.random()
                s = 0.0
                choice = remaining[-1][0]
                for (label_key, perf), w in zip(remaining, weights):
                    s += w
                    if r <= s:
                        choice = label_key
                        break
            selected.append(choice)
            remaining = [it for it in remaining if it[0] != choice]
        return selected

    def _random_select_labels(
        self, entry: dict[str, Any], k: int = 1, allowed_labels: list[str] | None = None
    ) -> list[str]:
        if not isinstance(entry, dict):
            return []
        candidates: list[str] = []
        for subkey, subval in entry.items():
            if subkey == "problem" or not isinstance(subval, dict):
                continue
            if allowed_labels is not None:
                lab = str(subkey)
                lab2 = str(subval.get("label")) if isinstance(subval.get("label"), str) else None
                if lab not in allowed_labels and (lab2 is None or lab2 not in allowed_labels):
                    continue
            candidates.append(str(subkey))
        if not candidates:
            return []
        k = min(k, len(candidates))
        try:
            return random.sample(candidates, k)
        except Exception:
            out: list[str] = []
            pool = candidates.copy()
            for _ in range(k):
                choice = random.choice(pool)
                out.append(choice)
                pool = [c for c in pool if c != choice]
                if not pool:
                    break
            return out

    def _get_selection_mode(self, step_config: StepConfig) -> str:
        try:
            v = step_config.selection_mode
            if isinstance(v, str) and v.strip():
                m = v.strip().lower()
                if m in ("weighted", "random"):
                    return m
            g = self.context.selection_mode
            if isinstance(g, str) and g.strip():
                m = g.strip().lower()
                if m in ("weighted", "random"):
                    return m
        except Exception:
            pass
        return "weighted"

    def _resolve_label_subkey(self, entry: dict[str, Any], label: str) -> str | None:
        """将外部提供的标签解析为 entry 的子键。
        优先匹配子键名，其次匹配子项内部的 `label` 字段。
        """
        if not isinstance(entry, dict):
            return None
        lab = str(label)
        if lab in entry and isinstance(entry.get(lab), dict):
            return lab
        for subkey, subval in entry.items():
            if subkey == "problem" or not isinstance(subval, dict):
                continue
            if str(subval.get("label")) == lab:
                return str(subkey)
        return None

    def _select_source_labels(self, entry: dict[str, Any], step_config: StepConfig, required_n: int) -> list[str]:
        """统一选择源轨迹标签。
        规则：
        - 若 `inputs` 标签数目 == required_n：直接使用 `inputs`
        - 若 `inputs` 标签数目 >  required_n：在 `inputs` 范围内加权采样 required_n 个
        - 若 `inputs` 标签数目 <  required_n：先使用已有 `inputs`，剩余从整个 entry 中加权采样补齐
        返回 entry 子键名列表，唯一且最多 required_n 个。
        """
        if not isinstance(entry, dict):
            return []
        inputs = step_config.inputs or []
        provided_labels = [str(i.get("label")) for i in inputs if isinstance(i, dict) and i.get("label")]
        # 解析为 entry 子键
        resolved = []
        seen = set()
        for lab in provided_labels:
            subkey = self._resolve_label_subkey(entry, lab)
            if subkey and subkey not in seen:
                resolved.append(subkey)
                seen.add(subkey)

        need = max(0, int(required_n))
        count = len(resolved)
        if count == need:
            return resolved
        if count > need:
            mode = self._get_selection_mode(step_config)
            if mode == "random":
                sampled = self._random_select_labels(entry, k=need, allowed_labels=resolved)
            else:
                sampled = self._weighted_select_labels(entry, k=need, allowed_labels=resolved)
            # 去重并返回
            out = []
            used = set()
            for s in sampled:
                if s not in used:
                    out.append(s)
                    used.add(s)
            return out

        # count < need：先用已有，再补齐
        out = list(resolved)
        used = set(out)
        # 构建候选集合（排除已选）
        all_subkeys = [str(k) for k, v in entry.items() if k != "problem" and isinstance(v, dict)]
        remaining = [k for k in all_subkeys if k not in used]
        if remaining:
            mode = self._get_selection_mode(step_config)
            if mode == "random":
                sampled_more = self._random_select_labels(entry, k=need - count, allowed_labels=remaining)
            else:
                sampled_more = self._weighted_select_labels(entry, k=need - count, allowed_labels=remaining)
            for s in sampled_more:
                if s not in used:
                    out.append(s)
                    used.add(s)
                if len(out) >= need:
                    break
        return out[:need]

    @abc.abstractmethod
    def get_name(self) -> str:
        """获取算子名称。"""
        pass

    @abc.abstractmethod
    def run_for_instance(
        self,
        step_config: StepConfig,
        instance_name: str,
        instance_entry: InstanceTrajectories | dict[str, Any],
        *,
        problem_description: str = "",
    ) -> OperatorResult:
        """处理单个实例，返回结构化结果。

        这是单实例模式下的标准调用接口。子类必须实现此方法。

        Args:
            step_config: 当前步骤的配置（StepConfig 对象）。
            instance_name: 实例名称。
            instance_entry: 该实例在轨迹池中的数据（InstanceTrajectories 或兼容 dict）。
            problem_description: 问题描述文本（由调用方显式传入）。

        Returns:
            OperatorResult 对象，包含 additional_requirements、initial_solution 等。
        """
        ...


class TemplateOperator(BaseOperator):
    """
    模板算子基类，用于为下一次 PerfAgent 运行生成初始代码。
    """


class EnhanceOperator(BaseOperator):
    """
    增强算子基类，用于为下一次 PerfAgent 运行生成增强历史配置。
    """
