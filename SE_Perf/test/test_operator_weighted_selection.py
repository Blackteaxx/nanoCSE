#!/usr/bin/env python3
"""测试 BaseOperator._weighted_select_labels 的加权选择逻辑。

覆盖场景：
- metric_higher_is_better=False（默认，越小越好）
- metric_higher_is_better=True（越大越好，如 LiveCodeBench）
- _select_source_labels 的 inputs 直通 / 补齐 / 截选
"""

import random
import sys
from pathlib import Path

import pytest

# 添加SE目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from operators.base import (
    BaseOperator,
    InstanceTrajectories,
    OperatorContext,
    OperatorResult,
    TrajectoryItem,
)
from perf_config import StepConfig

# ---------------------------------------------------------------------------
# DummyOperator：满足抽象方法要求的最小实现
# ---------------------------------------------------------------------------


class DummyOperator(BaseOperator):
    def get_name(self) -> str:
        return "dummy"

    def run_for_instance(self, step_config, instance_name, instance_entry, *, problem_description=""):
        return OperatorResult()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_operator(higher_is_better: bool = False) -> DummyOperator:
    ctx = OperatorContext(
        model_config={},
        prompt_config={},
        selection_mode="weighted",
        metric_higher_is_better=higher_is_better,
    )
    return DummyOperator(ctx)


def _make_entry() -> InstanceTrajectories:
    """构造含 4 条轨迹的 entry，metric 分布差异明显。"""
    return InstanceTrajectories(
        problem="p",
        trajectories={
            "A": TrajectoryItem(label="A", metric=1.0, solution=""),
            "B": TrajectoryItem(label="B", metric=0.5, solution=""),
            "C": TrajectoryItem(label="C", metric=10.0, solution=""),
            "INF": TrajectoryItem(label="INF", metric=float("inf"), solution=""),
        },
    )


@pytest.fixture
def entry():
    return _make_entry()


# ---------------------------------------------------------------------------
# metric_higher_is_better = False（默认，越小越好）
# ---------------------------------------------------------------------------


class TestWeightedLowerIsBetter:
    """metric 越小越好时，metric 更低的轨迹应获得更高的选择概率。"""

    @pytest.fixture(autouse=True)
    def setup(self, entry):
        random.seed(42)
        self.op = _make_operator(higher_is_better=False)
        self.entry = entry

    def test_lower_metric_selected_more(self):
        """B(0.5) 应该比 A(1.0) 被选中更多，A 比 C(10.0) 多。"""
        counts = {"A": 0, "B": 0, "C": 0}
        for _ in range(5000):
            lbls = self.op._weighted_select_labels(self.entry, k=1, allowed_labels=["A", "B", "C"])
            counts[lbls[0]] += 1
        assert counts["B"] > counts["A"] > counts["C"], f"期望 B > A > C，实际 {counts}"

    def test_allowed_labels_filter(self):
        """只在 allowed_labels 范围内采样。"""
        counts = {"A": 0, "B": 0}
        for _ in range(3000):
            lbls = self.op._weighted_select_labels(self.entry, k=1, allowed_labels=["A", "B"])
            counts[lbls[0]] += 1
        # B(0.5) 权重 = 1/0.5 = 2.0，A(1.0) 权重 = 1/1.0 = 1.0 => B 约占 2/3
        assert counts["B"] > counts["A"]

    def test_select_source_labels_exact_inputs(self):
        """inputs 数目 == required_n 时直接使用。"""
        sc = StepConfig(inputs=[{"label": "B"}])
        chosen = self.op._select_source_labels(self.entry, sc, required_n=1)
        assert chosen == ["B"]

    def test_select_source_labels_fill_prefers_low_metric(self):
        """inputs 不足时补齐，低 metric 的 B 更常被选（与 higher_is_better 对称测试）。"""
        # 使用不含 INF 的 entry，保持与 higher_is_better 测试对称
        entry_no_inf = InstanceTrajectories(
            problem="p",
            trajectories={
                "A": TrajectoryItem(label="A", metric=1.0, solution=""),
                "B": TrajectoryItem(label="B", metric=0.5, solution=""),
                "C": TrajectoryItem(label="C", metric=10.0, solution=""),
            },
        )
        sc = StepConfig(inputs=[{"label": "A"}])
        counts = {}
        for _ in range(2000):
            chosen = tuple(self.op._select_source_labels(entry_no_inf, sc, required_n=2))
            counts[chosen] = counts.get(chosen, 0) + 1
        # B(0.5) 权重 = 1/0.5 = 2.0，C(10.0) 权重 = 1/10 = 0.1 => B 应远多于 C
        b_count = sum(v for k, v in counts.items() if "B" in k)
        c_count = sum(v for k, v in counts.items() if "C" in k)
        assert b_count > c_count, f"期望 B 补齐多于 C，B={b_count}, C={c_count}"


# ---------------------------------------------------------------------------
# metric_higher_is_better = True（越大越好，如 LiveCodeBench）
# ---------------------------------------------------------------------------


class TestWeightedHigherIsBetter:
    """metric 越大越好时，metric 更高的轨迹应获得更高的选择概率。"""

    @pytest.fixture(autouse=True)
    def setup(self, entry):
        random.seed(42)
        self.op = _make_operator(higher_is_better=True)
        self.entry = entry

    def test_higher_metric_selected_more(self):
        """C(10.0) 应该比 A(1.0) 被选中更多，A 比 B(0.5) 多。"""
        counts = {"A": 0, "B": 0, "C": 0}
        for _ in range(5000):
            lbls = self.op._weighted_select_labels(self.entry, k=1, allowed_labels=["A", "B", "C"])
            counts[lbls[0]] += 1
        assert counts["C"] > counts["A"] > counts["B"], f"期望 C > A > B，实际 {counts}"

    def test_allowed_labels_filter(self):
        """只在 allowed_labels 范围内采样，高 metric 的 A 应比低 metric 的 B 多。"""
        counts = {"A": 0, "B": 0}
        for _ in range(3000):
            lbls = self.op._weighted_select_labels(self.entry, k=1, allowed_labels=["A", "B"])
            counts[lbls[0]] += 1
        # A(1.0) 权重 = 1.0，B(0.5) 权重 = 0.5 => A 约占 2/3
        assert counts["A"] > counts["B"], f"期望 A > B，实际 {counts}"

    def test_select_source_labels_fill_prefers_high_metric(self):
        """inputs 不足时补齐，高 metric 的 C 更常被选。"""
        # 使用不含 INF 的 entry，否则 inf 权重会垄断所有选择
        entry_no_inf = InstanceTrajectories(
            problem="p",
            trajectories={
                "A": TrajectoryItem(label="A", metric=1.0, solution=""),
                "B": TrajectoryItem(label="B", metric=0.5, solution=""),
                "C": TrajectoryItem(label="C", metric=10.0, solution=""),
            },
        )
        sc = StepConfig(inputs=[{"label": "A"}])
        counts = {}
        for _ in range(2000):
            chosen = tuple(self.op._select_source_labels(entry_no_inf, sc, required_n=2))
            counts[chosen] = counts.get(chosen, 0) + 1
        # C(10.0) 作为补齐项应最频繁
        c_count = sum(v for k, v in counts.items() if "C" in k)
        b_count = sum(v for k, v in counts.items() if "B" in k)
        assert c_count > b_count, f"期望 C 补齐多于 B，C={c_count}, B={b_count}"

    def test_default_metric_is_worst(self):
        """metric 为 None 时，在 higher_is_better=True 下默认为 0.0（最差），几乎不被选。"""
        entry_with_none = InstanceTrajectories(
            problem="p",
            trajectories={
                "good": TrajectoryItem(label="good", metric=1.0, solution=""),
                "none_metric": TrajectoryItem(label="none_metric", metric=None, solution=""),
            },
        )
        counts = {"good": 0, "none_metric": 0}
        for _ in range(3000):
            lbls = self.op._weighted_select_labels(entry_with_none, k=1)
            counts[lbls[0]] += 1
        assert counts["good"] > counts["none_metric"] * 5, f"None metric 应很少被选: {counts}"
