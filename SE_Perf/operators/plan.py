#!/usr/bin/env python3
"""
Plan Operator (LLM-based)

为单实例使用LLM规划 K 种不同的实现方案（策略），返回 K 个 OperatorResult，
每个对应一次独立的迭代。

特性：
- 严格的 JSON 输出格式约束与校验，失败时重试；不足 K 条时使用回退策略补齐
"""

import json
import re
import textwrap

from perf_config import StepConfig

from operators.base import InstanceTrajectories, OperatorResult, TemplateOperator

from .registry import register_operator


class PlanOperator(TemplateOperator):
    """LLM方案规划算子：为单实例生成 K 条多样化策略。

    run_for_instance 返回 list[OperatorResult]，每个元素对应一个 plan（一次独立迭代）。
    """

    def get_name(self) -> str:
        return "plan"

    def run_for_instance(
        self,
        step_config: StepConfig,
        instance_name: str,
        instance_entry: InstanceTrajectories,
        *,
        problem_description: str = "",
    ) -> list[OperatorResult]:
        """为单实例生成 K 个 plan，返回 OperatorResult 列表。

        每个 OperatorResult 的 additional_requirements 包含一个策略文本。
        """
        model_cfg = self.context.model_config or {}
        api_key = str(model_cfg.get("api_key", "")).strip()
        name_ok = bool(str(model_cfg.get("name", "")).strip())
        base_ok = bool(str(model_cfg.get("api_base", "")).strip())
        llm_enabled = name_ok and base_ok and api_key not in ("", "empty", "sk-PLACEHOLDER")

        num = step_config.num
        try:
            num = int(num) if num is not None else 1
        except Exception:
            num = 1

        # 使用显式传入的 problem_description
        problem_text = problem_description.strip() if problem_description else ""

        # 生成策略
        if problem_text and llm_enabled:
            strategies = self._llm_strategies_with_retry(problem_text, num)
        else:
            strategies = []

        # 不足时使用回退补齐
        if len(strategies) < num:
            for i in range(len(strategies) + 1, num + 1):
                strategies.append(self._fallback(i))

        # 包装为 OperatorResult 列表
        results: list[OperatorResult] = []
        for strategy_text in strategies[:num]:
            content = self._build_additional_requirements(strategy_text)
            results.append(OperatorResult(additional_requirements=content))

        return results

    # --- Internal helpers ---

    def _build_prompts(self, problem_text: str, k: int) -> tuple[str, str]:
        """构建提示词，强调算法多样性、复杂度分析和严格的 JSON 格式。"""
        pcfg = self.context.prompt_config or {}
        plan_cfg = pcfg.get("plan", {}) if isinstance(pcfg, dict) else {}
        sys_prompt = (
            plan_cfg.get("system_prompt")
            or """You are a world-class Algorithm Engineer and Competitive Programmer. Your task is to design EXACTLY K distinct, high-performance algorithmic strategies for a given problem.

Guidelines:
1. **Diversity**: The strategies MUST differ in algorithmic paradigms (e.g., Dynamic Programming, Greedy, BFS/DFS, Two Pointers, Sliding Window, Bit Manipulation) or Data Structures.
2. **Performance**: Prioritize optimal Time and Space Complexity. Avoid naive brute-force unless unavoidable.
3. **Content**: Each strategy description must be a concise English paragraph including: 
- The core logic/heuristic.
- Key data structures.
- Expected Time Complexity (Big-O) and Space Complexity.
4. **Format**: Return the JSON object wrapped in a Markdown code block with the language tag 'json'.
- Structure:
    ```json
    {"strategies": [string, string, ...]}
    ```
- The array must contain exactly K strings.
- NO conversational text outside the code block."""
        )

        up_template = plan_cfg.get("user_prompt_template") or (
            """
Instruction: Generate {k} diverse high-performance strategies.

Problem Description:
{problem_text}

Required Count: {k}
            """
        )
        user_prompt = up_template.format(k=k, problem_text=problem_text)
        return sys_prompt, user_prompt

    def _extract_json_fragment(self, text: str) -> str:
        """尽可能提取文本中的 JSON 片段。"""
        if not isinstance(text, str):
            return ""
        t = text.strip()
        fence = re.search(r"```json\s*(.*?)```", t, re.DOTALL | re.IGNORECASE)
        if fence:
            t = fence.group(1).strip()
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            return t[start : end + 1]
        return t

    def _parse_strategies(self, message: str, k: int) -> list[str] | None:
        """解析并校验 JSON，确保数组长度为 k。"""
        frag = self._extract_json_fragment(message)
        try:
            data = json.loads(frag)
        except Exception:
            return None
        arr = data.get("strategies") if isinstance(data, dict) else None
        if not isinstance(arr, list):
            return None
        vals = [str(x).strip() for x in arr if isinstance(x, (str, int, float))]
        return vals if len(vals) >= k and all(v for v in vals) else None

    def _llm_strategies_with_retry(self, problem_text: str, k: int, max_attempts: int = 3) -> list[str]:
        """调用LLM并进行格式校验，失败则重试。"""
        sys_prompt, user_prompt = self._build_prompts(problem_text, k)
        for attempt in range(1, max_attempts + 1):
            try:
                msg = self._call_llm_api(prompt=user_prompt, system_prompt=sys_prompt)
                vals = self._parse_strategies(msg, k)
                if vals:
                    return vals
                sys_prompt = sys_prompt + " Strictly output valid JSON now. Do not include commentary or code fences."
            except Exception:
                pass
        return []

    def _fallback(self, idx: int) -> str:
        plan_cfg = (self.context.prompt_config or {}).get("plan", {})
        patterns = plan_cfg.get(
            "fallback_patterns",
            [
                "Prefer DP/graph over greedy; restructure loops with memoization.",
                "Use alternative data structures (heap/deque/ordered-set) to avoid linear scans.",
                "Improve I/O throughput; batch parsing and reduce conversions.",
                "Precompute invariants and cache expensive calls to eliminate repeated work.",
                "Adopt divide-and-conquer or search; structure recursion/iteration for clarity and speed.",
            ],
        )
        try:
            body = str(patterns[(idx - 1) % len(patterns)])
        except Exception:
            body = "Diversify algorithmic approach; improve core performance pragmatically."
        header = f"DIVERSIFIED STRATEGY {idx}"
        return f"{header}\n{body}"

    def _build_additional_requirements(self, strategy_text: str) -> str:
        """包装为 additional_requirements 文本。"""
        st = textwrap.indent((strategy_text or "").strip(), "  ")
        plan_cfg = (self.context.prompt_config or {}).get("plan", {})
        header = plan_cfg.get("strategy_header") or (
            """
### STRATEGY MODE: PLAN STRATEGY
You must strictly follow and implement the outlined approach below.
            """
        )
        parts = []
        if isinstance(header, str) and header.strip():
            parts.append(header.strip())
        parts.append("\n" + st)
        return "\n".join(parts)


register_operator("plan", PlanOperator)
