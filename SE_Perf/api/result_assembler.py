"""
Result assembler — reads ``run_single_instance`` output artefacts and
builds an :class:`EvolveResponse`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .schemas import (
    CandidateInfo,
    EvolveResponse,
    TokenUsage,
    TrajectorySummary,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_preds(output_dir: Path) -> list[CandidateInfo]:
    """Parse the aggregated ``preds.json`` into a list of candidates."""
    preds_path = output_dir / "preds.json"
    if not preds_path.exists():
        return []
    try:
        with open(preds_path, encoding="utf-8") as f:
            data: dict = json.load(f)
    except Exception:
        return []

    candidates: list[CandidateInfo] = []
    for _instance_id, entries in data.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            candidates.append(
                CandidateInfo(
                    solution=entry.get("solution", ""),
                    metric=entry.get("metric"),
                    iteration=entry.get("iteration", 0),
                    success=entry.get("success", False),
                    artifacts=entry.get("artifacts") or {},
                )
            )
    return candidates


def _pick_best(
    candidates: list[CandidateInfo],
    *,
    higher_is_better: bool = False,
) -> CandidateInfo | None:
    """Select the best candidate by metric."""
    if not candidates:
        return None

    def _metric_key(c: CandidateInfo) -> float:
        try:
            v = float(c.metric) if c.metric is not None else None
        except (ValueError, TypeError):
            v = None
        if v is None or not math.isfinite(v):
            return -float("inf") if higher_is_better else float("inf")
        return v

    return (max if higher_is_better else min)(candidates, key=_metric_key)


def _read_token_usage(output_dir: Path) -> TokenUsage:
    """Sum up ``token_usage.jsonl``."""
    usage_path = output_dir / "token_usage.jsonl"
    if not usage_path.exists():
        return TokenUsage()
    prompt = completion = total = 0
    try:
        with open(usage_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    pt = int(rec.get("prompt_tokens") or 0)
                    ct = int(rec.get("completion_tokens") or 0)
                    tt = int(rec.get("total_tokens") or (pt + ct))
                    prompt += pt
                    completion += ct
                    total += tt
                except Exception:
                    continue
    except Exception:
        pass
    return TokenUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


def _read_traj_pool_summary(output_dir: Path) -> TrajectorySummary | None:
    """Build a lightweight trajectory-pool summary from ``traj.pool``."""
    pool_path = output_dir / "traj.pool"
    if not pool_path.exists():
        return None
    try:
        with open(pool_path, encoding="utf-8") as f:
            pool: dict = json.load(f)
    except Exception:
        return None

    reserved = {"problem"}
    labels = [k for k in pool if k not in reserved]
    if not labels:
        return TrajectorySummary(total_trajectories=0)

    best_label: str | None = None
    best_metric: float | None = None
    for label in labels:
        entry = pool.get(label)
        if not isinstance(entry, dict):
            continue
        try:
            m = entry.get("metric")
            if m is None:
                continue
            mf = float(m)
            if not math.isfinite(mf):
                continue
            if best_metric is None or mf < best_metric:
                best_metric = mf
                best_label = label
        except (ValueError, TypeError):
            continue

    return TrajectorySummary(
        total_trajectories=len(labels),
        best_label=best_label,
        labels=labels,
    )


def _enrich_candidates_from_pool(
    candidates: list[CandidateInfo],
    output_dir: Path,
) -> None:
    """Attach trajectory labels to candidates by matching solution text
    against the ``traj.pool`` entries (best-effort)."""
    pool_path = output_dir / "traj.pool"
    if not pool_path.exists():
        return
    try:
        with open(pool_path, encoding="utf-8") as f:
            pool: dict = json.load(f)
    except Exception:
        return

    solution_to_label: dict[str, str] = {}
    reserved = {"problem"}
    for key, entry in pool.items():
        if key in reserved or not isinstance(entry, dict):
            continue
        sol = entry.get("solution", "")
        if sol:
            solution_to_label[sol] = entry.get("label", key)

    for c in candidates:
        if c.label is None and c.solution in solution_to_label:
            c.label = solution_to_label[c.solution]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assemble_response(
    instance_id: str,
    status: str,
    output_dir: str,
    *,
    error: str | None = None,
    higher_is_better: bool = False,
    return_summary: bool = True,
) -> EvolveResponse:
    """Build an :class:`EvolveResponse` from on-disk artefacts.

    This is the single entry-point called by the evolve server after
    ``run_single_instance`` finishes.
    """
    out_path = Path(output_dir)
    candidates = _read_preds(out_path)
    _enrich_candidates_from_pool(candidates, out_path)
    best = _pick_best(candidates, higher_is_better=higher_is_better)
    token_usage = _read_token_usage(out_path)
    traj_summary = _read_traj_pool_summary(out_path) if return_summary else None

    return EvolveResponse(
        instance_id=instance_id,
        status=status,
        best_candidate=best,
        all_candidates=candidates,
        token_usage=token_usage,
        raw_output_dir=output_dir,
        trajectory_summary=traj_summary,
        error=error,
    )
