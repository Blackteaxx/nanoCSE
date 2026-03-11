"""
nanoCSE Evolve HTTP Service
===========================

Wraps ``perf_run.run_single_instance()`` as a ``POST /v1/evolve`` endpoint.

Usage::

    # Start from nanoCSE/SE_Perf directory:
    python -m api.evolve_server --host 0.0.0.0 --port 8800 --workers 64

    # Or with uvicorn directly:
    cd nanoCSE/SE_Perf && uvicorn api.evolve_server:app --host 0.0.0.0 --port 8800

Environment variables:

    NANOCSE_OUTPUT_ROOT  — root directory for per-request output dirs
                           (default: ``/tmp/nanocse_evolve_outputs``)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException

# Ensure SE_Perf root is importable
_SE_PERF_ROOT = Path(__file__).resolve().parent.parent
if str(_SE_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(_SE_PERF_ROOT))

from .result_assembler import assemble_response
from .schemas import EvolveRequest, EvolveResponse

# ---------------------------------------------------------------------------
# Thin wrapper for lazy import — patchable by tests
# ---------------------------------------------------------------------------


def _invoke_run_single_instance(
    config_path: str,
    instance_path: str,
    output_dir: str,
    mode: str,
    se_cfg: Any,
) -> dict:
    from perf_run import run_single_instance

    return run_single_instance(
        config_path=config_path,
        instance_path=instance_path,
        output_dir=output_dir,
        mode=mode,
        se_cfg=se_cfg,
    )


# ---------------------------------------------------------------------------
# Config assembly helpers
# ---------------------------------------------------------------------------

_OUTPUT_ROOT = os.environ.get(
    "NANOCSE_OUTPUT_ROOT",
    str(_SE_PERF_ROOT.parent / "trajectories_evolve_server"),
)


def _make_output_dir(instance_id: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    dirname = f"{instance_id}_{ts}_{uid}"
    out = os.path.join(_OUTPUT_ROOT, dirname)
    Path(out).mkdir(parents=True, exist_ok=True)
    return out


def _build_se_config_dict(req: EvolveRequest) -> dict[str, Any]:
    """Assemble a raw dict consumable by ``SEPerfRunSEConfig.from_dict``."""
    sc = req.search_config

    # Base: load from YAML if a config_path is given
    base: dict[str, Any] = {}
    if sc.config_path and Path(sc.config_path).is_file():
        with open(sc.config_path, encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}

    base["task_type"] = req.task_type
    base["metric_higher_is_better"] = sc.metric_higher_is_better
    base["max_iterations"] = sc.max_iterations

    # LLM model section — override from request
    lc = req.llm_config
    model: dict[str, Any] = base.get("model") or {}
    if lc.name is not None:
        model["name"] = lc.name
    if lc.api_base is not None:
        model["api_base"] = lc.api_base
    if lc.api_key is not None:
        model["api_key"] = lc.api_key
    if lc.max_output_tokens is not None:
        model["max_output_tokens"] = lc.max_output_tokens
    if lc.temperature is not None:
        model["temperature"] = lc.temperature
    if lc.request_timeout is not None:
        model.setdefault("extras", {})["request_timeout"] = lc.request_timeout
    if req.sampling_params:
        model.setdefault("extras", {}).update(req.sampling_params)
    base["model"] = model

    # Strategy override
    if sc.strategy is not None:
        base["strategy"] = sc.strategy
    if sc.local_memory is not None:
        base["local_memory"] = sc.local_memory
    if sc.global_memory_bank is not None:
        base["global_memory_bank"] = sc.global_memory_bank
    if sc.prompt_config is not None:
        base["prompt_config"] = sc.prompt_config
    if sc.base_config is not None:
        base["base_config"] = sc.base_config

    return base


def _write_instance_payload(payload: dict, instance_id: str, tmpdir: str) -> str:
    """Write inline instance_payload to a temp JSON file and return its path."""
    filename = f"{instance_id}.json"
    path = os.path.join(tmpdir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    Path(_OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="nanoCSE Evolve Service",
    version="0.1.0",
    lifespan=_lifespan,
)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/v1/evolve", response_model=EvolveResponse)
def evolve(req: EvolveRequest) -> EvolveResponse:
    """Run a single evolutionary search instance and return the results."""

    # Resolve instance file path
    instance_path: str | None = None
    tmp_dir: str | None = None

    if req.instance_payload is not None:
        tmp_dir = tempfile.mkdtemp(prefix="nanocse_inst_")
        instance_path = _write_instance_payload(req.instance_payload, req.instance_id, tmp_dir)
    elif req.task_data_path is not None:
        instance_path = req.task_data_path
    else:
        raise HTTPException(
            status_code=422,
            detail="Either instance_payload or task_data_path must be provided.",
        )

    if not Path(instance_path).is_file():
        raise HTTPException(
            status_code=422,
            detail=f"Instance file not found: {instance_path}",
        )

    # Build SE config
    se_config_dict = _build_se_config_dict(req)

    # Prepare output directory
    output_dir = _make_output_dir(req.instance_id)
    se_config_dict["output_dir"] = output_dir

    from perf_config import SEPerfRunSEConfig

    se_cfg = SEPerfRunSEConfig.from_dict(se_config_dict)

    result = _invoke_run_single_instance(
        config_path="__unused__",
        instance_path=instance_path,
        output_dir=output_dir,
        mode="execute",
        se_cfg=se_cfg,
    )

    return assemble_response(
        instance_id=result.get("instance_id", req.instance_id),
        status=result.get("status", "error"),
        output_dir=output_dir,
        error=result.get("error"),
        higher_is_better=se_cfg.metric_higher_is_better,
        return_summary=req.return_summary,
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main():
    import argparse
    import signal
    import subprocess

    parser = argparse.ArgumentParser(description="nanoCSE Evolve HTTP Service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    server_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "api.evolve_server:app",
            "--host", args.host,
            "--port", str(args.port),
            "--workers", str(args.workers),
        ],
        start_new_session=True,
    )

    def _force_shutdown(signum, frame):
        print("\nForce-killing all server processes...", flush=True)
        try:
            os.killpg(os.getpgid(server_proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _force_shutdown)
    signal.signal(signal.SIGTERM, _force_shutdown)

    sys.exit(server_proc.wait())


if __name__ == "__main__":
    main()
