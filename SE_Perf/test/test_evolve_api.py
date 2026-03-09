"""
Unit tests for the ``/v1/evolve`` API layer.

Coverage:
  1. Schema validation (request / response models)
  2. Result assembler (``assemble_response`` & internal helpers)
  3. HTTP endpoint (``POST /v1/evolve``) with monkeypatched ``run_single_instance``
  4. Error boundaries (missing fields, bad instance path, execution failure)
  5. ``raw_output_dir`` & trajectory summary

Run::

    cd nanoCSE/SE_Perf && python -m pytest test/test_evolve_api.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Ensure SE_Perf root is on sys.path so that ``perf_config`` etc. resolve
_SE_PERF_ROOT = str(Path(__file__).resolve().parent.parent)
if _SE_PERF_ROOT not in sys.path:
    sys.path.insert(0, _SE_PERF_ROOT)

from api.evolve_server import _build_se_config_dict, app
from api.result_assembler import (
    _enrich_candidates_from_pool,
    _pick_best,
    _read_preds,
    _read_token_usage,
    _read_traj_pool_summary,
    assemble_response,
)
from api.schemas import (
    CandidateInfo,
    EvolveRequest,
    EvolveResponse,
    LLMConfig,
    SearchConfig,
    TokenUsage,
    TrajectorySummary,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MINIMAL_LLM_CONFIG = {
    "name": "test-model",
    "api_base": "http://localhost:8000/v1",
    "api_key": "EMPTY",
    "max_output_tokens": 4096,
    "temperature": 0.7,
}


def _make_evolve_body(**overrides) -> dict:
    """Build a minimal valid ``/v1/evolve`` request body."""
    body = {
        "task_type": "effibench",
        "instance_id": "test_001",
        "instance_payload": {
            "question_title": "A. Short Sort",
            "question_content": "Given a string of length 3...",
            "question_id": "1873_A",
            "public_test_cases": [],
        },
        "llm_config": _MINIMAL_LLM_CONFIG,
        "search_config": {
            "strategy": {"iterations": [{"operator": "plan", "num": 2, "trajectory_labels": ["s1", "s2"]}]},
        },
    }
    body.update(overrides)
    return body


@pytest.fixture
def output_dir(tmp_path: Path):
    """Create a temporary output directory pre-populated with mock artefacts."""
    d = tmp_path / "output"
    d.mkdir()
    return d


def _write_mock_artefacts(
    output_dir: Path,
    *,
    num_candidates: int = 3,
    with_pool: bool = True,
    with_tokens: bool = True,
    higher_is_better: bool = False,
):
    """Populate ``output_dir`` with preds.json, traj.pool, token_usage.jsonl."""
    # preds.json
    entries = []
    for i in range(1, num_candidates + 1):
        entries.append(
            {
                "iteration": i,
                "solution": f"def solve_{i}(): pass",
                "metric": float(i) * 0.5,
                "success": True,
                "artifacts": {"lang": "python"},
            }
        )
    preds = {"test_001": entries}
    (output_dir / "preds.json").write_text(json.dumps(preds), encoding="utf-8")

    # traj.pool
    if with_pool:
        pool: dict = {"problem": "Optimize this function"}
        for i in range(1, num_candidates + 1):
            label = f"iter1_sol{i}"
            pool[label] = {
                "label": label,
                "iteration": i,
                "solution": f"def solve_{i}(): pass",
                "metric": float(i) * 0.5,
            }
        (output_dir / "traj.pool").write_text(json.dumps(pool), encoding="utf-8")

    # token_usage.jsonl
    if with_tokens:
        lines = [
            json.dumps({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}),
            json.dumps({"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280}),
        ]
        (output_dir / "token_usage.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mock_run_single_instance(
    config_path: str,
    instance_path: str,
    output_dir: str,
    mode: str = "execute",
    se_cfg=None,
) -> dict:
    """Stub that writes mock artefacts instead of actually running the search."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_mock_artefacts(out)
    return {
        "instance_id": "test_001",
        "status": "success",
        "output_dir": output_dir,
        "error": None,
        "best_metric": 0.5,
    }


def _mock_run_error(
    config_path: str,
    instance_path: str,
    output_dir: str,
    mode: str = "execute",
    se_cfg=None,
) -> dict:
    """Stub that simulates an execution error."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return {
        "instance_id": "test_001",
        "status": "error",
        "output_dir": output_dir,
        "error": "TaskRunner crashed",
        "best_metric": None,
    }


# ===================================================================
# 1. Schema validation tests
# ===================================================================


class TestSchemaValidation:
    """Request / response Pydantic model tests."""

    def test_minimal_request(self):
        req = EvolveRequest(
            instance_id="x1",
            llm_config=LLMConfig(name="m"),
        )
        assert req.task_type == "effibench"
        assert req.instance_payload is None
        assert req.search_config.metric_higher_is_better is False
        assert req.return_summary is True

    def test_full_request_roundtrip(self):
        body = _make_evolve_body()
        req = EvolveRequest(**body)
        data = req.model_dump()
        req2 = EvolveRequest(**data)
        assert req2.instance_id == req.instance_id
        assert req2.llm_config.api_base == req.llm_config.api_base

    def test_instance_id_required(self):
        with pytest.raises(Exception):
            EvolveRequest(llm_config=LLMConfig())

    def test_llm_config_defaults(self):
        lc = LLMConfig()
        assert lc.api_key == "EMPTY"
        assert lc.temperature is None

    def test_search_config_defaults(self):
        sc = SearchConfig()
        assert sc.metric_higher_is_better is False
        assert sc.max_iterations == 1
        assert sc.strategy is None

    def test_response_json_roundtrip(self):
        resp = EvolveResponse(
            instance_id="x",
            status="success",
            best_candidate=CandidateInfo(solution="s", metric=1.0),
            all_candidates=[CandidateInfo(solution="s", metric=1.0)],
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            raw_output_dir="/tmp/x",
            trajectory_summary=TrajectorySummary(total_trajectories=1, best_label="a", labels=["a"]),
        )
        blob = resp.model_dump_json()
        resp2 = EvolveResponse.model_validate_json(blob)
        assert resp2.best_candidate.solution == "s"
        assert resp2.trajectory_summary.total_trajectories == 1


# ===================================================================
# 2. Result assembler tests
# ===================================================================


class TestResultAssembler:
    """Tests for result_assembler module functions."""

    # -- _read_preds --

    def test_read_preds_normal(self, output_dir: Path):
        _write_mock_artefacts(output_dir, num_candidates=3)
        candidates = _read_preds(output_dir)
        assert len(candidates) == 3
        assert candidates[0].solution == "def solve_1(): pass"

    def test_read_preds_missing_file(self, output_dir: Path):
        assert _read_preds(output_dir) == []

    def test_read_preds_corrupt_file(self, output_dir: Path):
        (output_dir / "preds.json").write_text("not json {{{", encoding="utf-8")
        assert _read_preds(output_dir) == []

    def test_read_preds_non_list_entries(self, output_dir: Path):
        (output_dir / "preds.json").write_text(json.dumps({"inst": "not_a_list"}), encoding="utf-8")
        assert _read_preds(output_dir) == []

    # -- _pick_best --

    def test_pick_best_lower_is_better(self):
        candidates = [
            CandidateInfo(solution="a", metric=2.0),
            CandidateInfo(solution="b", metric=0.5),
            CandidateInfo(solution="c", metric=1.5),
        ]
        best = _pick_best(candidates, higher_is_better=False)
        assert best.solution == "b"

    def test_pick_best_higher_is_better(self):
        candidates = [
            CandidateInfo(solution="a", metric=2.0),
            CandidateInfo(solution="b", metric=0.5),
        ]
        best = _pick_best(candidates, higher_is_better=True)
        assert best.solution == "a"

    def test_pick_best_empty(self):
        assert _pick_best([]) is None

    def test_pick_best_none_metrics(self):
        candidates = [
            CandidateInfo(solution="a", metric=None),
            CandidateInfo(solution="b", metric=1.0),
        ]
        best = _pick_best(candidates, higher_is_better=False)
        assert best.solution == "b"

    def test_pick_best_inf_metric(self):
        candidates = [
            CandidateInfo(solution="a", metric=float("inf")),
            CandidateInfo(solution="b", metric=3.0),
        ]
        best = _pick_best(candidates, higher_is_better=False)
        assert best.solution == "b"

    def test_pick_best_string_metric(self):
        candidates = [
            CandidateInfo(solution="a", metric="nan"),
            CandidateInfo(solution="b", metric=2.0),
        ]
        best = _pick_best(candidates, higher_is_better=False)
        assert best.solution == "b"

    # -- _read_token_usage --

    def test_read_token_usage_normal(self, output_dir: Path):
        _write_mock_artefacts(output_dir, with_pool=False)
        usage = _read_token_usage(output_dir)
        assert usage.prompt_tokens == 300
        assert usage.completion_tokens == 130
        assert usage.total_tokens == 430

    def test_read_token_usage_missing(self, output_dir: Path):
        usage = _read_token_usage(output_dir)
        assert usage == TokenUsage()

    def test_read_token_usage_malformed_lines(self, output_dir: Path):
        (output_dir / "token_usage.jsonl").write_text(
            "invalid_json\n" + json.dumps({"prompt_tokens": 10, "completion_tokens": 5}) + "\n",
            encoding="utf-8",
        )
        usage = _read_token_usage(output_dir)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 5

    # -- _read_traj_pool_summary --

    def test_read_traj_pool_summary(self, output_dir: Path):
        _write_mock_artefacts(output_dir, num_candidates=3)
        summary = _read_traj_pool_summary(output_dir)
        assert summary is not None
        assert summary.total_trajectories == 3
        assert summary.best_label == "iter1_sol1"
        assert len(summary.labels) == 3

    def test_read_traj_pool_summary_missing(self, output_dir: Path):
        assert _read_traj_pool_summary(output_dir) is None

    def test_read_traj_pool_summary_empty_pool(self, output_dir: Path):
        (output_dir / "traj.pool").write_text(json.dumps({"problem": "desc"}), encoding="utf-8")
        summary = _read_traj_pool_summary(output_dir)
        assert summary.total_trajectories == 0

    def test_read_traj_pool_summary_entries_no_metric(self, output_dir: Path):
        pool = {"problem": "desc", "sol1": {"label": "sol1", "solution": "x"}}
        (output_dir / "traj.pool").write_text(json.dumps(pool), encoding="utf-8")
        summary = _read_traj_pool_summary(output_dir)
        assert summary.total_trajectories == 1
        assert summary.best_label is None

    # -- _enrich_candidates_from_pool --

    def test_enrich_labels(self, output_dir: Path):
        _write_mock_artefacts(output_dir, num_candidates=2)
        candidates = [
            CandidateInfo(solution="def solve_1(): pass"),
            CandidateInfo(solution="def solve_2(): pass"),
        ]
        _enrich_candidates_from_pool(candidates, output_dir)
        assert candidates[0].label == "iter1_sol1"
        assert candidates[1].label == "iter1_sol2"

    def test_enrich_no_pool_file(self, output_dir: Path):
        candidates = [CandidateInfo(solution="x")]
        _enrich_candidates_from_pool(candidates, output_dir)
        assert candidates[0].label is None

    def test_enrich_preserves_existing_label(self, output_dir: Path):
        _write_mock_artefacts(output_dir, num_candidates=1)
        candidates = [CandidateInfo(solution="def solve_1(): pass", label="manual")]
        _enrich_candidates_from_pool(candidates, output_dir)
        assert candidates[0].label == "manual"

    # -- assemble_response --

    def test_assemble_success(self, output_dir: Path):
        _write_mock_artefacts(output_dir)
        resp = assemble_response(
            instance_id="test_001",
            status="success",
            output_dir=str(output_dir),
        )
        assert resp.status == "success"
        assert resp.best_candidate is not None
        assert resp.best_candidate.metric == 0.5
        assert resp.best_candidate.label == "iter1_sol1"
        assert len(resp.all_candidates) == 3
        assert resp.token_usage.total_tokens == 430
        assert resp.trajectory_summary is not None
        assert resp.raw_output_dir == str(output_dir)
        assert resp.error is None

    def test_assemble_error(self, output_dir: Path):
        resp = assemble_response(
            instance_id="test_001",
            status="error",
            output_dir=str(output_dir),
            error="boom",
        )
        assert resp.status == "error"
        assert resp.error == "boom"
        assert resp.best_candidate is None
        assert resp.all_candidates == []

    def test_assemble_no_summary(self, output_dir: Path):
        _write_mock_artefacts(output_dir)
        resp = assemble_response(
            instance_id="test_001",
            status="success",
            output_dir=str(output_dir),
            return_summary=False,
        )
        assert resp.trajectory_summary is None

    def test_assemble_higher_is_better(self, output_dir: Path):
        _write_mock_artefacts(output_dir, num_candidates=3)
        resp = assemble_response(
            instance_id="test_001",
            status="success",
            output_dir=str(output_dir),
            higher_is_better=True,
        )
        assert resp.best_candidate.metric == 1.5


# ===================================================================
# 3. Config assembly tests
# ===================================================================


class TestConfigAssembly:
    """Tests for ``_build_se_config_dict``."""

    def test_basic_config_assembly(self):
        req = EvolveRequest(**_make_evolve_body())
        cfg = _build_se_config_dict(req)
        assert cfg["task_type"] == "effibench"
        assert cfg["model"]["name"] == "test-model"
        assert cfg["model"]["api_base"] == "http://localhost:8000/v1"
        assert cfg["model"]["max_output_tokens"] == 4096
        assert cfg["strategy"]["iterations"][0]["operator"] == "plan"

    def test_config_from_yaml_file(self, tmp_path: Path):
        yaml_path = tmp_path / "base.yaml"
        yaml_path.write_text(
            "task_type: aime\nmodel:\n  name: from-yaml\n  temperature: 0.3\n"
            "strategy:\n  iterations:\n    - operator: plan\n      num: 1\n",
            encoding="utf-8",
        )
        req = EvolveRequest(
            **_make_evolve_body(
                search_config={
                    "config_path": str(yaml_path),
                    "strategy": None,
                }
            )
        )
        cfg = _build_se_config_dict(req)
        assert cfg["model"]["name"] == "test-model"
        assert cfg["model"]["temperature"] == 0.7
        assert cfg["strategy"]["iterations"][0]["num"] == 1

    def test_sampling_params_merged(self):
        req = EvolveRequest(**_make_evolve_body(sampling_params={"top_p": 0.9}))
        cfg = _build_se_config_dict(req)
        assert cfg["model"]["extras"]["top_p"] == 0.9

    def test_request_timeout_in_extras(self):
        body = _make_evolve_body()
        body["llm_config"]["request_timeout"] = 120
        req = EvolveRequest(**body)
        cfg = _build_se_config_dict(req)
        assert cfg["model"]["extras"]["request_timeout"] == 120


# ===================================================================
# 4. HTTP endpoint tests (with monkeypatch)
# ===================================================================


class TestEvolveEndpoint:
    """Integration-style tests using FastAPI TestClient."""

    @pytest.fixture(autouse=True)
    def _patch_run(self):
        with patch(
            "api.evolve_server._invoke_run_single_instance",
            side_effect=_mock_run_single_instance,
        ) as mock_run:
            self.mock_run = mock_run
            yield

    @pytest.fixture
    def client(self):
        return TestClient(app, raise_server_exceptions=False)

    def test_success_response(self, client: TestClient):
        resp = client.post("/v1/evolve", json=_make_evolve_body())
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["instance_id"] == "test_001"
        assert data["best_candidate"] is not None
        assert data["best_candidate"]["metric"] == 0.5
        assert len(data["all_candidates"]) == 3
        assert data["raw_output_dir"] != ""
        assert data["error"] is None

    def test_token_usage_populated(self, client: TestClient):
        resp = client.post("/v1/evolve", json=_make_evolve_body())
        data = resp.json()
        assert data["token_usage"]["prompt_tokens"] == 300
        assert data["token_usage"]["completion_tokens"] == 130

    def test_trajectory_summary_returned(self, client: TestClient):
        resp = client.post("/v1/evolve", json=_make_evolve_body())
        data = resp.json()
        summary = data["trajectory_summary"]
        assert summary is not None
        assert summary["total_trajectories"] == 3
        assert "iter1_sol1" in summary["labels"]

    def test_no_summary_when_disabled(self, client: TestClient):
        body = _make_evolve_body(return_summary=False)
        resp = client.post("/v1/evolve", json=body)
        data = resp.json()
        assert data["trajectory_summary"] is None

    def test_raw_output_dir_exists(self, client: TestClient):
        resp = client.post("/v1/evolve", json=_make_evolve_body())
        data = resp.json()
        assert Path(data["raw_output_dir"]).is_dir()

    def test_candidates_have_labels(self, client: TestClient):
        resp = client.post("/v1/evolve", json=_make_evolve_body())
        data = resp.json()
        labels = [c["label"] for c in data["all_candidates"]]
        assert "iter1_sol1" in labels

    def test_task_data_path_fallback(self, client: TestClient, tmp_path: Path):
        inst_file = tmp_path / "inst.json"
        inst_file.write_text(json.dumps({"question_id": "42"}), encoding="utf-8")
        body = _make_evolve_body(instance_payload=None, task_data_path=str(inst_file))
        resp = client.post("/v1/evolve", json=body)
        assert resp.status_code == 200

    def test_healthz(self, client: TestClient):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ===================================================================
# 5. Error boundary tests
# ===================================================================


class TestErrorBoundaries:
    """Validate that bad inputs / failures are handled gracefully."""

    @pytest.fixture
    def client(self):
        return TestClient(app, raise_server_exceptions=False)

    def test_missing_instance_id(self, client: TestClient):
        body = _make_evolve_body()
        del body["instance_id"]
        resp = client.post("/v1/evolve", json=body)
        assert resp.status_code == 422

    def test_missing_llm_config(self, client: TestClient):
        body = _make_evolve_body()
        del body["llm_config"]
        resp = client.post("/v1/evolve", json=body)
        assert resp.status_code == 422

    def test_no_payload_and_no_path(self, client: TestClient):
        body = _make_evolve_body()
        body["instance_payload"] = None
        body["task_data_path"] = None
        with patch(
            "api.evolve_server._invoke_run_single_instance",
            side_effect=_mock_run_single_instance,
        ):
            resp = client.post("/v1/evolve", json=body)
        assert resp.status_code == 422
        assert "instance_payload" in resp.json()["detail"].lower() or "task_data_path" in resp.json()["detail"].lower()

    def test_nonexistent_task_data_path(self, client: TestClient):
        body = _make_evolve_body(
            instance_payload=None,
            task_data_path="/nonexistent/path/inst.json",
        )
        with patch(
            "api.evolve_server._invoke_run_single_instance",
            side_effect=_mock_run_single_instance,
        ):
            resp = client.post("/v1/evolve", json=body)
        assert resp.status_code == 422
        assert "not found" in resp.json()["detail"].lower()

    def test_run_single_instance_error_returned(self, client: TestClient):
        with patch(
            "api.evolve_server._invoke_run_single_instance",
            side_effect=_mock_run_error,
        ):
            resp = client.post("/v1/evolve", json=_make_evolve_body())
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
        assert data["error"] == "TaskRunner crashed"
        assert data["best_candidate"] is None

    def test_empty_body(self, client: TestClient):
        resp = client.post("/v1/evolve", json={})
        assert resp.status_code == 422

    def test_invalid_json(self, client: TestClient):
        resp = client.post(
            "/v1/evolve",
            content="not-json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422
