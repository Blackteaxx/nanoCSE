"""
/v1/evolve API request / response schemas.

Pydantic models defining the contract between CSEAgentLoop (verl side)
and the nanoCSE evolutionary search service.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request sub-models
# ---------------------------------------------------------------------------


class LLMConfig(BaseModel):
    """OpenAI-compatible LLM endpoint configuration.

    The ``api_base`` should point to a ``/v1`` root, e.g.
    ``http://host:port/v1``.  The nanoCSE service will append
    ``/chat/completions`` internally.
    """

    name: str | None = None
    api_base: str | None = None
    api_key: str | None = Field(default="EMPTY")
    max_output_tokens: int | None = None
    temperature: float | None = None
    request_timeout: int | None = None


class SearchConfig(BaseModel):
    """Controls the evolutionary search behaviour.

    Callers can either provide a ``config_path`` to a full SE YAML config
    on the server file-system, or supply individual override fields.
    When *both* are given, the YAML file is loaded first and then the
    explicit fields override the corresponding sections.
    """

    config_path: str | None = Field(
        default=None,
        description="Absolute path to a SE YAML config on the server.",
    )
    strategy: dict | None = Field(
        default=None,
        description="Strategy override (same schema as SEPerfRunSEConfig.strategy).",
    )
    local_memory: dict | None = None
    global_memory_bank: dict | None = None
    prompt_config: dict | None = None
    base_config: str | None = Field(
        default=None,
        description="Path to perfagent base config YAML (relative to nanoCSE root).",
    )
    metric_higher_is_better: bool = False
    max_iterations: int = 1


# ---------------------------------------------------------------------------
# Top-level request
# ---------------------------------------------------------------------------


class EvolveRequest(BaseModel):
    """``POST /v1/evolve`` request body."""

    task_type: str = Field(
        default="effibench",
        description="Registered task runner name: effibench | livecodebench | aime.",
    )
    instance_id: str = Field(
        ...,
        description="Unique identifier for this problem instance.",
    )
    instance_payload: dict | None = Field(
        default=None,
        description=(
            "Inline instance data (e.g. question_title, question_content, "
            "public_test_cases …).  Written to a temp file before execution."
        ),
    )
    messages: list[dict] | None = Field(
        default=None,
        description="Reserved — chat context from the RL training prompt.",
    )
    llm_config: LLMConfig
    sampling_params: dict | None = Field(
        default=None,
        description="Extra sampling params forwarded to the LLM client.",
    )
    search_config: SearchConfig = Field(default_factory=SearchConfig)
    return_summary: bool = Field(
        default=True,
        description="Whether to return trajectory summary in response.",
    )
    task_data_path: str | None = Field(
        default=None,
        description="Fallback: path to instance JSON on the server file-system.",
    )


# ---------------------------------------------------------------------------
# Response sub-models
# ---------------------------------------------------------------------------


class CandidateInfo(BaseModel):
    """A single candidate solution produced during the search."""

    solution: str = ""
    metric: float | str | None = None
    iteration: int = 0
    success: bool = False
    artifacts: dict = Field(default_factory=dict)
    label: str | None = Field(
        default=None,
        description="Trajectory pool label (e.g. iter1_sol3).",
    )


class TokenUsage(BaseModel):
    """Aggregated token consumption across all LLM calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class TrajectorySummary(BaseModel):
    """Lightweight summary of the trajectory pool (returned when
    ``return_summary=True``).  Full pool kept in ``raw_output_dir``."""

    total_trajectories: int = 0
    best_label: str | None = None
    labels: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level response
# ---------------------------------------------------------------------------


class EvolveResponse(BaseModel):
    """``POST /v1/evolve`` response body."""

    instance_id: str
    status: str = Field(
        ...,
        description="success | skipped | error",
    )
    best_candidate: CandidateInfo | None = None
    all_candidates: list[CandidateInfo] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    raw_output_dir: str = ""
    trajectory_summary: TrajectorySummary | None = None
    error: str | None = None
