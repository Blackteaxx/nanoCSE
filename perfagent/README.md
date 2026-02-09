# PerfAgent - Code Performance Optimization Agent

PerfAgent is a task-agnostic code performance optimization agent built on top of the CSE framework, designed for iterative code efficiency optimization through LLM-driven refinement.

## Features

- **Iterative Optimization**: Continuously improve code performance through multiple rounds
- **Task-Agnostic Design**: Plugin-based TaskRunner system supports multiple task types
- **Performance Evaluation**: Evaluate code efficiency using EffiBench-X and LiveCodeBench benchmarks
- **Trajectory Recording**: Complete logging of optimization process for analysis and reproduction
- **Diff Application**: Automatic parsing and application of SEARCH/REPLACE code modifications
- **Flexible Configuration**: Support for YAML configuration files and command-line overrides
- **Protocol Interface**: Standardized `AgentRequest` / `AgentResult` communication with SE_Perf

## Directory Structure

```text
perfagent/
├── __init__.py                  # Package entry, exports PerfAgent / PerfAgentConfig
├── agent.py                     # Core PerfAgent class (iterative optimization loop)
├── config.py                    # Configuration management (YAML loading)
├── diff_applier.py              # SEARCH/REPLACE diff parser and applier
├── llm_client.py                # LLM API client (retry / token stats / IO logging)
├── protocols.py                 # AgentRequest / AgentResult protocol definitions
├── run.py                       # CLI single-instance runner
├── task_registry.py             # TaskRunner registration factory
├── task_runner.py               # BaseTaskRunner abstract interface
├── trajectory.py                # Trajectory logging system
├── config_example.yaml          # Config example (new format)
├── config_example_original.yaml # Config example (old format, backward compat)
├── pytest.ini                   # Pytest configuration
├── README.md
│
├── effibench/                   # EffiBench-X performance evaluation
│   ├── __init__.py
│   ├── analysis.py              #   Statistical analysis (IQR / confidence interval)
│   ├── benchmark.py             #   Performance benchmark orchestrator
│   ├── run_tests.py             #   Test execution (sandbox backend submission)
│   ├── utils.py                 #   Language registry & code processing utilities
│   └── backends/
│       ├── __init__.py
│       └── backend_utils.py     #   Sandbox backend client & health management
│
├── lcb_eval/                    # LiveCodeBench evaluation
│   ├── __init__.py
│   └── testing_util.py          #   Test execution (compile / run / compare)
│
├── tasks/                       # Task-specific TaskRunner implementations
│   ├── __init__.py
│   ├── effibench.py             #   EffiBenchRunner (performance optimization)
│   └── livecodebench.py         #   LiveCodeBenchRunner (code generation)
│
├── tests/                       # Test suite
│   ├── conftest.py              #   Shared pytest fixtures
│   ├── test_agent.py            #   PerfAgent core tests
│   ├── test_config.py           #   Config system tests
│   ├── test_diff_applier.py     #   Diff parser tests
│   ├── test_integration.py      #   Integration tests
│   ├── test_run_refactored.py   #   Refactored run flow tests
│   ├── test_trajectory.py       #   Trajectory logger tests
│   └── README.md
│
└── utils/                       # Utility modules
    ├── json_utils.py            #   JSON-safe serialization helper
    └── log.py                   #   Logging utilities
```

## Key Modules

### Core Components

| File               | Core Class / Function                            | Responsibility                                               |
| ------------------ | ------------------------------------------------ | ------------------------------------------------------------ |
| `agent.py`         | `PerfAgent`, `RunContext`                         | Iterative optimization loop: LLM gen -> eval -> feedback -> refine |
| `config.py`        | `PerfAgentConfig`, `ModelConfig`                  | YAML config loading, CLI overrides, backward compatibility   |
| `protocols.py`     | `AgentRequest`, `AgentResult`, `TaskMetadata`     | Standardized SE_Perf <-> PerfAgent communication protocol    |
| `diff_applier.py`  | `DiffApplier`                                     | SEARCH/REPLACE block parsing and code modification           |
| `llm_client.py`    | `LLMClient`                                       | LLM calls with retry, token stats, IO logging                |
| `task_runner.py`   | `BaseTaskRunner` (ABC)                            | Task plugin interface: load / evaluate / prompt / extract    |
| `task_registry.py` | `register_task_runner()`, `create_task_runner()`  | Task type registration factory                               |
| `trajectory.py`    | `TrajectoryLogger`, `OptimizationStep`            | Trajectory recording (steps / history / metadata)            |
| `run.py`           | `run_single_instance()`, `main()`                 | CLI single-instance execution entry point                    |

### Task Plugins (tasks/)

| File              | Class                | Task Type               | Key Features                                     |
| ----------------- | -------------------- | ----------------------- | ------------------------------------------------ |
| `effibench.py`    | `EffiBenchRunner`    | Performance optimization | Cascade benchmark (single -> multi run), diff/direct mode |
| `livecodebench.py`| `LiveCodeBenchRunner`| Code generation          | Public/private test cases, detailed failure feedback |

### EffiBench Evaluation Infrastructure (effibench/)

| File                       | Core Function / Class            | Responsibility                                       |
| -------------------------- | -------------------------------- | ---------------------------------------------------- |
| `analysis.py`              | `analyze_samples()`              | IQR outlier removal, confidence intervals, trimmed mean |
| `benchmark.py`             | `run_performance_benchmark()`    | Concurrent runs, aggregate runtime/memory/integral   |
| `run_tests.py`             | `run_tests()`                    | Sandbox backend submission and result polling         |
| `utils.py`                 | `EFFIBENCH_REGISTRY`             | Language registry (Python/C++/Java/JS/Go/Ruby), code assembly |
| `backends/backend_utils.py`| `BackendManager`                 | Sandbox health monitoring and automatic failover     |

### LiveCodeBench Evaluation (lcb_eval/)

| File              | Core Function | Responsibility                                        |
| ----------------- | ------------- | ----------------------------------------------------- |
| `testing_util.py` | `run_test()`  | Compile & execute code, evaluate outputs, security sandbox |

## Quick Start

### Single Instance Run

```bash
python -m perfagent.run \
    --instance /path/to/instance.json \
    --base-dir /path/to/output
```

### Command Line Arguments

| Argument           | Description                          |
| ------------------ | ------------------------------------ |
| `--config`         | Configuration file path              |
| `--instance`       | Single instance file path            |
| `--output`         | Result output file path              |
| `--base-dir`       | Instance output base directory       |
| `--max-iterations` | Maximum number of iterations         |
| `--model`          | Model name                           |
| `--log-level`      | Log level (DEBUG/INFO/WARNING/ERROR) |
| `--trajectory-dir` | Trajectory save directory            |
| `--log-dir`        | Log save directory                   |

## Configuration

Create a YAML configuration file to customize PerfAgent behavior:

```yaml
# Basic configuration
max_iterations: 10
time_limit: 300
memory_limit: 1024

# Model configuration
model_name: "deepseek-chat"
temperature: 0.1
max_tokens: 4000

# Performance evaluation configuration
num_runs: 5
trim_ratio: 0.1
max_workers: 4

# Task-specific configuration
task_config:
  task_type: "effibench"
  # ... task-specific settings

# Trajectory and log configuration
save_trajectory: true
trajectory_dir: "./trajectories"
log_dir: "./logs"
log_level: "INFO"
```

## Architecture

### Optimization Flow

1. **Initialization**: Load configuration and instance data via TaskRunner
2. **Performance Evaluation**: Evaluate initial code performance
3. **Iterative Optimization**:
   - Generate optimization suggestions via LLM
   - Parse and apply SEARCH/REPLACE diff
   - Evaluate optimized performance via TaskRunner
   - Record optimization history in trajectory
4. **Result Output**: Save best code and trajectory

### TaskRunner Plugin System

PerfAgent uses a plugin-based design. Each task type implements `BaseTaskRunner`:

```python
from perfagent.task_runner import BaseTaskRunner

class MyTaskRunner(BaseTaskRunner):
    @classmethod
    def load_metadata(cls, instance_path) -> TaskMetadata: ...
    def load_instance(self, instance_path): ...
    def get_initial_solution(self) -> str: ...
    def evaluate(self, solution) -> tuple[float, dict]: ...
    def build_system_prompt(self) -> str: ...
    def build_optimization_prompt(self, ...) -> str: ...
    def extract_solution(self, response) -> str: ...
```

Registered task types: `effibench`, `livecodebench`, `aime`

## Output Files

### Trajectory File

Saved in `<base_dir>/<task_name>/<task_name>.traj`:

```json
{
  "metadata": {
    "instance_id": "test_001",
    "start_time": "2024-01-01T10:00:00",
    "end_time": "2024-01-01T10:05:00",
    "total_iterations": 5,
    "success": true
  },
  "steps": [
    {
      "step_id": 1,
      "timestamp": "...",
      "action": "initial_evaluation",
      "input_data": {...},
      "output_data": {...},
      "performance_metrics": {...}
    }
  ]
}
```

## Testing

```bash
# Run all tests
python -m pytest perfagent/tests/

# Run specific test
python -m pytest perfagent/tests/test_agent.py
```

## Security & Configuration Tips

- **API Keys**: Do not store plaintext API keys in the repository. Use environment variables:

  ```bash
  export OPENROUTER_API_KEY=xxxxx
  ```

- **LLM Logging**: Enable request/response logging with `--llm-log-io` and `--llm-log-sanitize`

- **Early Stopping**: Use `--early-stop-no-improve N` or set `early_stop_no_improve: N` in YAML

## Important Notes

1. Ensure sufficient disk space for trajectory and log files
2. Adjust `time_limit` and `memory_limit` based on actual requirements
3. EffiBench-X backend service must be running for performance evaluation
4. Check network connectivity for API calls

## Related Documentation

- [Main Project README](../README.md)
- [SE_Perf Documentation](../SE_Perf/README.md)
- [Configuration Examples](config_example.yaml)
