# SE_Perf - Core Evolution Framework

The core evolution framework module of CSE (Controlled Self-Evolution), implementing diversified planning initialization, controlled genetic evolution, and hierarchical memory systems.

## Module Overview

SE_Perf is the core engine of EvoControl, responsible for:

- **Evolution Strategy Orchestration**: Coordinating multi-round iterative evolution
- **Operator System**: Implementing Plan, Mutation, Crossover, and other evolution operators
- **Memory System**: Managing Local Memory and Global Memory
- **Parallel Execution**: Supporting multi-instance parallel optimization

## Directory Structure

```text
SE_Perf/
├── perf_run.py                      # Main entry - single instance evolution runner
├── perf_config.py                   # Configuration parser (typed dataclasses)
├── iteration_executor.py            # Iteration step executor
├── trajectory_handler.py            # Trajectory processing and summarization
├── results_io.py                    # Result I/O and aggregation
├── run_models.py                    # Pipeline data models
├── run_helpers.py                   # Helper functions
├── README.md
├── operators.md
│
├── core/
│   ├── __init__.py
│   ├── global_memory/               # Global memory system (inter-task learning)
│   │   ├── __init__.py
│   │   ├── bank.py                  #   GlobalMemoryBank - vector storage
│   │   ├── utils/
│   │   │   └── config.py            #   Global memory config dataclasses
│   │   ├── embeddings/
│   │   │   └── openai.py            #   OpenAI Embedding wrapper
│   │   └── memory/
│   │       ├── base.py              #   MemoryBackend abstract base class
│   │       └── chroma.py            #   ChromaDB backend implementation
│   │
│   └── utils/                       # Core utilities
│       ├── __init__.py
│       ├── llm_client.py            #   Unified LLM client (retry / logging)
│       ├── problem_manager.py       #   Problem description manager
│       ├── se_logger.py             #   SE logging system
│       ├── log.py                   #   Base logging infrastructure
│       ├── traj_pool_manager.py     #   Trajectory pool manager
│       ├── traj_extractor.py        #   Trajectory data extractor
│       ├── traj_summarizer.py       #   Trajectory summarization prompts
│       ├── trajectory_processor.py  #   .tra file processor
│       ├── local_memory_manager.py  #   Local memory manager (intra-task)
│       ├── global_memory_manager.py #   Global memory manager (inter-task)
│       ├── instance_data_manager.py #   Instance data unified interface
│       └── generate_tra_files.py    #   CLI tool for .tra generation
│
├── operators/                       # Evolution operator system
│   ├── __init__.py
│   ├── README.md                    #   Operator development guide
│   ├── base.py                      #   BaseOperator / OperatorContext
│   ├── registry.py                  #   Operator registration system
│   ├── plan.py                      #   PlanOperator - diverse planning
│   ├── crossover.py                 #   CrossoverOperator - trajectory crossover
│   ├── filter.py                    #   FilterTrajectoriesOperator - pool filtering
│   ├── reflection_refine.py         #   ReflectionRefineOperator - reflection refinement
│   ├── alternative_strategy.py      #   AlternativeStrategyOperator - orthogonal strategy
│   ├── traj_pool_summary.py         #   TrajPoolSummaryOperator - pool analysis
│   └── trajectory_analyzer.py       #   TrajectoryAnalyzerOperator - trajectory analysis
│
└── test/                            # Test suite
    ├── run_operator_tests.py
    ├── test_operators.py
    ├── test_alternative_strategy.py
    ├── test_global_memory.py
    ├── test_instance_data_system.py
    ├── test_operator_data_access.py
    ├── test_operator_weighted_selection.py
    ├── test_problem_interface.py
    ├── test_traj_extractor.py
    ├── test_traj_pool_summary.py
    ├── test_traj_pool.py
    ├── test_unified_data_interface.py
    └── ...
```

## Key Modules

### Root Level - Execution Pipeline

| File                    | Core Classes / Functions                                            | Responsibility                                      |
| ----------------------- | ------------------------------------------------------------------- | --------------------------------------------------- |
| `perf_run.py`           | `main()`                                                            | Entry point: init LLM / memory / traj pool, orchestrate iterations |
| `perf_config.py`        | `ModelConfig`, `StrategyConfig`, `SEPerfRunSEConfig`                | Typed configuration parsing                         |
| `iteration_executor.py` | `execute_iteration()`, `run_single_perfagent()`, `run_operator()`   | Single iteration orchestration: operator -> PerfAgent -> post-process |
| `trajectory_handler.py` | `build_trajectory_from_result()`, `process_and_summarize()`         | Build trajectory from AgentResult and update pool   |
| `results_io.py`         | `aggregate_all_iterations_preds()`, `write_final_json_from_preds()` | Result writing and best-solution aggregation        |
| `run_models.py`         | `TrajectoryData`, `GlobalMemoryContext`, `PredictionEntry`          | Pipeline data models                                |
| `run_helpers.py`        | `build_perf_agent_config()`, `retrieve_global_memory()`             | Config building and global memory retrieval         |

### core/utils/ - Core Utilities

| File                       | Core Class              | Responsibility                                              |
| -------------------------- | ----------------------- | ----------------------------------------------------------- |
| `llm_client.py`            | `LLMClient`             | Unified LLM calls with exponential backoff, token logging   |
| `traj_pool_manager.py`     | `TrajPoolManager`       | Trajectory pool CRUD, best-label selection, step extraction |
| `local_memory_manager.py`  | `LocalMemoryManager`    | Intra-task memory (direction board + experience library)    |
| `global_memory_manager.py` | `GlobalMemoryManager`   | Inter-task vector-retrieval memory                          |
| `instance_data_manager.py` | `InstanceDataManager`   | Unified instance data access interface                      |
| `traj_extractor.py`        | `TrajExtractor`         | Extract instance data from iteration directories            |
| `traj_summarizer.py`       | `TrajSummarizer`        | Trajectory summarization prompt templates                   |
| `trajectory_processor.py`  | `TrajectoryProcessor`   | .traj to .tra file conversion                               |
| `problem_manager.py`       | `ProblemManager`        | Standardized problem description interface                  |

### core/global_memory/ - Global Memory System

| File                | Core Class              | Responsibility                         |
| ------------------- | ----------------------- | -------------------------------------- |
| `bank.py`           | `GlobalMemoryBank`      | Vector storage for cross-task experiences |
| `utils/config.py`   | `GlobalMemoryConfig`    | Configuration dataclasses              |
| `embeddings/openai.py` | `OpenAIEmbeddingModel` | OpenAI embedding wrapper              |
| `memory/base.py`    | `MemoryBackend`         | Abstract base class for backends       |
| `memory/chroma.py`  | `ChromaMemoryBackend`   | ChromaDB implementation                |

### operators/ - Evolution Operator System

| Operator                      | Class                          | Function                                   |
| ----------------------------- | ------------------------------ | ------------------------------------------ |
| `plan`                        | `PlanOperator`                 | Generate K diverse algorithmic strategies  |
| `reflection_refine`           | `ReflectionRefineOperator`     | Feedback-guided controlled mutation        |
| `crossover`                   | `CrossoverOperator`            | Compositional crossover, merge strengths   |
| `filter`                      | `FilterTrajectoriesOperator`   | Clustering-based trajectory pool filtering |
| `alternative_strategy`        | `AlternativeStrategyOperator`  | Explore orthogonal alternative strategies  |
| `traj_pool_summary`           | `TrajPoolSummaryOperator`      | Analyze pool for risk-aware guidance       |
| `trajectory_analyzer`         | `TrajectoryAnalyzerOperator`   | Analyze trajectory snapshot for strategy   |

> For detailed operator development guide, see [operators/README.md](operators/README.md)

## Quick Start

### Single Instance Run

```bash
python SE_Perf/perf_run.py \
    --config configs/Plan-Weighted-Local-Global-30.yaml \
    --instance instances/aizu_1444_yokohama-phenomena.json \
    --output-dir trajectories_perf/test_run
```

## Configuration System

### Two-Layer Configuration Architecture

| Config Type         | File                                         | Purpose                                   |
| ------------------- | -------------------------------------------- | ----------------------------------------- |
| **Base Config**     | `configs/perf_configs/config_integral.yaml`  | Model parameters, runtime limits, prompts |
| **Strategy Config** | `configs/Plan-Weighted-Local-Global-30.yaml` | Evolution strategy orchestration          |

### Strategy Configuration Example

```yaml
# Model configuration
model:
  name: "deepseek-chat"
  api_base: "https://api.deepseek.com/v1"
  api_key: "your-api-key"

# Operator model configuration
operator_models:
  name: "deepseek-chat"
  api_base: "https://api.deepseek.com/v1"
  api_key: "your-api-key"

# Global memory configuration
global_memory_bank:
  enabled: true
  embedding_model:
    model: "text-embedding-3-small"
    api_key: "your-embedding-key"

# Strategy orchestration
strategy:
  iterations:
    - operator: "plan"
      num: 5
      trajectory_labels: ["iter1_sol1", "iter1_sol2", ...]
    - operator: "reflection_refine"
      trajectory_label: "iter1_sol6"
    - operator: "crossover"
      trajectory_label: "iter1_sol7"
```

## Memory System

### Local Memory (Intra-task)

- Records success/failure experiences for current task
- Avoids repeated exploration of failed directions
- Guides optimization direction for subsequent iterations

### Global Memory (Inter-task)

- Extracts reusable optimization patterns from successful cases
- Retrieves relevant experiences based on semantic similarity
- Accelerates optimization process for new tasks

## Output Structure

```text
trajectories_perf/experiment_{timestamp}/
├── {instance_name}/
│   ├── iteration_{n}/
│   │   ├── result.json         # Evaluation results
│   │   └── *.traj              # Trajectory files
│   ├── final.json              # Best solution
│   ├── traj.pool               # Solution pool
│   ├── token_usage.jsonl       # Token usage log
│   └── se_framework.log        # Execution log
├── all_hist.json               # Aggregated history
├── final.json                  # All best solutions
└── total_token_usage.json      # Token statistics
```

## Development & Testing

```bash
# Run test suite
python SE_Perf/test/run_operator_tests.py

# Test specific operators
python SE_Perf/test/test_operators.py

# Test global memory
python SE_Perf/test/test_global_memory.py
```

## Important Notes

1. **Working Directory**: Commands must be executed from the project root
2. **API Configuration**: Valid API keys must be configured before running
3. **EffiBench-X Backend**: EffiBench-X evaluation service must be running

## Related Documentation

- [Main Project README](../README.md)
- [Operator Development Guide](operators/README.md)
- [PerfAgent Documentation](../perfagent/README.md)
