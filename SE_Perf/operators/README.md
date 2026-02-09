# SE Operators System

SE框架的算子系统提供了模块化的轨迹分析和策略生成功能。通过不同的算子，SE可以分析历史执行轨迹，生成个性化的解决策略。

## Architecture

### Class Hierarchy

```
BaseOperator (抽象基类)
├── TemplateOperator (模板算子基类)
│   ├── PlanOperator              - 多样化策略生成
│   ├── TrajPoolSummaryOperator   - 轨迹池分析总结
│   └── TrajectoryAnalyzerOperator - 轨迹快照分析
│
└── (直接继承 BaseOperator)
    ├── CrossoverOperator          - 轨迹交叉组合
    ├── ReflectionRefineOperator   - 反思精炼突变
    ├── AlternativeStrategyOperator - 正交替代策略
    └── FilterTrajectoriesOperator - 轨迹池过滤
```

### Operator Types

#### 1. **TemplateOperator** (模板算子)
- **功能**: 生成个性化的系统提示模板
- **输出**: `OperatorResult` 列表，包含策略文本（`additional_requirements`）
- **用途**: 为每个实例生成针对性的解决策略提示

#### 2. **Direct BaseOperator subclasses** (直接基类子类)
- **功能**: 直接操作轨迹池或生成附加策略
- **输出**: `OperatorResult` 包含 `additional_requirements` 或直接修改轨迹池

## File Structure

```
SE_Perf/operators/
├── __init__.py                # Unified entry and exports
├── README.md                  # This document
├── base.py                    # Base class definitions (BaseOperator, TemplateOperator, OperatorContext)
├── registry.py                # Operator registration system
├── plan.py                    # PlanOperator - diverse strategy generation
├── crossover.py               # CrossoverOperator - trajectory crossover
├── filter.py                  # FilterTrajectoriesOperator - pool filtering
├── reflection_refine.py       # ReflectionRefineOperator - reflection refinement
├── alternative_strategy.py    # AlternativeStrategyOperator - orthogonal strategy
├── traj_pool_summary.py       # TrajPoolSummaryOperator - pool analysis
└── trajectory_analyzer.py     # TrajectoryAnalyzerOperator - trajectory analysis
```

## Implemented Operators

### 1. **PlanOperator**
- **Type**: TemplateOperator
- **Function**: 生成 K 个多样化的算法策略，用于初始轮次并行探索
- **Key Method**: `run_for_instance()` → 返回 `list[OperatorResult]`（多个策略）
- **Features**:
  - 利用 LLM 一次性生成多个不同方向的策略
  - 支持 LLM 失败时的 fallback 策略
  - 适用于迭代的起始阶段进行广泛探索

### 2. **ReflectionRefineOperator**
- **Type**: BaseOperator (直接继承)
- **Function**: 选择一条现有轨迹，通过反思其执行过程生成改进策略
- **Strategy**: 反馈驱动的受控突变
- **Features**:
  - 基于性能加权选择源轨迹
  - 分析轨迹中的成功/失败模式
  - 生成针对性的精炼优化建议

### 3. **CrossoverOperator**
- **Type**: BaseOperator (直接继承)
- **Function**: 选择两条轨迹，交叉组合两者优势生成混合策略
- **Strategy**: 组合交叉
- **Features**:
  - 要求轨迹池中有效条数 >= 2
  - 分析两条轨迹的互补优势
  - 生成综合两种方法优点的混合策略

### 4. **FilterTrajectoriesOperator**
- **Type**: BaseOperator (直接继承)
- **Function**: 基于聚类分析过滤轨迹池，移除低质量或冗余轨迹
- **Strategy**: 轨迹池精简
- **Features**:
  - 基于聚类的过滤算法
  - 直接修改轨迹池（删除条目）
  - 返回空的 `OperatorResult`（不生成新策略）

### 5. **AlternativeStrategyOperator**
- **Type**: BaseOperator (直接继承)
- **Function**: 基于最近失败的尝试生成截然不同的替代解决策略
- **Strategy**: 正交探索
- **Features**:
  - 分析最近失败的尝试
  - 生成与已有方向正交的解决方案
  - 支持 LLM 生成和默认降级策略

### 6. **TrajPoolSummaryOperator**
- **Type**: TemplateOperator
- **Function**: 分析轨迹池中所有历史尝试，识别常见盲区和风险点
- **Strategy**: 风险感知指导
- **Features**:
  - 跨迭代分析，识别系统性风险
  - 生成简洁的风险感知指导
  - 专注于盲区避免和风险点识别

### 7. **TrajectoryAnalyzerOperator**
- **Type**: TemplateOperator
- **Function**: 分析轨迹快照，提取详细的问题陈述和执行数据，生成策略
- **Strategy**: 基于快照的深度分析
- **Features**:
  - 直接从轨迹文件分析
  - 提取详细的问题陈述和执行统计
  - 生成基于完整轨迹内容的解决策略

## Base Class Capabilities

### BaseOperator 提供的通用功能

- **LLM Integration**: `_call_llm_api()` 统一调用 LLM API
- **Model Management**: `_setup_model()` 自动配置和管理 LLM 模型实例
- **Source Selection**: `_weighted_select_labels()` 性能加权选择 / `_random_select_labels()` 随机选择
- **Unified Selection**: `_select_source_labels()` 统一的源标签选择接口

### TemplateOperator 额外提供

- **Template Generation**: 生成策略文本并封装为 `OperatorResult`
- **Content Formatting**: 统一的策略前缀和格式化

## Operator Registration System

```python
from SE_Perf.operators import register_operator, list_operators, create_operator

# Register operator
register_operator("my_operator", MyOperatorClass)

# List all operators
operators = list_operators()

# Create operator instance
operator = create_operator("my_operator", config)
```

## Data Flow

### Input Data

算子系统使用以下标准化数据（通过 `OperatorContext`）：

1. **问题描述**: 从实例数据中获取
2. **轨迹池**: `TrajPoolManager` 提供的 `InstanceTrajectories`
3. **局部记忆**: `LocalMemoryManager` 提供的任务内学习记忆
4. **全局记忆**: `GlobalMemoryManager` 提供的跨任务检索记忆
5. **配置信息**: `OperatorContext` 中的模型配置和运行参数

### Output Format

```python
@dataclass
class OperatorResult:
    additional_requirements: str   # 策略文本，注入到 PerfAgent 的系统提示中
    trajectory_label: str          # 标识此次运行的标签
    # ... other fields
```

## Strategy Configuration

在 SE_Perf 配置文件中使用算子：

```yaml
strategy:
  iterations:
    - operator: "plan"
      num: 5                                    # PlanOperator 生成 5 个策略
      trajectory_labels: ["sol1", "sol2", ...]
    - operator: "reflection_refine"
      trajectory_label: "sol6"
    - operator: "crossover"
      trajectory_label: "sol7"
    - operator: "filter"                        # FilterOperator 不生成新策略
    - operator: "alternative_strategy"
      trajectory_label: "sol8"
```

## Custom Operator Development

```python
from SE_Perf.operators import TemplateOperator, register_operator

class MyOperator(TemplateOperator):
    def get_name(self) -> str:
        return "my_operator"

    def run_for_instance(self, instance_id, context) -> list[OperatorResult]:
        # 1. Access trajectory pool, memory, problem description via context
        # 2. Analyze data and generate strategy
        # 3. Return OperatorResult list
        ...

# Register
register_operator("my_operator", MyOperator)
```

## Best Practices

1. **Error Handling**: 所有算子都应提供降级策略，LLM 调用失败时返回默认内容
2. **Logging**: 使用统一的日志系统记录处理过程和错误
3. **Data Validation**: 在处理前验证输入数据的完整性
4. **Content Length**: 生成的策略内容应简洁明了，避免过长的提示

---

*算子系统是 SE 框架的核心组件，通过模块化设计实现了灵活的策略生成和轨迹分析能力。*
