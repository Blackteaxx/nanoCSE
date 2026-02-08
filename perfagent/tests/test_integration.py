"""
集成测试的 pytest 测试用例
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from perfagent.agent import EffiBenchXInstance, PerfAgent
from perfagent.config import LoggingConfig, ModelConfig, PerfAgentConfig  # noqa: F401


class TestIntegration:
    """集成测试"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        return Path(tempfile.mkdtemp())

    @pytest.fixture
    def test_instance_file(self, temp_dir):
        """创建测试实例文件"""
        instance_file = temp_dir / "test_instance.py"
        content = """
def inefficient_sum(n):
    # 这是一个低效的求和函数
    result = 0
    for i in range(n):
        for j in range(i):
            result += 1
    return result

def main():
    import time
    start_time = time.time()
    result = inefficient_sum(1000)
    end_time = time.time()
    print(f"结果: {result}")
    print(f"执行时间: {end_time - start_time:.4f}秒")

if __name__ == "__main__":
    main()
"""
        instance_file.write_text(content)
        return str(instance_file)

    @pytest.fixture
    def config(self, temp_dir):
        """创建测试配置"""
        from conftest import TEST_OPTIMIZATION_TEMPLATE, TEST_SYSTEM_TEMPLATE

        from perfagent.config import PromptConfig

        cfg = PerfAgentConfig(
            max_iterations=2,  # 减少迭代次数以加快测试
            model=ModelConfig(name="gpt-4", temperature=0.1),
            logging=LoggingConfig(trajectory_dir=temp_dir / "trajectories", log_dir=temp_dir / "logs"),
            prompts=PromptConfig(
                system_template=TEST_SYSTEM_TEMPLATE,
                optimization_template=TEST_OPTIMIZATION_TEMPLATE,
            ),
            task_config={"language": "python3"},
        )
        return cfg

    def test_full_optimization_flow(self, config, test_instance_file, temp_dir, fake_llm):
        """测试完整的优化流程"""
        # 创建 PerfAgent 实例
        agent = PerfAgent(config)
        # 注入假的 LLM 客户端，避免真实调用
        agent.llm_client = fake_llm
        agent.config.task_config["language"] = "python3"

        # 创建实例数据（dataclass）
        inst = EffiBenchXInstance(
            id="test_instance",
            title="",
            title_slug="",
            description="",
            description_md="",
            source="",
            url="",
            type="",
            starter_code=open(test_instance_file).read(),
            language="python3",
        )

        # 运行优化
        result = agent.run(inst)

        # 验证结果
        assert "success" in result
        assert "metric" in result
        assert "trajectory_file" in result

        # 验证轨迹文件是否创建
        trajectory_file = result["trajectory_file"]
        assert os.path.exists(trajectory_file)

        # 验证轨迹文件内容
        with open(trajectory_file, encoding="utf-8") as f:
            trajectory_data = json.load(f)

        # 兼容当前实现的元数据键名（info 或 metadata）
        assert ("metadata" in trajectory_data) or ("info" in trajectory_data)
        assert "steps" in trajectory_data
        md = trajectory_data.get("metadata") or trajectory_data.get("info")
        assert md["instance_id"] == "test_instance"

    def test_batch_optimization(self, config, temp_dir):
        """测试批量优化"""
        # 创建多个测试实例
        instances_dir = temp_dir / "instances"
        instances_dir.mkdir()

        # 创建第一个实例
        instance1 = instances_dir / "instance1.py"
        instance1.write_text("""
def slow_function():
    result = 0
    for i in range(10000):
        result += i * i
    return result

if __name__ == "__main__":
    print(slow_function())
""")

        # 创建第二个实例
        instance2 = instances_dir / "instance2.py"
        instance2.write_text("""
def another_slow_function():
    data = []
    for i in range(5000):
        data.append(i)
        data.sort()  # 每次都排序，很低效
    return len(data)

if __name__ == "__main__":
    print(another_slow_function())
""")

        # 创建批量配置文件
        batch_config = temp_dir / "batch_config.yaml"
        batch_config.write_text(f"""
logging:
  trajectory_dir: "{temp_dir / "trajectories"}"
max_iterations: 2
""")

        # 运行批量优化（模拟命令行调用）
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ["run.py", "--config", str(batch_config), "--batch"]
            config_loaded = PerfAgentConfig.from_yaml(batch_config)
            assert config_loaded.max_iterations == 2
            assert config_loaded.logging.trajectory_dir == Path(str(temp_dir / "trajectories"))

        finally:
            sys.argv = original_argv

    def test_error_handling(self, config, temp_dir, fake_llm):
        """测试错误处理"""
        # 创建一个有语法错误的文件
        invalid_file = temp_dir / "invalid.py"
        invalid_file.write_text("def broken_function(\n    # 缺少闭合括号")

        agent = PerfAgent(config)
        agent.llm_client = fake_llm
        agent.config.task_config["language"] = "python3"

        # 创建实例数据（dataclass）
        inst = EffiBenchXInstance(
            id="invalid_instance",
            title="",
            title_slug="",
            description="",
            description_md="",
            source="",
            url="",
            type="",
            starter_code=invalid_file.read_text(),
            language="python3",
        )

        # 应该能够处理错误而不崩溃
        result = agent.run(inst)

        # 验证错误被正确处理
        assert "success" in result

    def test_performance_improvement_detection(self, config):
        """测试性能改进检测（通过 EffiBenchRunner）"""
        from perfagent.tasks.effibench import EffiBenchRunner

        runner = EffiBenchRunner(task_config=config.task_config)
        inst = EffiBenchXInstance(
            id="test_instance", title="", title_slug="", description="", description_md="", source="", url="", type=""
        )

        # 没有 evaluator 和 test_cases 时应返回默认结果
        code = "def efficient_sum(n):\n    return n * (n - 1) // 2\n"
        metric, artifacts = runner.evaluate(code, inst, config)
        assert isinstance(metric, float)
        assert "problem_description" not in artifacts  # problem_description 不再放入 artifacts

    def test_trajectory_completeness(self, config, test_instance_file, temp_dir, fake_llm):
        """测试轨迹完整性"""
        agent = PerfAgent(config)
        agent.llm_client = fake_llm
        agent.config.task_config["language"] = "python3"

        inst = EffiBenchXInstance(
            id="traj_instance",
            title="",
            title_slug="",
            description="",
            description_md="",
            source="",
            url="",
            type="",
            starter_code=open(test_instance_file).read(),
            language="python3",
        )

        result = agent.run(inst)
        trajectory_file = result.get("trajectory_file")
        assert trajectory_file and os.path.exists(trajectory_file)

        with open(trajectory_file, encoding="utf-8") as f:
            traj = json.load(f)

        assert ("metadata" in traj) or ("info" in traj)
        md = traj.get("metadata") or traj.get("info")
        # 轨迹元数据不再由 agent 硬编码设置 language/optimization_target
        assert md.get("instance_id") == "traj_instance"

    def test_multiple_iterations(self, config, temp_dir, fake_llm):
        """测试多次迭代流程"""
        agent = PerfAgent(config)
        agent.llm_client = fake_llm
        agent.config.task_config["language"] = "python3"

        # 使用文件内容作为 starter_code
        test_file = temp_dir / "multi_iter.py"
        test_file.write_text("""
def slow_loop():
    s = 0
    for i in range(100000):
        s += i
    return s
""")

        inst = EffiBenchXInstance(
            id="multi_iter_test",
            title="",
            title_slug="",
            description="",
            description_md="",
            source="",
            url="",
            type="",
            starter_code=test_file.read_text(),
            language="python3",
        )

        result = agent.run(inst)
        assert "success" in result
        assert "metric" in result
