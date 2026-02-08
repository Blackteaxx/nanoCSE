"""
PerfAgent 的 pytest 测试用例
"""

import tempfile
from pathlib import Path

import pytest

from perfagent.agent import EffiBenchXInstance, PerfAgent
from perfagent.config import PerfAgentConfig
from perfagent.tasks.effibench import EffiBenchRunner, EffiBenchTaskConfig


class TestPerfAgent:
    """测试 PerfAgent"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        return Path(tempfile.mkdtemp())

    @pytest.fixture
    def config(self, temp_dir):
        """创建测试配置"""
        cfg = PerfAgentConfig(
            max_iterations=3,
            task_config={"language": "python3"},
        )
        cfg.logging.trajectory_dir = temp_dir
        cfg.logging.log_dir = temp_dir
        return cfg

    @pytest.fixture
    def agent(self, config):
        """创建 PerfAgent 实例"""
        return PerfAgent(config)

    @pytest.fixture
    def runner(self, config):
        """创建 EffiBenchRunner 实例"""
        return EffiBenchRunner(task_config=config.task_config)

    def test_yaml_config(self, temp_dir):
        """测试 YAML 配置加载"""
        config_file = temp_dir / "test_config.yaml"
        config_content = """
model:
  name: "test-model"
max_iterations: 5
logging:
  trajectory_dir: "/tmp/test"
"""
        config_file.write_text(config_content)

        config = PerfAgentConfig.from_yaml(str(config_file))
        assert config.model.name == "test-model"
        assert config.max_iterations == 5
        assert config.logging.trajectory_dir == Path("/tmp/test")

    def test_get_initial_solution(self, runner):
        """测试提取初始代码（返回占位符）"""
        inst = EffiBenchXInstance(
            id="inst-1",
            title="",
            title_slug="",
            description="",
            description_md="",
            source="",
            url="",
            type="",
            starter_code="def slow_function():\n    pass\n",
        )

        code = runner.get_initial_solution(inst, None)
        assert isinstance(code, str)
        assert "# Start your code here" in code

    def test_task_config_language(self, agent):
        """测试 task_config 语言配置"""
        assert agent.config.task_config.get("language") == "python3"

        agent.config.task_config["language"] = "java"
        assert agent.config.task_config["language"] == "java"

        agent.config.task_config["language"] = "cpp"
        assert agent.config.task_config["language"] == "cpp"

    def test_evaluate_performance(self, runner, temp_dir):
        """测试性能评估（通过 EffiBenchRunner）"""
        inst = EffiBenchXInstance(
            id="test_instance", title="", title_slug="", description="", description_md="", source="", url="", type=""
        )
        code = """
import time

def test_function():
    time.sleep(0.01)
    return sum(range(1000))

if __name__ == "__main__":
    start_time = time.time()
    result = test_function()
    end_time = time.time()
    print(f"执行时间: {end_time - start_time:.4f}秒")
    print(f"结果: {result}")
"""
        # 没有 evaluator 和 test_cases，应返回默认失败结构
        metric, artifacts = runner.evaluate(code, inst, None)
        assert isinstance(metric, float)
        assert "problem_description" not in artifacts  # problem_description 不再放入 artifacts

    def test_extract_diff_from_response(self, runner):
        """测试从响应中提取 diff（通过 EffiBenchRunner）"""
        response_with_search_replace = (
            "这里是一些解释文本。\n\n"
            "<<<<<<< SEARCH\n"
            "line2\nline3\n"
            "=======\n"
            "LINE2\nLINE3\n"
            ">>>>>>> REPLACE\n\n"
            "这是一段额外的说明。"
        )
        diff = EffiBenchRunner._extract_diff_from_response(response_with_search_replace)
        assert "<<<<<<< SEARCH" in diff
        assert ">>>>>>> REPLACE" in diff
        assert "line2" in diff and "LINE3" in diff

    def test_extract_diff_no_diff_block(self, runner):
        """测试没有 diff 块的响应"""
        response_without_diff = "这是一个没有 diff 的响应。"
        diff = EffiBenchRunner._extract_diff_from_response(response_without_diff)
        assert diff == ""

    def test_extract_diff_multiple_blocks(self, runner):
        """测试多个 SEARCH/REPLACE 区块的响应提取"""
        response_with_multiple_blocks = (
            "前文说明\n\n"
            "<<<<<<< SEARCH\n"
            "oldA\n"
            "=======\n"
            "newA\n"
            ">>>>>>> REPLACE\n\n"
            "一些说明\n\n"
            "<<<<<<< SEARCH\n"
            "oldB\n"
            "=======\n"
            "newB\n"
            ">>>>>>> REPLACE\n"
            "后文文本\n"
        )
        diff = EffiBenchRunner._extract_diff_from_response(response_with_multiple_blocks)
        assert diff.count("<<<<<<< SEARCH") == 2
        assert diff.count(">>>>>>> REPLACE") == 2
        assert "oldA" in diff and "newB" in diff

    def test_agent_initialization(self, config):
        """测试 Agent 初始化"""
        agent = PerfAgent(config)
        assert agent.config == config

    def test_config_validation(self):
        """测试配置验证"""
        config = PerfAgentConfig(max_iterations=5)
        assert config.max_iterations == 5

        default_config = PerfAgentConfig()
        assert default_config.max_iterations == 10
        assert default_config.model.name == "gpt-4"

    def test_task_config_defaults(self, agent):
        """测试 task_config 默认值"""
        assert agent.config.task_config.get("language") == "python3"
        assert agent.config.task_config.get("target") is None

    def test_effibench_task_config_from_dict(self):
        """测试 EffiBenchTaskConfig 从字典创建"""
        tc = EffiBenchTaskConfig.from_dict({
            "target": "memory",
            "language": "cpp",
            "num_runs": 5,
            "unknown_key": "ignored",
        })
        assert tc.target == "memory"
        assert tc.language == "cpp"
        assert tc.num_runs == 5
        # 未知键应被忽略
        assert not hasattr(tc, "unknown_key")

    def test_effibench_task_config_defaults(self):
        """测试 EffiBenchTaskConfig 默认值"""
        tc = EffiBenchTaskConfig()
        assert tc.target == "runtime"
        assert tc.code_generation_mode == "diff"
        assert tc.language == "python3"
        assert tc.time_limit == 20
        assert tc.num_runs == 10
