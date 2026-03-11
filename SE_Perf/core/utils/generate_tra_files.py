#!/usr/bin/env python3

"""
SE Framework - .tra文件生成工具

独立的命令行工具，用于为现有的轨迹目录生成.tra文件。
可以处理单个iteration目录或整个workspace目录。
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from SE.core.utils.se_logger import get_se_logger, setup_se_logging
from SE.core.utils.trajectory_processor import TrajectoryProcessor


def main():
    """主函数：.tra文件生成命令行工具"""

    parser = argparse.ArgumentParser(
        description="SE框架 - .tra文件生成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理整个workspace目录
  python SE/core/utils/generate_tra_files.py SE/trajectories/Demo_Structure
  
  # 处理指定的iteration目录
  python SE/core/utils/generate_tra_files.py SE/trajectories/Demo_Structure/iteration_1 --single-iteration
  
  # 只处理特定的iterations
  python SE/core/utils/generate_tra_files.py SE/trajectories/Demo_Structure --iterations 1 2
  
  # 强制重新生成（覆盖已存在的.tra文件）
  python SE/core/utils/generate_tra_files.py SE/trajectories/Demo_Structure --force
        """,
    )

    parser.add_argument("target_dir", help="目标目录路径（workspace目录或iteration目录）")
    parser.add_argument(
        "--single-iteration", action="store_true", help="指定target_dir是单个iteration目录（如iteration_1/）"
    )
    parser.add_argument(
        "--iterations", type=int, nargs="+", help="指定要处理的iteration编号（仅在处理workspace时有效）"
    )
    parser.add_argument("--force", action="store_true", help="强制重新生成，覆盖已存在的.tra文件")
    parser.add_argument("--dry-run", action="store_true", help="只显示将要处理的文件，不实际生成")
    parser.add_argument("--extract-problems", action="store_true", help="同时提取problem描述文件(.problem)")
    parser.add_argument("--problems-only", action="store_true", help="只提取problem文件，不生成.tra文件")

    args = parser.parse_args()

    # 验证目标目录
    target_path = Path(args.target_dir)
    if not target_path.exists():
        print(f"❌ 错误: 目标目录不存在: {target_path}")
        return 1

    if not target_path.is_dir():
        print(f"❌ 错误: 目标路径不是目录: {target_path}")
        return 1

    # 设置日志系统
    if args.single_iteration:
        log_dir = target_path.parent
    else:
        log_dir = target_path

    log_file, _handler_id = setup_se_logging(log_dir)
    logger = get_se_logger("generate_tra_files", emoji="🎬")

    print("=== SE框架 - .tra文件生成工具 ===")
    print(f"目标目录: {target_path}")
    print(f"处理模式: {'单iteration' if args.single_iteration else 'workspace'}")
    print(f"日志文件: {log_file}")

    if args.dry_run:
        print("🔍 DRY RUN模式 - 只分析，不生成文件")
    elif args.problems_only:
        print("📝 PROBLEMS模式 - 只提取problem文件")
    elif args.extract_problems:
        print("🎯 增强模式 - 生成.tra和.problem文件")

    try:
        processor = TrajectoryProcessor()

        if args.single_iteration:
            # 处理单个iteration目录
            logger.info(f"开始处理单个iteration目录: {target_path}")

            if args.dry_run:
                # Dry run: 只显示会处理的文件
                _show_traj_files(target_path)
                return 0

            if args.problems_only:
                # 只提取problem文件
                problem_result = processor.process_problems_in_iteration(target_path)
                if problem_result and problem_result.get("total_problems", 0) > 0:
                    print("✅ Problem提取完成!")
                    print(f"  - 提取.problem文件: {problem_result['total_problems']}")
                    if problem_result["failed_extractions"]:
                        print(f"  - 失败提取数: {len(problem_result['failed_extractions'])}")
                else:
                    print("⚠️ 未提取任何problem文件")
                    return 1
            else:
                # 生成.tra文件
                result = processor.process_iteration_directory(target_path)

                if result and result.get("total_tra_files", 0) > 0:
                    print("✅ .tra文件处理完成!")
                    print(f"  - 创建.tra文件: {result['total_tra_files']}")
                    print(f"  - 总token数: ~{result['total_tokens']}")
                    print(f"  - 处理实例数: {len(result['processed_instances'])}")

                    if result["failed_instances"]:
                        print(f"  - 失败实例数: {len(result['failed_instances'])}")

                    # 如果需要，同时提取problem文件
                    if args.extract_problems:
                        print("\n📝 开始提取problem文件...")
                        problem_result = processor.process_problems_in_iteration(target_path)
                        if problem_result and problem_result.get("total_problems", 0) > 0:
                            print("✅ Problem提取完成!")
                            print(f"  - 提取.problem文件: {problem_result['total_problems']}")
                            if problem_result["failed_extractions"]:
                                print(f"  - 失败提取数: {len(problem_result['failed_extractions'])}")
                        else:
                            print("⚠️ 未提取任何problem文件")

                else:
                    print("⚠️ 未生成任何.tra文件")
                    return 1

        else:
            # 处理整个workspace目录
            logger.info(f"开始处理workspace目录: {target_path}")

            if args.dry_run:
                # Dry run: 显示所有会处理的iteration和文件
                _show_workspace_overview(target_path, args.iterations)
                return 0

            result = processor.process_workspace_directory(target_path, args.iterations)

            if result and result.get("total_tra_files", 0) > 0:
                print("✅ 处理完成!")
                print(f"  - 处理iterations: {len(result['iterations_processed'])}")
                print(f"  - 创建.tra文件: {result['total_tra_files']}")
                print(f"  - 总token数: ~{result['total_tokens']}")

                if result["processing_errors"]:
                    print(f"  - 处理错误数: {len(result['processing_errors'])}")
                    for error in result["processing_errors"]:
                        print(f"    * iteration_{error['iteration_number']}: {error['error']}")
            else:
                print("⚠️ 未生成任何.tra文件")
                return 1

        logger.info(".tra文件生成完成")
        return 0

    except Exception as e:
        logger.error(f"处理过程中出错: {e}", exc_info=True)
        print(f"❌ 错误: {e}")
        return 1


def _show_traj_files(iteration_dir: Path):
    """显示iteration目录中的.traj文件（dry run模式）"""
    print(f"\n🔍 在 {iteration_dir} 中找到的.traj文件:")

    instance_count = 0
    traj_count = 0

    for instance_dir in iteration_dir.iterdir():
        if not instance_dir.is_dir() or instance_dir.name.startswith("."):
            continue

        traj_files = list(instance_dir.glob("*.traj"))
        if traj_files:
            instance_count += 1
            print(f"  📁 {instance_dir.name}/")
            for traj_file in traj_files:
                tra_file = instance_dir / (traj_file.stem + ".tra")
                exists = "✅" if tra_file.exists() else "➕"
                print(f"    {exists} {traj_file.name} -> {tra_file.name}")
                traj_count += 1

    if traj_count == 0:
        print("    (未找到任何.traj文件)")
    else:
        print(f"\n📊 统计: {instance_count} 个实例, {traj_count} 个.traj文件")


def _show_workspace_overview(workspace_dir: Path, target_iterations=None):
    """显示workspace目录的概览（dry run模式）"""
    print(f"\n🔍 workspace目录概览: {workspace_dir}")

    import re

    iteration_pattern = re.compile(r"^iteration_(\d+)$")
    iterations = []

    for item in workspace_dir.iterdir():
        if item.is_dir():
            match = iteration_pattern.match(item.name)
            if match:
                iteration_num = int(match.group(1))
                if target_iterations is None or iteration_num in target_iterations:
                    iterations.append((iteration_num, item))

    iterations.sort(key=lambda x: x[0])

    if not iterations:
        print("    (未找到任何iteration目录)")
        return

    total_instances = 0
    total_traj_files = 0

    for iteration_num, iteration_dir in iterations:
        print(f"\n  📂 iteration_{iteration_num}/")

        instance_count = 0
        traj_count = 0

        for instance_dir in iteration_dir.iterdir():
            if not instance_dir.is_dir() or instance_dir.name.startswith("."):
                continue

            traj_files = list(instance_dir.glob("*.traj"))
            if traj_files:
                instance_count += 1
                traj_count += len(traj_files)

        print(f"    📊 {instance_count} 个实例, {traj_count} 个.traj文件")
        total_instances += instance_count
        total_traj_files += traj_count

    print(f"\n📊 总计: {len(iterations)} 个iteration, {total_instances} 个实例, {total_traj_files} 个.traj文件")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
