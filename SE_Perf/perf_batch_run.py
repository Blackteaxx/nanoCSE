#!/usr/bin/env python3
"""
PerfAgent 批量实例执行脚本

功能：
    扫描指定目录下的所有实例文件，逐实例调用 run_single_instance()，
    并在执行完毕后聚合结果（all_final.json、all_preds.json、token_summary.json、batch_summary.json）。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

# 添加 SE 根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from perf_config import SEPerfRunSEConfig
from perf_run import run_single_instance


# ---------------------------------------------------------------------------
# 公共工具
# ---------------------------------------------------------------------------


def _make_result(
    instance_id: str,
    status: str,
    output_dir: str,
    *,
    error: str | None = None,
    best_metric=None,
    duration_s: float = 0,
) -> dict:
    """构造统一的实例结果 dict。"""
    return {
        "instance_id": instance_id,
        "status": status,
        "output_dir": output_dir,
        "error": error,
        "best_metric": best_metric,
        "duration_s": duration_s,
    }


def _write_json(path: Path | str, data) -> None:
    """写入 JSON 文件。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _count_by_status(results: list[dict]) -> tuple[int, int, int]:
    """统计结果状态，返回 (success, failed, skipped)。"""
    c = Counter(r["status"] for r in results)
    return c["success"], c["error"], c["skipped"]


def _quiet_se_loggers() -> None:
    """将 SE 框架 logger 的控制台级别提升到 WARNING。

    仅影响控制台输出，文件 handler 不受影响。
    """
    try:
        from core.utils.log import set_stream_handler_levels
        set_stream_handler_levels(logging.WARNING)
    except Exception:
        pass


@contextlib.contextmanager
def _capture_instance_output(output_dir: str):
    """将 stdout 重定向到实例目录下的 console.log。

    tqdm 在创建时已持有原始 stdout 的引用，不受此重定向影响。
    stderr 不做重定向，WARNING/ERROR 日志和异常信息仍然显示在终端。
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    console_log_path = Path(output_dir) / "console.log"
    old_stdout = sys.stdout
    try:
        log_f = open(console_log_path, "w", encoding="utf-8")  # noqa: SIM115
        sys.stdout = log_f
        yield
    finally:
        sys.stdout = old_stdout
        try:
            log_f.close()
        except Exception:
            pass


def _filter_resume(
    instances: list[Path],
    batch_output_dir: Path,
    logger: logging.Logger,
) -> tuple[list[Path], list[dict]]:
    """预过滤已完成实例，返回 (pending, skipped_results)。"""
    pending: list[Path] = []
    skipped: list[dict] = []
    for inst_path in instances:
        name = inst_path.stem
        out_dir = str(batch_output_dir / name)
        if (Path(out_dir) / "final.json").exists():
            skipped.append(_make_result(name, "skipped", out_dir))
            logger.info(f"跳过已完成实例: {name}")
        else:
            pending.append(inst_path)
    return pending, skipped


def _setup_batch_logger(batch_output_dir: Path) -> logging.Logger:
    """设置批量执行专用 logger，输出到 batch.log。"""
    logger = logging.getLogger("batch_run")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(batch_output_dir / "batch.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(logging.WARNING)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Phase 1: 实例发现
# ---------------------------------------------------------------------------


def discover_instances(instance_dir: str) -> list[Path]:
    """扫描 instance_dir 下所有 .json 文件，按文件名排序返回。"""
    inst_dir = Path(instance_dir)
    if not inst_dir.is_dir():
        msg = f"实例目录不存在: {inst_dir}"
        raise FileNotFoundError(msg)
    return sorted(inst_dir.glob("*.json"), key=lambda p: p.name)


# ---------------------------------------------------------------------------
# Phase 2: 批量执行
# ---------------------------------------------------------------------------


def _worker_run_instance(
    config_path: str,
    instance_path: str,
    output_dir: str,
    mode: str,
) -> dict:
    """进程池 worker：在独立进程中执行单实例。

    不接收 se_cfg 以避免跨进程 pickle 问题，每个 worker 独立加载配置。
    """
    _quiet_se_loggers()
    inst_start = time.time()
    instance_name = Path(instance_path).stem
    try:
        with _capture_instance_output(output_dir):
            _quiet_se_loggers()
            res = run_single_instance(
                config_path=config_path,
                instance_path=instance_path,
                output_dir=output_dir,
                mode=mode,
                se_cfg=None,
            )
    except Exception as e:
        res = _make_result(instance_name, "error", output_dir, error=str(e))
    res["duration_s"] = round(time.time() - inst_start, 1)
    return res


def _run_instances(
    instances: list[Path],
    config_path: str,
    batch_output_dir: Path,
    mode: str,
    resume: bool,
    max_workers: int,
    logger: logging.Logger,
) -> list[dict]:
    """使用进程池执行所有实例。

    通过 ProcessPoolExecutor 实现进程级隔离，每个实例独立加载配置。
    max_workers=1 时等价于串行执行。
    """
    # 预过滤 resume 跳过的实例
    if resume:
        pending_instances, skipped_results = _filter_resume(
            instances, batch_output_dir, logger,
        )
    else:
        pending_instances, skipped_results = instances, []

    all_results: list[dict] = list(skipped_results)
    skipped_count = len(skipped_results)
    success_count = 0
    failed_count = 0

    # 构建待提交任务列表
    pending_tasks = [
        (str(p), str(batch_output_dir / p.stem), p.stem)
        for p in pending_instances
    ]

    if not pending_tasks:
        print(f"所有 {skipped_count} 个实例均已完成，无需执行")
        return all_results

    print(
        f"执行: {len(pending_tasks)} 个待执行, "
        f"{skipped_count} 个已跳过, workers={max_workers}"
    )

    pbar = tqdm(
        total=len(pending_tasks),
        desc="Batch Run",
        unit="inst",
        ncols=120,
        file=sys.stdout,
    )
    pbar.set_postfix_str(f"✓{success_count} ✗{failed_count} ⊘{skipped_count}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {}
        for inst_path_str, inst_output_dir, inst_name in pending_tasks:
            future = executor.submit(
                _worker_run_instance,
                config_path=config_path,
                instance_path=inst_path_str,
                output_dir=inst_output_dir,
                mode=mode,
            )
            future_to_name[future] = inst_name

        for future in as_completed(future_to_name):
            inst_name = future_to_name[future]
            try:
                res = future.result()
            except Exception as e:
                res = _make_result(
                    inst_name, "error",
                    str(batch_output_dir / inst_name), error=str(e),
                )

            if res["status"] == "success":
                success_count += 1
            elif res["status"] != "skipped":
                failed_count += 1

            all_results.append(res)
            pbar.update(1)
            pbar.set_postfix_str(
                f"✓{success_count} ✗{failed_count} ⊘{skipped_count} | done: {inst_name}"
            )
            logger.info(
                f"实例完成: {inst_name} | 状态: {res['status']} | "
                f"耗时: {res.get('duration_s', 0)}s | "
                f"best_metric: {res.get('best_metric')}"
            )

    pbar.close()
    return all_results


# ---------------------------------------------------------------------------
# Phase 3: 后处理
# ---------------------------------------------------------------------------


def _aggregate_instance_json(
    batch_output_dir: Path,
    results: list[dict],
    src_file: str,
    out_file: str,
    merge_fn,
) -> Path | None:
    """通用实例 JSON 聚合。

    遍历所有成功/跳过的实例，读取 src_file，用 merge_fn(accumulated, data) 合并，
    最终写入 out_file。
    """
    accumulated: dict = {}
    for res in results:
        if res["status"] not in ("success", "skipped"):
            continue
        src_path = Path(res["output_dir"]) / src_file
        if not src_path.exists():
            continue
        try:
            with open(src_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                merge_fn(accumulated, data)
        except Exception:
            continue

    if not accumulated:
        return None
    out_path = batch_output_dir / out_file
    _write_json(out_path, accumulated)
    return out_path


def _merge_final(acc: dict, data: dict) -> None:
    """final.json 合并策略：直接 update。"""
    acc.update(data)


def _merge_preds(acc: dict, data: dict) -> None:
    """preds.json 合并策略：按 instance_id extend list。"""
    for instance_id, entries in data.items():
        acc.setdefault(instance_id, [])
        if isinstance(entries, list):
            acc[instance_id].extend(entries)
        else:
            acc[instance_id].append(entries)


def _aggregate_token_summary(
    batch_output_dir: Path, results: list[dict],
) -> Path | None:
    """遍历所有实例的 token_usage.jsonl，生成 token_summary.json。"""
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    by_instance: dict[str, dict[str, int]] = {}
    by_context: dict[str, dict[str, int]] = {}
    instance_count = 0

    for res in results:
        if res["status"] not in ("success", "skipped"):
            continue
        token_log = Path(res["output_dir"]) / "token_usage.jsonl"
        if not token_log.exists():
            continue

        inst = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        instance_count += 1

        try:
            with open(token_log, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        pt = int(rec.get("prompt_tokens") or 0)
                        ct = int(rec.get("completion_tokens") or 0)
                        tt = int(rec.get("total_tokens") or (pt + ct))
                        ctx = str(rec.get("context") or "unknown")

                        for d in (inst, total):
                            d["prompt_tokens"] += pt
                            d["completion_tokens"] += ct
                            d["total_tokens"] += tt

                        ctx_agg = by_context.setdefault(
                            ctx,
                            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        )
                        ctx_agg["prompt_tokens"] += pt
                        ctx_agg["completion_tokens"] += ct
                        ctx_agg["total_tokens"] += tt
                    except Exception:
                        continue
        except Exception:
            continue

        by_instance[res["instance_id"]] = inst

    out_path = batch_output_dir / "token_summary.json"
    _write_json(out_path, {
        "total": total,
        "by_instance": by_instance,
        "by_context": by_context,
        "instance_count": instance_count,
        "avg_tokens_per_instance": (
            total["total_tokens"] // instance_count if instance_count > 0 else 0
        ),
    })
    return out_path


def _generate_batch_summary(
    batch_output_dir: Path,
    results: list[dict],
    start_time: str,
    end_time: str,
    token_summary_path: Path | None,
) -> Path:
    """生成 batch_summary.json。"""
    success, failed, skipped = _count_by_status(results)

    token_summary = {}
    if token_summary_path and token_summary_path.exists():
        try:
            with open(token_summary_path, encoding="utf-8") as f:
                token_summary = json.load(f)
        except Exception:
            pass

    out_path = batch_output_dir / "batch_summary.json"
    _write_json(out_path, {
        "start_time": start_time,
        "end_time": end_time,
        "total_instances": len(results),
        "success": success,
        "failed": failed,
        "skipped": skipped,
        "results": results,
        "token_summary": token_summary,
    })
    return out_path


# ---------------------------------------------------------------------------
# 编排入口
# ---------------------------------------------------------------------------


def run_batch(
    config_path: str,
    instance_dir: str,
    mode: str = "execute",
    resume: bool = False,
    max_workers: int = 1,
    limit: int | None = None,
):
    """批量执行主入口。

    Args:
        config_path: SE 配置文件路径。
        instance_dir: 实例 JSON 文件所在目录。
        mode: 运行模式。
        resume: 是否断点续跑。
        max_workers: 并发进程数（1 = 串行）。
        limit: 最多执行前 N 个实例，None 表示全部执行。
    """
    start_time_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    start_ts = time.time()

    # 加载配置
    with open(config_path, encoding="utf-8") as f:
        se_raw = yaml.safe_load(f) or {}
    se_cfg = SEPerfRunSEConfig.from_dict(se_raw)

    # 实例发现
    instances = discover_instances(instance_dir)
    if not instances:
        print(f"在 {instance_dir} 下未找到任何 .json 实例文件")
        return

    if limit is not None and limit > 0:
        instances = instances[:limit]
        print(f"发现实例文件，限制执行前 {limit} 个（共 {len(instances)} 个）")
    else:
        print(f"发现 {len(instances)} 个实例文件")

    # 创建批量输出根目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = se_cfg.output_dir.replace("{timestamp}", timestamp)
    batch_output_dir = Path(base_output)
    batch_output_dir.mkdir(parents=True, exist_ok=True)

    logger = _setup_batch_logger(batch_output_dir)
    logger.info(
        f"批量执行开始: config={config_path}, instance_dir={instance_dir}, "
        f"mode={mode}, max_workers={max_workers}"
    )
    logger.info(f"实例总数: {len(instances)}, 输出目录: {batch_output_dir}")

    # 批量执行
    all_results = _run_instances(
        instances, config_path, batch_output_dir, mode, resume, max_workers, logger,
    )

    # 后处理
    print("\n=== 后处理阶段 ===")
    end_time_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    for label, path in [
        ("all_final.json", _aggregate_instance_json(
            batch_output_dir, all_results, "final.json", "all_final.json", _merge_final,
        )),
        ("all_preds.json", _aggregate_instance_json(
            batch_output_dir, all_results, "preds.json", "all_preds.json", _merge_preds,
        )),
        ("token_summary.json", _aggregate_token_summary(batch_output_dir, all_results)),
    ]:
        if path:
            print(f"  已生成 {label}: {path}")
            logger.info(f"已生成 {label}: {path}")

    summary_path = _generate_batch_summary(
        batch_output_dir, all_results, start_time_str, end_time_str,
        batch_output_dir / "token_summary.json",
    )
    print(f"  已生成 batch_summary.json: {summary_path}")
    logger.info(f"已生成 batch_summary.json: {summary_path}")

    # 打印最终摘要
    success, failed, skipped = _count_by_status(all_results)
    total_duration = round(time.time() - start_ts, 1)
    print(f"\n=== 批量执行完成 ===")
    print(f"  总耗时: {total_duration}s")
    print(f"  实例总数: {len(all_results)}")
    print(f"  成功: {success}, 失败: {failed}, 跳过: {skipped}")
    print(f"  输出目录: {batch_output_dir}")
    logger.info(
        f"批量执行完成: 总耗时={total_duration}s, "
        f"成功={success}, 失败={failed}, 跳过={skipped}"
    )


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="SE 框架 PerfAgent 批量实例执行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python SE_Perf/perf_batch_run.py \\
      --config configs/Plan-Weighted-Local-Global-30.yaml \\
      --instance-dir instances/ \\
      --mode execute \\
      --resume
        """,
    )
    parser.add_argument("--config", default="configs/Plan-Weighted-Local-Global-30.yaml", help="SE 配置文件路径")
    parser.add_argument("--instance-dir", required=True, help="实例 JSON 文件所在目录")
    parser.add_argument("--mode", choices=["demo", "execute"], default="execute", help="运行模式")
    parser.add_argument("--max-workers", type=int, default=1, help="并发数（默认 1 = 串行）")
    parser.add_argument("--resume", action="store_true", help="断点续跑：跳过已有 final.json 的实例")
    parser.add_argument("--limit", type=int, default=None, help="只执行前 N 个实例（默认全部执行）")
    args = parser.parse_args()

    print("=== CSE 批量实例执行 ===")

    run_batch(
        config_path=args.config,
        instance_dir=args.instance_dir,
        mode=args.mode,
        resume=args.resume,
        max_workers=args.max_workers,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
