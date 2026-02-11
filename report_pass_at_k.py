#!/usr/bin/env python3
"""
Report pass@k metrics from all_preds.json.

For each k in k_list, considers the first k iterations per problem,
and computes pass@k = fraction of problems where at least one iteration passed (metric > 0).
"""

import argparse
import json
import os


def compute_pass_at_k(data: dict, k_list: list[int]) -> dict:
    """
    Args:
        data: {problem_id: [{iteration, metric, ...}, ...], ...}
        k_list: list of k values to evaluate

    Returns:
        dict with pass@k results
    """
    # Pre-sort each problem's predictions by iteration
    sorted_data = {}
    for pid, preds in data.items():
        sorted_data[pid] = sorted(preds, key=lambda x: x["iteration"])

    num_problems = len(sorted_data)
    max_iter = max(len(preds) for preds in sorted_data.values())

    results = {}
    detail = {}

    for k in k_list:
        if k > max_iter:
            print(f"Warning: k={k} exceeds max iterations ({max_iter}), skipping.")
            continue

        passed_count = 0
        per_problem = {}

        for pid, preds in sorted_data.items():
            first_k = preds[:k]
            any_passed = any(p["metric"] > 0 for p in first_k)
            per_problem[pid] = 1.0 if any_passed else 0.0
            if any_passed:
                passed_count += 1

        pass_rate = passed_count / num_problems
        results[f"pass@{k}"] = pass_rate
        detail[f"pass@{k}"] = per_problem

    results["detail"] = detail
    results["num_problems"] = num_problems
    return results


def main():
    parser = argparse.ArgumentParser(description="Compute pass@k from all_preds.json")
    parser.add_argument(
        "--input",
        type=str,
        default="RL/nanoCSE/trajectories_perf/livecodebench_batch_20260210_025024/all_preds.json",
        help="Path to all_preds.json",
    )
    parser.add_argument(
        "--k_list",
        type=int,
        nargs="+",
        default=[1, 2, 3, 5, 10, 15, 20, 25, 30],
        help="List of k values for pass@k",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON results (default: same dir as input)",
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input} ...")
    with open(args.input, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} problems.")

    # Compute pass@k
    results = compute_pass_at_k(data, sorted(args.k_list))

    # Print formatted table
    print()
    print("=" * 40)
    print(f"  pass@k Report  ({results['num_problems']} problems)")
    print("=" * 40)
    for k in sorted(args.k_list):
        key = f"pass@{k}"
        if key in results:
            val = results[key]
            passed = int(val * results["num_problems"])
            print(f"  {key:>10s} = {val:.4f}  ({passed}/{results['num_problems']})")
    print("=" * 40)

    # Save JSON results
    output_path = args.output
    if output_path is None:
        input_dir = os.path.dirname(args.input)
        output_path = os.path.join(input_dir, "pass_at_k_report.json")

    # Remove detail for the summary file (keep it lightweight)
    summary = {k: v for k, v in results.items() if k != "detail"}
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Also save detailed per-problem results
    detail_path = output_path.replace(".json", "_detail.json")
    with open(detail_path, "w") as f:
        json.dump(results["detail"], f, indent=2)
    print(f"Detail results saved to {detail_path}")


if __name__ == "__main__":
    main()
