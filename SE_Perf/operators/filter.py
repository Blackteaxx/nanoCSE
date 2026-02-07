#!/usr/bin/env python3
"""
Filter Trajectories Operator

根据指定的策略（如多样性、性能）对轨迹池中的一部分轨迹进行过滤、排序和选择。
此算子直接修改 traj_pool_manager，不触发 PerfAgent 调用。
"""

import math
from typing import Any

from core.utils.traj_pool_manager import TrajPoolManager
from perf_config import StepConfig

from operators.base import BaseOperator, OperatorResult


class FilterTrajectoriesOperator(BaseOperator):
    """
    轨迹过滤算子：
    根据综合评分（多样性 + 性能）对输入的轨迹进行排序、过滤，
    并根据配置执行重标签和删除操作。

    注意：此算子直接修改 traj_pool_manager，run_for_instance 返回空 OperatorResult。
    """

    def get_name(self) -> str:
        return "filter_trajectories"

    def run_for_instance(
        self,
        step_config: StepConfig,
        instance_name: str,
        instance_entry: dict[str, Any],
        *,
        problem_description: str = "",
        traj_pool_manager: TrajPoolManager | None = None,
    ) -> OperatorResult:
        """处理单个实例的轨迹过滤。

        此方法需要额外的 traj_pool_manager 参数来执行重标签和删除操作。
        返回空 OperatorResult（filter 不触发 PerfAgent 调用）。

        Args:
            step_config: 步骤配置（StepConfig 对象）。
            instance_name: 实例名称。
            instance_entry: 该实例在轨迹池中的数据字典。
            traj_pool_manager: 轨迹池管理器（用于执行 relabel/delete 操作）。

        Returns:
            空 OperatorResult。
        """
        if not isinstance(instance_entry, dict):
            return OperatorResult()

        input_labels = [item.get("label") for item in (step_config.inputs or []) if item.get("label")]

        filter_strategy = step_config.filter_strategy or {}
        top_k = filter_strategy.get("top_k")
        relabel_as: list[str] = []
        if isinstance(filter_strategy.get("relabel_as"), list):
            relabel_as = [str(x) for x in filter_strategy.get("relabel_as")]
        relabel_map: dict[str, str] = (
            filter_strategy.get("relabel", {}) if isinstance(filter_strategy.get("relabel"), dict) else {}
        )

        # 提取实例的标签
        inst_labels: list[str] = []
        for k, v in instance_entry.items():
            if k == "problem":
                continue
            if isinstance(v, dict):
                if v.get("label"):
                    inst_labels.append(str(v.get("label")))
                else:
                    inst_labels.append(str(k))

        if input_labels:
            inst_labels = [l for l in inst_labels if l in input_labels]
        if not inst_labels:
            return OperatorResult()

        # 计算要保留和删除的标签
        kept, deleted = self._calculate_kept_and_deleted_labels(inst_labels, instance_entry, top_k)

        # 应用重标签规则
        relabel_ops: list[tuple[str, str]] = []
        if relabel_as:
            limit = min(len(relabel_as), len(kept))
            for idx in range(limit):
                old = kept[idx]
                new = relabel_as[idx]
                if old != new:
                    relabel_ops.append((old, new))
            # 超出 relabel_as 范围的保持原标签
        else:
            for old in kept:
                new = relabel_map.get(old, old)
                if new != old:
                    relabel_ops.append((old, new))

        # 执行 relabel 和 delete 操作
        if traj_pool_manager is not None:
            for old, new in relabel_ops:
                try:
                    traj_pool_manager.relabel(
                        old,
                        new,
                        instance_name=instance_name,
                        operator_name=self.get_name(),
                        delete_old=False,
                    )
                except Exception:
                    pass

            has_relabel_strategy = bool(relabel_as) or bool(relabel_map)
            if deleted and not has_relabel_strategy:
                try:
                    traj_pool_manager.delete_trajectories(deleted, instance_name=instance_name)
                except Exception:
                    pass

        return OperatorResult()

    def _calculate_kept_and_deleted_labels(
        self, inst_labels: list[str], entry: dict[str, Any], top_k: int | None = None
    ) -> tuple[list[str], list[str]]:
        n = len(inst_labels)
        self.logger.info(f"开始过滤轨迹，共 {n} 个标签，top_k={top_k}")

        if top_k is None or int(top_k) >= n:
            self.logger.info(f"top_k 为 None 或大于等于标签数量，保留所有 {n} 个标签")
            return inst_labels, []
        k = max(0, int(top_k))

        candidates: list[dict[str, Any]] = []
        for l in inst_labels:
            sub = entry.get(l) if isinstance(entry, dict) else None
            code_text = sub.get("solution", "") if isinstance(sub, dict) else ""

            perf_val = sub.get("metric", "") if isinstance(sub, dict) else ""
            try:
                perf_num = float(perf_val) if perf_val is not None else None
            except Exception:
                perf_num = None
            score = 0.0
            if perf_num is not None and math.isfinite(perf_num) and perf_num > 0.0:
                score = 1.0 / perf_num
            candidates.append({"label": l, "text": code_text or "", "score": float(score)})

        if n <= k:
            kept_labels = [c["label"] for c in sorted(candidates, key=lambda x: x["score"], reverse=True)]
            self.logger.info(f"标签数量 {n} 小于等于 top_k {k}，按分数排序保留所有标签")
            return kept_labels, [l for l in inst_labels if l not in kept_labels]

        try:
            self.logger.info("使用聚类算法进行轨迹过滤")
            import numpy as np

            try:
                import Levenshtein

                self.logger.info("成功导入 Levenshtein 模块")
            except ImportError:
                Levenshtein = None
                self.logger.warning("未找到 Levenshtein 模块，将使用其他方法")
            from sklearn.cluster import AgglomerativeClustering

            texts = [c["text"] for c in candidates]
            dist = np.zeros((n, n), dtype=float)
            for i in range(n):
                for j in range(i + 1, n):
                    if Levenshtein:
                        sim = Levenshtein.ratio(texts[i], texts[j])
                    else:
                        raise ImportError("Levenshtein module is required for clustering.")
                    d = 1.0 - float(sim)
                    dist[i, j] = d
                    dist[j, i] = d

            num_clusters = min(k, n)
            self.logger.info(f"进行层次聚类，聚类数量: {num_clusters}")
            labels_arr = None
            try:
                clustering = AgglomerativeClustering(n_clusters=num_clusters, metric="precomputed", linkage="average")
                labels_arr = clustering.fit_predict(dist)
            except TypeError:
                clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity="precomputed", linkage="average")
                labels_arr = clustering.fit_predict(dist)

            selected_indices: list[int] = []
            unique_cluster_ids = set(labels_arr)
            self.logger.info(f"聚类完成，实际聚类数量: {len(unique_cluster_ids)}")

            for cluster_id in unique_cluster_ids:
                indices = [idx for idx in range(n) if labels_arr[idx] == cluster_id]
                if not indices:
                    continue
                best_idx = max(indices, key=lambda idx: candidates[idx]["score"])
                selected_indices.append(best_idx)

            selected = [candidates[idx] for idx in selected_indices]
            selected.sort(key=lambda x: x["score"], reverse=True)

            kept_labels = [c["label"] for c in selected]
            self.logger.info(f"聚类过滤完成，保留 {len(kept_labels)} 个标签，删除 {n - len(kept_labels)} 个标签")

        except Exception as e:
            self.logger.warning(f"聚类算法失败: {e}，将使用分数排序作为备用方案")
            candidates.sort(key=lambda x: x["score"], reverse=True)
            kept_labels = [c["label"] for c in candidates[:k]]
            self.logger.info(f"使用分数排序备用方案，保留前 {k} 个标签")

        deleted = [l for l in inst_labels if l not in kept_labels]
        self.logger.info(f"过滤完成，最终保留 {len(kept_labels)} 个标签，删除 {len(deleted)} 个标签")
        return kept_labels, deleted


# 注册算子
from .registry import register_operator

register_operator("filter_trajectories", FilterTrajectoriesOperator)
