"""GradientComm：一次梯度 allreduce 的完整压缩通信流程。

串联：压缩资格判定 -> EF/动量(两种顺序可切换) -> 压缩器 -> 拼桶 ->
传输(TCP 多进程或进程内参照) -> 重建 -> 残差回写 -> 指标记录。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .compressors import (DensePayload, LowRankState, Payload,
                          compress_dense, compress_lowrank, compress_sign,
                          compress_topk)
from .ef import EFBank
from .framing import bucketize
from .schedule import CompressionSchedule
from .tensors import TensorRegistry
from .transport import (DEFAULT_BANDWIDTH_BPS, DEFAULT_RTT_SECONDS,
                        InProcessClient, SocketClient, StepTiming)


@dataclass
class TensorStepStat:
    step: int
    tid: int
    name: str
    kind: str
    raw_bytes: int
    compressed_bytes: int          # 该 worker 本步该张量的线上载荷字节
    residual_norm: float
    input_norm: float


@dataclass
class StepRecord:
    step: int
    tensor_stats: list[TensorStepStat]
    timing: StepTiming
    total_residual_norm: float
    bucket_count: int


class GradientComm:
    def __init__(self,
                 names_shapes: list[tuple[str, tuple[int, ...]]],
                 client: SocketClient | InProcessClient,
                 method: str = "topk",
                 *,
                 world_size: int,
                 momentum: float = 0.9,
                 ef_order: str = "ef_first",
                 use_ef: bool = True,
                 bucket_bytes: int = 4096,
                 min_compress_numel: int = 256,
                 target_density: float = 0.1,
                 target_rank: int = 2,
                 warmup_steps: int = 3,
                 ramp_steps: int = 10,
                 lowrank_seed: int = 1234,
                 bandwidth_bps: float = DEFAULT_BANDWIDTH_BPS,
                 rtt_seconds: float = DEFAULT_RTT_SECONDS,
                 grad_clip: float = 5.0,
                 ef_norm_limit: float = 2.0) -> None:
        if method not in ("none", "topk", "sign", "lowrank"):
            raise ValueError(f"未知压缩方法: {method}")
        self.registry = TensorRegistry(names_shapes, min_compress_numel)
        self.client = client
        self.method = method
        self.world_size = world_size
        self.bucket_bytes = int(bucket_bytes)
        self.grad_clip = float(grad_clip)
        self.schedule = CompressionSchedule(warmup_steps, ramp_steps,
                                            target_density)
        self.ef = EFBank(names_shapes, momentum=momentum, order=ef_order,
                         use_ef=use_ef, dtype=np.float64,
                         ef_norm_limit=ef_norm_limit)
        self.target_rank = int(target_rank)
        # 每个可压缩张量一个低秩状态，seed 按 tid 错开但固定，保证可复现
        self.lowrank_states = {
            spec.tid: LowRankState(self.target_rank, lowrank_seed + spec.tid)
            for spec, ok in zip(self.registry.specs, self.registry.compressible)
            if ok
        }
        self.records: list[StepRecord] = []

    # ------------------------------------------------------------------ #
    def _compress_one(self, step: int, tid: int, v: np.ndarray) -> Payload:
        spec = self.registry.specs[tid]
        dense_step = self.schedule.is_dense_step(step)
        if self.method == "none" or not self.registry.compressible[tid] \
                or dense_step:
            payload, _ = compress_dense(v.astype(np.float32), tid)
            return payload

        if self.method == "topk":
            density = self.schedule.topk_density(step)
            payload, _ = compress_topk(v.astype(np.float32), tid, density)
            return payload

        if self.method == "sign":
            payload, _ = compress_sign(v.astype(np.float32), tid)
            return payload

        # lowrank：只支持二维矩阵；资格判定已保证
        if v.ndim != 2:  # 防御：一维永远走 dense
            payload, _ = compress_dense(v.astype(np.float32), tid)
            return payload
        rank = self.schedule.lowrank_rank(step, self.target_rank)
        payload, _ = compress_lowrank(v.astype(np.float32), tid,
                                      self.lowrank_states[tid], rank)
        return payload

    # ------------------------------------------------------------------ #
    def allreduce_gradients(self, step: int,
                            grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """输入本 worker 的原始梯度，返回服务端平均后的（压缩重建）梯度。"""
        payloads: list[Payload] = []
        prepared: list[np.ndarray] = []

        for spec in self.registry.specs:
            g = grads[spec.name]
            # 1) EF/动量准备出待压缩张量 v（顺序可切换）
            v = self.ef.prepare(spec.name, g)
            prepared.append(v)
            # 2) 压缩
            payload = self._compress_one(step, spec.tid, v)
            # 3) 残差在通信前于本地落盘：e = v - 本地重建 hat(v)
            self.ef.commit_residual(spec.name, v,
                                    payload.to_dense().astype(np.float64))
            payloads.append(payload)

        # 4) 拼桶 + 传输（服务端把各 worker 的重建值平均后下发）
        buckets = bucketize(payloads, self.bucket_bytes)
        result_buckets, timing = self.client.allreduce(
            step, self.world_size, buckets)
        merged: list[DensePayload] = [
            p for bucket in result_buckets for p in bucket
        ]

        # 5) 服务端平均结果按 EF 顺序产出最终更新量；记录指标
        out: dict[str, np.ndarray] = {}
        tensor_stats: list[TensorStepStat] = []
        total_res = 0.0
        for spec, v, payload, dense_out in zip(
                self.registry.specs, prepared, payloads, merged):
            assert dense_out.tid == spec.tid
            avg_reconstructed = dense_out.to_dense().astype(np.float64)
            update = self.ef.apply_server_update(spec.name, avg_reconstructed)
            res_norm = self.ef.residual_norm(spec.name)
            total_res += res_norm ** 2
            out[spec.name] = update
            tensor_stats.append(TensorStepStat(
                step=step, tid=spec.tid, name=spec.name,
                kind=payload.kind,
                raw_bytes=spec.numel * 4,
                compressed_bytes=payload.nbytes(),
                residual_norm=res_norm,
                input_norm=float(np.linalg.norm(v)),
            ))

        record = StepRecord(
            step=step,
            tensor_stats=tensor_stats,
            timing=timing,
            total_residual_norm=float(np.sqrt(total_res)),
            bucket_count=len(buckets),
        )
        self.records.append(record)
        return out

    # ------------------------------------------------------------------ #
    def summary(self) -> dict:
        """汇总一次训练的通信与压缩指标。"""
        uplink = sum(r.timing.uplink_bytes for r in self.records)
        downlink = sum(r.timing.downlink_bytes for r in self.records)
        modeled = sum(r.timing.modeled_seconds for r in self.records)
        measured = sum(r.timing.measured_seconds for r in self.records)
        raw_total = 0
        comp_total = 0
        per_tensor: dict[str, dict] = {}
        for r in self.records:
            for ts in r.tensor_stats:
                raw_total += ts.raw_bytes
                comp_total += ts.compressed_bytes
                slot = per_tensor.setdefault(ts.name, {
                    "raw_bytes": 0, "compressed_bytes": 0,
                    "kinds": set(), "shape": self.registry.get(ts.tid).shape,
                })
                slot["raw_bytes"] += ts.raw_bytes
                slot["compressed_bytes"] += ts.compressed_bytes
                slot["kinds"].add(ts.kind)
        for slot in per_tensor.values():
            slot["kinds"] = sorted(slot["kinds"])
            slot["ratio"] = (slot["compressed_bytes"] / slot["raw_bytes"]
                             if slot["raw_bytes"] else 0.0)
        return {
            "method": self.method,
            "world_size": self.world_size,
            "steps": len(self.records),
            "uplink_bytes": uplink,
            "downlink_bytes": downlink,
            "total_bytes": uplink + downlink,
            "modeled_seconds": modeled,
            "measured_seconds": measured,
            "raw_uplink_bytes_if_uncompressed": raw_total,
            "compression_ratio": comp_total / raw_total if raw_total else 0.0,
            "per_tensor": per_tensor,
        }
