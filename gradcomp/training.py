"""训练循环与实验配置。

worker 逻辑在多进程（TCP）和单进程参照（线程 + InProcessTransport）下
完全一致：同样的确定性批次划分、同样的模型初始化种子、同样的压缩器参数，
因此两条路径输出应当逐位相同（浮点容差内）。
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field

import numpy as np

from .comm import GradientComm
from .data import batch_rng, load_dataset, shard_indices
from .models import MLP, mse_loss
from .transport import (DEFAULT_BANDWIDTH_BPS, DEFAULT_RTT_SECONDS,
                        InProcessClient, InProcessTransport, SocketClient)


@dataclass
class RunConfig:
    method: str = "topk"               # none/topk/sign/lowrank
    dataset: str = "generic"
    layer_sizes: list[int] = field(default_factory=lambda: [6, 32, 16, 1])
    steps: int = 60
    batch_size: int = 32
    lr: float = 0.05
    momentum: float = 0.9
    ef_order: str = "ef_first"
    use_ef: bool = True
    world_size: int = 4
    bucket_bytes: int = 4096
    min_compress_numel: int = 256
    target_density: float = 0.1
    target_rank: int = 2
    warmup_steps: int = 3
    ramp_steps: int = 10
    grad_clip: float = 5.0
    ef_norm_limit: float = 2.0
    # 是否参与“EF 精度必须落在容差内”的硬性校验。
    # 低秩在大矩阵上受“单步幂迭代+低秩”约束只能作展示，置 False。
    accuracy_gated: bool = True
    model_seed: int = 100
    data_seed: int = 42
    bandwidth_bps: float = DEFAULT_BANDWIDTH_BPS
    rtt_seconds: float = DEFAULT_RTT_SECONDS
    label: str = ""

    def key(self) -> str:
        tag = self.label or self.method
        if self.method != "none":
            tag += f"|ef={'on' if self.use_ef else 'off'}|{self.ef_order}"
        return tag


def serialize_config(cfg: RunConfig) -> str:
    return json.dumps(asdict(cfg), sort_keys=True)


def config_from_dict(d: dict) -> RunConfig:
    known = RunConfig.__dataclass_fields__.keys()
    return RunConfig(**{k: v for k, v in d.items() if k in known})


def make_comm(cfg: RunConfig, names_shapes, client) -> GradientComm:
    return GradientComm(
        names_shapes, client=client, method=cfg.method,
        world_size=cfg.world_size, momentum=cfg.momentum,
        ef_order=cfg.ef_order, use_ef=cfg.use_ef,
        bucket_bytes=cfg.bucket_bytes,
        min_compress_numel=cfg.min_compress_numel,
        target_density=cfg.target_density, target_rank=cfg.target_rank,
        warmup_steps=cfg.warmup_steps, ramp_steps=cfg.ramp_steps,
        lowrank_seed=cfg.model_seed + 77,
        bandwidth_bps=cfg.bandwidth_bps, rtt_seconds=cfg.rtt_seconds,
        grad_clip=cfg.grad_clip, ef_norm_limit=cfg.ef_norm_limit,
    )


def train_worker(cfg: RunConfig, rank: int,
                 client: SocketClient | InProcessClient) -> dict:
    """一个 worker 的完整训练，返回指标字典（rank 0 的结果用于汇总）。"""
    data = load_dataset(cfg.dataset)
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]
    n_train = x_train.shape[0]

    # 所有 worker 同种子初始化 -> 初始参数一致（数据并行的前提）
    model = MLP(cfg.layer_sizes, seed=cfg.model_seed)
    comm = make_comm(cfg, model.names_shapes, client)

    own_idx = shard_indices(cfg.batch_size, cfg.world_size, rank)
    loss_curve: list[float] = []
    test_curve: list[float] = []

    for step in range(cfg.steps):
        rng = batch_rng(step, cfg.data_seed)
        batch_idx = rng.integers(0, n_train, size=cfg.batch_size)
        xb, yb = x_train[batch_idx], y_train[batch_idx]
        xb_local = xb[own_idx]
        yb_local = yb[own_idx]

        pred = model.forward(xb_local)
        _ = mse_loss(pred, yb_local)
        local_grads = model.backward(yb_local)

        # 全局梯度范数裁剪：压缩（尤其低秩/符号）的 EF 环路对异常大批梯度
        # 敏感，裁剪保证旧硬件上的数值稳定性（数据并行训练的标准工程做法）
        if cfg.grad_clip and cfg.grad_clip > 0:
            gnorm = np.sqrt(sum((g ** 2).sum() for g in local_grads.values()))
            if gnorm > cfg.grad_clip:
                scale = cfg.grad_clip / gnorm
                local_grads = {k: g * scale for k, g in local_grads.items()}

        # 压缩 + allreduce（服务端做平均）
        avg_grads = comm.allreduce_gradients(step, local_grads)
        model.sgd_update(avg_grads, cfg.lr)

        # 全体测试集损失（各 worker 模型一致，记录同一份数值）
        train_loss = mse_loss(model.forward(x_train), y_train)
        test_loss = mse_loss(model.forward(x_test), y_test)
        loss_curve.append(train_loss)
        test_curve.append(test_loss)

    summary = comm.summary()
    step_records = []
    for rec in comm.records:
        step_records.append({
            "step": rec.step,
            "uplink_bytes": rec.timing.uplink_bytes,
            "downlink_bytes": rec.timing.downlink_bytes,
            "measured_seconds": rec.timing.measured_seconds,
            "modeled_seconds": rec.timing.modeled_seconds,
            "total_residual_norm": rec.total_residual_norm,
            "bucket_count": rec.bucket_count,
            "tensors": [{
                "tid": ts.tid, "name": ts.name, "kind": ts.kind,
                "raw_bytes": ts.raw_bytes,
                "compressed_bytes": ts.compressed_bytes,
                "residual_norm": ts.residual_norm,
                "input_norm": ts.input_norm,
            } for ts in rec.tensor_stats],
        })

    return {
        "rank": rank,
        "label": cfg.key(),
        "config": asdict(cfg),
        "final_train_loss": loss_curve[-1],
        "final_test_loss": test_curve[-1],
        "loss_curve": loss_curve,
        "test_curve": test_curve,
        "summary": summary,
        "step_records": step_records,
        "final_params": {k: v.tolist() for k, v in model.params.items()},
    }


def run_inprocess(cfg: RunConfig) -> list[dict]:
    """单进程参照：W 个线程跑同一套 worker 逻辑，InProcessTransport 合并。"""
    transport = InProcessTransport(cfg.world_size)
    results: list[dict] = []

    def _work(rank: int) -> None:
        client = InProcessClient(
            transport, rank,
            bandwidth_bps=cfg.bandwidth_bps, rtt_seconds=cfg.rtt_seconds)
        results.append(train_worker(cfg, rank, client))

    threads = [threading.Thread(target=_work, args=(r,))
               for r in range(cfg.world_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    results.sort(key=lambda r: r["rank"])
    return results
