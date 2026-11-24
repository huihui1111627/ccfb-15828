"""固定随机种子的合成数据。

两个小数据集：
- ``generic``   ：通用非线性回归，供快速正确性测试
- ``wastewater``：模拟造纸厂污水处理出水指标（COD、出水悬浮物等），
  输入为可测的进水/工艺量，目标含非线性与交叉项，贴近真实预测场景

所有 worker 用同一个数据生成种子；数据并行通过按 rank 对每批样本做
确定性切分实现（同样的 step -> 同样的批次划分，单进程参照可逐位复现）。
"""

from __future__ import annotations

import numpy as np

DATASETS = ("generic", "wastewater")


def _standardize(train: np.ndarray, *others: np.ndarray
                 ) -> tuple[np.ndarray, ...]:
    mean, std = train.mean(axis=0), train.std(axis=0) + 1e-8
    out = [(d - mean) / std for d in (train, *others)]
    return tuple(out)


def make_generic(n_train: int = 256, n_test: int = 64, seed: int = 42
                 ) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    x = rng.standard_normal((n, 6))
    w = rng.standard_normal((6, 1))
    y = (x @ w + 0.5 * np.sin(2.0 * x[:, :1])
         + 0.3 * (x[:, 2:3] * x[:, 3:4])
         + 0.05 * rng.standard_normal((n, 1)))
    x_s, = _standardize(x)
    y_s, = _standardize(y)
    return {
        "name": "generic",
        "x_train": x_s[:n_train], "y_train": y_s[:n_train],
        "x_test": x_s[n_train:], "y_test": y_s[n_train:],
        "feature_names": [f"x{i}" for i in range(6)],
        "target_names": ["y"],
    }


# 进水 COD、进水悬浮物 SS、进水 pH、流量、溶解氧 DO、温度、污泥浓度 MLSS、药剂投加量
WW_FEATURES = ["in_cod", "in_ss", "in_ph", "flow", "do",
               "temp", "mlss", "chem_dose"]
WW_TARGETS = ["out_cod", "out_ss"]


def make_wastewater(n_train: int = 320, n_test: int = 80, seed: int = 7
                    ) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    x = np.column_stack([
        rng.uniform(300, 1200, n),     # 进水 COD mg/L
        rng.uniform(100, 600, n),      # 进水 SS
        rng.uniform(6.5, 8.5, n),      # pH
        rng.uniform(50, 300, n),       # 流量 m3/h
        rng.uniform(0.5, 6.0, n),      # DO
        rng.uniform(15, 35, n),        # 温度
        rng.uniform(2000, 6000, n),    # MLSS
        rng.uniform(20, 150, n),       # 药剂投加
    ])
    in_cod, in_ss, ph, flow, do, temp, mlss, dose = [x[:, i] for i in range(8)]
    out_cod = (
        0.35 * in_cod
        + 0.08 * in_ss
        + 12.0 * np.abs(ph - 7.2)
        + 0.10 * flow
        - 9.0 * do
        + 0.004 * in_cod * (flow / (mlss + 1.0))
        - 0.12 * dose
        + 0.5 * (temp - 25.0) ** 2 * 0.1
        + rng.normal(0, 8.0, n)
    )
    out_ss = (
        0.25 * in_ss
        + 0.03 * in_cod
        + 0.05 * flow
        - 3.0 * do
        - 0.08 * dose
        + 0.01 * in_ss * (1.0 - mlss / 8000.0)
        + rng.normal(0, 3.0, n)
    )
    y = np.column_stack([out_cod, out_ss])
    x_s, = _standardize(x)
    y_s, = _standardize(y)
    return {
        "name": "wastewater",
        "x_train": x_s, "y_train": y_s,
        "x_test": x_s[n_train:], "y_test": y_s[n_train:],
        "feature_names": WW_FEATURES,
        "target_names": WW_TARGETS,
    }


def load_dataset(name: str) -> dict[str, np.ndarray]:
    if name == "generic":
        return make_generic()
    if name == "wastewater":
        return make_wastewater()
    raise ValueError(f"未知数据集: {name}，可选 {DATASETS}")


def shard_indices(n: int, world_size: int, rank: int) -> np.ndarray:
    """把 0..n-1 确定性均分给各 worker（余数给靠前的 rank）。"""
    base, rem = divmod(n, world_size)
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return np.arange(start, end)


def batch_rng(step: int, epoch_seed: int) -> np.random.Generator:
    """每步一个确定性 RNG，所有 worker 一致。"""
    return np.random.default_rng(epoch_seed * 100003 + step)
