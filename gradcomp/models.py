"""纯 numpy 的多层感知机（ReLU 隐藏层 + 线性输出），手写反向传播。

用于在没有深度学习框架的旧机器上验证压缩通信。参数命名规则：
W1/b1, W2/b2, ...，二维 W 是压缩候选，一维 b 永远原样汇总。
"""

from __future__ import annotations

import numpy as np


def _init_layer(fan_in: int, fan_out: int, rng: np.random.Generator
                ) -> tuple[np.ndarray, np.ndarray]:
    # He 初始化
    scale = np.sqrt(2.0 / fan_in)
    w = rng.standard_normal((fan_in, fan_out)) * scale
    b = np.zeros(fan_out)
    return w.astype(np.float64), b.astype(np.float64)


class MLP:
    def __init__(self, layer_sizes: list[int], seed: int = 0) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("至少需要输入/输出两层")
        rng = np.random.default_rng(seed)
        self.layer_sizes = list(layer_sizes)
        self.params: dict[str, np.ndarray] = {}
        for i in range(len(layer_sizes) - 1):
            w, b = _init_layer(layer_sizes[i], layer_sizes[i + 1], rng)
            self.params[f"W{i + 1}"] = w
            self.params[f"b{i + 1}"] = b
        self._cache: dict[str, np.ndarray] = {}

    @property
    def names_shapes(self) -> list[tuple[str, tuple[int, ...]]]:
        return [(name, p.shape) for name, p in self.params.items()]

    def copy_params(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.params.items()}

    def load_params(self, params: dict[str, np.ndarray]) -> None:
        for k, v in params.items():
            self.params[k] = v.copy()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = {"a0": x}
        a = x
        n_layers = len(self.layer_sizes) - 1
        for i in range(1, n_layers + 1):
            w, b = self.params[f"W{i}"], self.params[f"b{i}"]
            z = a @ w + b
            self._cache[f"z{i}"] = z
            if i < n_layers:
                a = np.maximum(z, 0.0)
            else:
                a = z  # 输出层线性
            self._cache[f"a{i}"] = a
        return a

    def backward(self, y: np.ndarray) -> dict[str, np.ndarray]:
        """MSE 损失（含 1/2 因子）的反向传播，返回各参数梯度。"""
        n = y.shape[0]
        n_layers = len(self.layer_sizes) - 1
        pred = self._cache[f"a{n_layers}"]
        delta = (pred - y) / n  # d L / d 输出
        grads: dict[str, np.ndarray] = {}
        for i in range(n_layers, 0, -1):
            a_prev = self._cache[f"a{i - 1}"]
            grads[f"W{i}"] = a_prev.T @ delta
            grads[f"b{i}"] = delta.sum(axis=0)
            if i > 1:
                w = self.params[f"W{i}"]
                delta = (delta @ w.T) * (self._cache[f"z{i - 1}"] > 0)
        return grads

    def sgd_update(self, grads: dict[str, np.ndarray], lr: float) -> None:
        for name, g in grads.items():
            self.params[name] -= lr * g


def mse_loss(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((pred - y) ** 2))
