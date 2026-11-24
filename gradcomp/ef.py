"""误差累积（Error Feedback, EF）与动量的两种先后顺序（需求点8）。

输入始终是本地原始梯度 g_t，输出是“本步要送去压缩的张量 v_t”，
压缩后立即在本地落盘残差 e_t = v_t - hat(v_t)，服务端平均重建值用于更新。

两种顺序（区别只在动量插在 EF 环路的里面还是外面）：

ef_first（经典 EF-SGD，Stich et al. 2018 的标准形式，先 EF 后动量）
    v_t = g_t + e_{t-1}
    hat(v_t) = compress(v_t)，e_t = v_t - hat(v_t)
    m_t = mu * m_{t-1} + hat_bar_t          # hat_bar 为服务端平均重建值
    参数更新用 m_t
这种顺序下 EF 环路不经过动量，残差有收敛保证，是默认且推荐的方式。

mom_first（先动量后压缩，误差围绕动量轨迹累积）
    m_t = mu * m_{t-1} + g_t
    v_t = m_t
    hat(v_t) = compress(v_t)，e_t = v_t - hat(v_t)
    参数更新用服务端平均后的 hat(v_t)
压缩偏激进时该顺序可能抖动，保留它用于两种顺序的切换对比实验。

``use_ef=False`` 时残差每步清零（e_t 不保留、下一步不回加），用于对照实验。
"""

from __future__ import annotations

import numpy as np


class EFState:
    """一个张量的动量与误差累积状态。"""

    def __init__(self, shape: tuple[int, ...], dtype=np.float64) -> None:
        self.momentum = np.zeros(shape, dtype=dtype)
        self.error = np.zeros(shape, dtype=dtype)


class EFBank:
    """全部张量的 EF/动量状态集合。"""

    def __init__(self, names_shapes: list[tuple[str, tuple[int, ...]]],
                 momentum: float = 0.9, order: str = "ef_first",
                 use_ef: bool = True, dtype=np.float64,
                 ef_norm_limit: float = 2.0) -> None:
        if order not in ("ef_first", "mom_first"):
            raise ValueError(f"未知 EF 顺序: {order}")
        self.order = order
        self.momentum = float(momentum)
        self.use_ef = bool(use_ef)
        # 误差回加上限：||e|| 超过该倍数的 ||g|| 时按比例收缩，
        # 防止强压缩（低秩/符号）下误差反馈形成正反馈而发散
        self.ef_norm_limit = float(ef_norm_limit)
        self.states = {name: EFState(shape, dtype) for name, shape in names_shapes}

    def prepare(self, name: str, grad: np.ndarray) -> np.ndarray:
        """计算本步要送去压缩的张量 v。"""
        st = self.states[name]
        if self.order == "ef_first":
            # 先把上一步残差回加到原始梯度；对误差做范数正则化防发散
            if self.use_ef and np.any(st.error):
                gnorm = float(np.linalg.norm(grad))
                enorm = float(np.linalg.norm(st.error))
                if gnorm > 0 and enorm > self.ef_norm_limit * gnorm:
                    error = st.error * (self.ef_norm_limit * gnorm / enorm)
                else:
                    error = st.error
                v = grad + error
            else:
                v = grad + (st.error if self.use_ef else 0.0)
        else:
            # 先动量，动量轨迹本身作为待压缩量
            st.momentum = self.momentum * st.momentum + grad
            v = st.momentum.copy()
        return v

    def commit_residual(self, name: str, v: np.ndarray,
                        local_reconstructed: np.ndarray) -> None:
        """通信前在本地落盘残差 e = v - hat(v)；use_ef=False 时丢弃。

        必须用“本地重建值”而非服务端平均值：平均值含其他 worker 的信息，
        写进本地误差会让 EF 失去无偏性。
        """
        st = self.states[name]
        if self.use_ef:
            st.error = v - local_reconstructed
        else:
            st.error = np.zeros_like(v)

    def apply_server_update(self, name: str,
                            avg_reconstructed: np.ndarray) -> np.ndarray:
        """服务端平均重建值到达后，按顺序产出最终参数更新量。"""
        st = self.states[name]
        if self.order == "ef_first":
            # 动量作用在 EF 环路输出上
            st.momentum = self.momentum * st.momentum + avg_reconstructed
            return st.momentum
        # mom_first：动量已在 prepare 里计入 v，直接用平均重建值更新
        return avg_reconstructed

    def residual_norm(self, name: str) -> float:
        return float(np.linalg.norm(self.states[name].error))
