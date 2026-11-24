"""梯度压缩载荷：编码 / 解码 / 跨进程合并。

四种载荷（kind）：
- dense  : 不压缩，原样 float32
- topk   : 绝对值最大的 k 个元素，发送 (值, 下标)，需求点1/4
- sign   : 每个元素 1 个正负号比特 + 整层一个缩放系数，需求点2
- lowrank: G ≈ P @ Q^T 的一次幂迭代低秩分解，需求点3

服务端把任意类型的多个 worker 载荷合并成一个 dense 载荷再下发，
因此 worker 收到的永远是和原始 allreduce 语义一致的平均梯度（压缩误差留在各 worker 本地）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FLOAT = np.float32
INDEX = np.int32


# --------------------------------------------------------------------------- #
# 载荷定义
# --------------------------------------------------------------------------- #
@dataclass
class DensePayload:
    tid: int
    shape: tuple[int, ...]
    values: np.ndarray  # 已展平 float32

    kind = "dense"

    def nbytes(self) -> int:
        return int(self.values.nbytes)

    def to_wire(self) -> tuple[dict, list[np.ndarray]]:
        header = {"kind": self.kind, "tid": self.tid, "shape": list(self.shape)}
        return header, [self.values.astype(FLOAT, copy=False)]

    def to_dense(self) -> np.ndarray:
        return self.values.reshape(self.shape)


@dataclass
class TopKPayload:
    tid: int
    shape: tuple[int, ...]
    n: int
    values: np.ndarray   # (k,) float32
    indices: np.ndarray  # (k,) int32

    kind = "topk"

    def nbytes(self) -> int:
        return int(self.values.nbytes + self.indices.nbytes)

    def to_wire(self) -> tuple[dict, list[np.ndarray]]:
        header = {"kind": self.kind, "tid": self.tid,
                  "shape": list(self.shape), "n": self.n, "k": int(len(self.values))}
        return header, [self.values.astype(FLOAT, copy=False),
                        self.indices.astype(INDEX, copy=False)]

    def to_dense(self) -> np.ndarray:
        out = np.zeros(self.n, dtype=FLOAT)
        np.add.at(out, self.indices, self.values)
        return out.reshape(self.shape)


@dataclass
class SignPayload:
    tid: int
    shape: tuple[int, ...]
    n: int
    scale: float
    bits: np.ndarray  # uint8 打包，1 表示正号

    kind = "sign"

    def nbytes(self) -> int:
        return int(self.bits.nbytes) + 4  # 缩放系数按 float32 计 4 字节

    def to_wire(self) -> tuple[dict, list[np.ndarray]]:
        header = {"kind": self.kind, "tid": self.tid,
                  "shape": list(self.shape), "n": self.n,
                  "scale": float(self.scale)}
        return header, [self.bits]

    def to_dense(self) -> np.ndarray:
        signs = np.unpackbits(self.bits)[: self.n].astype(FLOAT) * 2.0 - 1.0
        return (signs * np.float32(self.scale)).reshape(self.shape)


@dataclass
class LowRankPayload:
    tid: int
    shape: tuple[int, ...]
    p: np.ndarray  # (m, r) float32
    q: np.ndarray  # (n, r) float32

    kind = "lowrank"

    def nbytes(self) -> int:
        return int(self.p.nbytes + self.q.nbytes)

    def to_wire(self) -> tuple[dict, list[np.ndarray]]:
        m, r = self.p.shape
        n = self.q.shape[0]
        header = {"kind": self.kind, "tid": self.tid,
                  "shape": list(self.shape), "m": int(m), "n": int(n),
                  "r": int(r)}
        return header, [self.p.astype(FLOAT, copy=False),
                        self.q.astype(FLOAT, copy=False)]

    def to_dense(self) -> np.ndarray:
        return (self.p @ self.q.T).reshape(self.shape)


Payload = DensePayload | TopKPayload | SignPayload | LowRankPayload


def payload_from_wire(header: dict, buffers: list[np.ndarray]) -> Payload:
    kind = header["kind"]
    shape = tuple(header["shape"])
    tid = int(header["tid"])
    if kind == "dense":
        return DensePayload(tid, shape, buffers[0].astype(FLOAT, copy=False))
    if kind == "topk":
        return TopKPayload(tid, shape, int(header["n"]),
                           buffers[0].astype(FLOAT, copy=False),
                           buffers[1].astype(INDEX, copy=False))
    if kind == "sign":
        return SignPayload(tid, shape, int(header["n"]),
                           float(header["scale"]),
                           buffers[0].astype(np.uint8, copy=False))
    if kind == "lowrank":
        m, n, r = int(header["m"]), int(header["n"]), int(header["r"])
        return LowRankPayload(
            tid, shape,
            buffers[0].astype(FLOAT, copy=False).reshape(m, r),
            buffers[1].astype(FLOAT, copy=False).reshape(n, r))
    raise ValueError(f"未知载荷类型: {kind}")


# --------------------------------------------------------------------------- #
# 服务端合并：需求点4 —— 各 worker 选中的位置不同，先交换下标和值再求和
# --------------------------------------------------------------------------- #
def merge_payloads(payloads: list[Payload], world_size: int) -> DensePayload:
    """把 W 个 worker 的同一张量载荷合并成平均后的 dense 载荷。"""
    first = payloads[0]
    tid, shape = first.tid, first.shape
    kind = first.kind
    w = np.float32(world_size)

    if kind == "dense":
        total = sum(p.values for p in payloads)
        return DensePayload(tid, shape, (total / w).astype(FLOAT))

    if kind == "topk":
        # 各 worker 的下标集合一般不同：重建到同一 dense 缓冲后再求和，
        # 未被任何 worker 选中的位置自然为 0
        n = first.n
        total = np.zeros(n, dtype=np.float64)
        for p in payloads:
            np.add.at(total, p.indices, p.values.astype(np.float64))
        return DensePayload(tid, shape, (total / w).astype(FLOAT))

    if kind == "sign":
        n = first.n
        total = np.zeros(n, dtype=np.float64)
        for p in payloads:
            total += p.to_dense().ravel().astype(np.float64)
        return DensePayload(tid, shape, (total / w).astype(FLOAT))

    if kind == "lowrank":
        total = sum(p.to_dense().ravel().astype(np.float64) for p in payloads)
        return DensePayload(tid, shape, (total / w).astype(FLOAT))

    raise ValueError(f"未知载荷类型: {kind}")


# --------------------------------------------------------------------------- #
# worker 端压缩函数，返回 (载荷, 残差)；残差交给误差累积
# --------------------------------------------------------------------------- #
def compress_dense(g: np.ndarray, tid: int) -> tuple[DensePayload, np.ndarray]:
    flat = g.ravel().astype(FLOAT, copy=True)
    return DensePayload(tid, g.shape, flat), np.zeros_like(g)


def compress_topk(g: np.ndarray, tid: int, density: float
                  ) -> tuple[TopKPayload, np.ndarray]:
    """需求点1：按绝对值挑最大的一小部分元素发送，其余累积进误差。

    计算保持 g 的 dtype（训练侧为 float64，残差精确），仅线上载荷转 float32。
    """
    work_dtype = g.dtype
    flat = g.ravel()
    n = flat.size
    k = max(1, int(np.ceil(n * min(1.0, density))))
    if k >= n:
        idx = np.arange(n, dtype=INDEX)
        vals = flat.copy()
        approx = flat
    else:
        # argpartition 取候选，再按下标排序保证结果确定
        cand = np.argpartition(-np.abs(flat), kth=k - 1)[:k]
        idx = np.sort(cand).astype(INDEX)
        vals = flat[idx].copy()
        approx = np.zeros(n, dtype=work_dtype)
        approx[idx] = vals
    residual = flat - approx
    payload = TopKPayload(tid, g.shape, n,
                          vals.astype(FLOAT), idx)
    return payload, residual.reshape(g.shape)


def pack_sign_bits(sign_positive: np.ndarray) -> np.ndarray:
    n = sign_positive.size
    pad = (-n) % 8
    if pad:
        sign_positive = np.concatenate(
            [sign_positive.view(np.uint8), np.zeros(pad, dtype=np.uint8)])
    return np.packbits(sign_positive.astype(np.uint8))


def compress_sign(g: np.ndarray, tid: int) -> tuple[SignPayload, np.ndarray]:
    """需求点2：只发正负号比特 + 整层一个缩放系数。

    缩放系数取 RMS（平方均值开根）而不是绝对值均值：对 sign 量化来说
    E[s^2]=1，RMS 是让重建在最小二乘意义下最优的逐张量缩放，
    能显著降低单系数 1-bit 压缩的误差、避免误差累积被撑爆。
    """
    flat = g.ravel()
    n = flat.size
    scale = float(np.sqrt(np.mean(flat.astype(np.float64) ** 2))) if n else 0.0
    positive = flat >= 0
    bits = pack_sign_bits(positive)
    approx = np.where(positive,
                      np.asarray(scale, dtype=flat.dtype),
                      np.asarray(-scale, dtype=flat.dtype))
    residual = flat - approx
    return SignPayload(tid, g.shape, n, scale, bits), residual.reshape(g.shape)


def _orthonormalize(mat: np.ndarray) -> np.ndarray:
    """数值稳健的正交化：优先 SVD（对秩亏/近退化矩阵比 QR 更稳）。

    幂迭代里 Q 必须严格正交，否则误差会沿非正交方向在 EF 环路中累积放大。
    """
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    # 全零矩阵：返回确定性的标准基前 r 列，避免随机方向破坏可复现性
    if not np.any(mat):
        r = mat.shape[1]
        return np.eye(mat.shape[0], r, dtype=mat.dtype)
    try:
        u, s, _ = np.linalg.svd(mat, full_matrices=False)
        # SVD 已给出正交列；若有效秩不足则补标准基
        if u.shape[1] < mat.shape[1] or np.any(s < 1e-10):
            q, _ = np.linalg.qr(mat)
            if q.shape[1] == mat.shape[1]:
                return q.astype(mat.dtype)
        return u[:, : mat.shape[1]].astype(mat.dtype)
    except np.linalg.LinAlgError:
        q, _ = np.linalg.qr(mat)
        return q.astype(mat.dtype)


class LowRankState:
    """每个被低秩压缩的张量维护一个右因子 Q，跨步复用（需求点3的热启动）。"""

    def __init__(self, target_rank: int, seed: int,
                 dtype: np.dtype = np.float64) -> None:
        self.target_rank = int(target_rank)
        self._seed = int(seed)
        self.dtype = np.dtype(dtype)
        self.q: np.ndarray | None = None  # (n, r)

    def _init_q(self, n: int, r: int) -> np.ndarray:
        rng = np.random.default_rng(self._seed)
        base = rng.standard_normal((n, r)).astype(self.dtype)
        return _orthonormalize(base)

    def get_q(self, n: int, r: int) -> np.ndarray:
        if self.q is None:
            self.q = self._init_q(n, r)
        cur = self.q.shape[1]
        if r == cur:
            return self.q
        if r < cur:  # 秩调度只会增长，这里做防御性截断
            self.q = self.q[:, :r]
            return self.q
        # 扩秩：新列随机初始化并与旧列正交化
        rng = np.random.default_rng(self._seed + cur)
        extra = rng.standard_normal((n, r - cur)).astype(self.dtype)
        self.q = _orthonormalize(np.concatenate([self.q, extra], axis=1))
        return self.q


def compress_lowrank(g: np.ndarray, tid: int, state: LowRankState, rank: int
                     ) -> tuple[LowRankPayload, np.ndarray]:
    """需求点3：一步幂迭代，Q 用上一步结果作初值。

    幂迭代与残差在 g 的 dtype（训练侧 float64）下进行，避免低精度残差
    在 EF 环路里被反复放大；只有线上的 p/q 载荷转 float32 传输。
    """
    work = g.dtype
    m, n = g.shape
    r = min(rank, m, n)
    if state.dtype != work:
        state.dtype = np.dtype(work)
        state.q = None if state.q is None else state.q.astype(work)
    q_old = _orthonormalize(state.get_q(n, r))
    p = (g @ q_old).astype(work)          # (m, r)
    q_new = _orthonormalize(g.T @ p)      # (n, r)，一步幂迭代
    state.q = q_new
    approx = (p @ q_new.T).astype(work)
    residual = (g - approx).astype(work)
    payload = LowRankPayload(tid, g.shape,
                             p.astype(FLOAT), q_new.astype(FLOAT))
    return payload, residual
