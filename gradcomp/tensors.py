"""参数张量登记表与压缩资格判定。

需求点5：很小的层和偏置不压缩，直接按原样汇总；哪些层压缩按大小阈值自动决定。

规则（可通过 GradientComm 参数调整）：
- 一维张量（偏置向量）永远走原样 dense；
- 二维权重矩阵中元素个数小于 ``min_compress_numel`` 的也走 dense；
- 其余二维张量按所选压缩方法处理。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TensorSpec:
    """一个参数张量的静态描述。"""

    tid: int
    name: str
    shape: tuple[int, ...]

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape))


class TensorRegistry:
    """登记一次训练中参与 allreduce 的全部张量。"""

    def __init__(self, names_shapes: list[tuple[str, tuple[int, ...]]],
                 min_compress_numel: int = 256) -> None:
        self.specs: list[TensorSpec] = [
            TensorSpec(tid=i, name=name, shape=shape)
            for i, (name, shape) in enumerate(names_shapes)
        ]
        self.min_compress_numel = int(min_compress_numel)
        self.compressible = [self._is_compressible(s) for s in self.specs]

    def _is_compressible(self, spec: TensorSpec) -> bool:
        # 偏置等一维张量不压缩；二维但太小的也不压缩
        if len(spec.shape) < 2:
            return False
        return spec.numel >= self.min_compress_numel

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.specs]

    def get(self, tid: int) -> TensorSpec:
        return self.specs[tid]

    def describe(self) -> list[str]:
        lines = []
        for spec, ok in zip(self.specs, self.compressible):
            kind = "压缩" if ok else "原样"
            lines.append(f"  [{spec.tid}] {spec.name} shape={spec.shape} "
                         f"numel={spec.numel} -> {kind}")
        return lines
