"""压缩比例调度（需求点7）。

训练开头 ``warmup_steps`` 步完全不压缩（level=0），随后在 ``ramp_steps`` 步内
把压缩程度线性提高到目标值（level=1），之后保持目标压缩比例。

各压缩器对 level 的解释：
- topk   : 采样密度 density = 1 - level*(1-target_density)，level=0 时退化为原样 dense
- sign   : level=0 走 dense，之后走符号压缩
- lowrank: level=0 走 dense，之后秩随 level 从 1 增长到目标 rank
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompressionSchedule:
    warmup_steps: int = 3
    ramp_steps: int = 10
    target_density: float = 0.1  # top-k 目标保留比例

    def level(self, step: int) -> float:
        """返回第 step 步（从 0 开始）的压缩程度，取值 [0, 1]。"""
        if step < self.warmup_steps:
            return 0.0
        if self.ramp_steps <= 0:
            return 1.0
        t = (step - self.warmup_steps + 1) / self.ramp_steps
        return min(1.0, max(0.0, t))

    def topk_density(self, step: int) -> float:
        level = self.level(step)
        return 1.0 - level * (1.0 - self.target_density)

    def use_sign(self, step: int) -> bool:
        return self.level(step) > 0.0

    def lowrank_rank(self, step: int, max_rank: int) -> int:
        level = self.level(step)
        if level <= 0.0:
            return 0  # 0 表示走 dense
        return max(1, min(max_rank, round(level * max_rank)))

    def is_dense_step(self, step: int) -> bool:
        return self.level(step) <= 0.0
