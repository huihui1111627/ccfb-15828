"""数据并行训练的梯度压缩通信模块（纯 numpy + 标准库实现）。

模块组成::

    tensors      张量登记与“哪些层需要压缩”的阈值判定
    schedule     热身/逐步提高压缩比例的调度
    compressors  dense / top-k / 符号 / 低秩 四种载荷的编解码与跨进程合并
    ef           误差累积(EF)与动量(momentum)两种先后顺序
    bucketing    小张量拼桶
    framing      二进制线上分帧（字节数精确可计）
    transport    TCP 多进程通信 + 千兆网耗时建模，以及单进程参照用的 InProcess 通道
    comm         GradientComm：把上面各部件串成一次 allreduce
    models/data  numpy 小模型与固定种子数据
    training     单进程训练循环与指标记录
    runner       多进程实验编排、参照对比、校验
    dashboard    离线 HTML 仪表盘
"""

__version__ = "1.0.0"

METHODS = ("none", "topk", "sign", "lowrank")
EF_ORDERS = ("ef_first", "mom_first")
