# 数据并行训练的梯度压缩通信模块（纯 numpy + 标准库）

面向多台旧显卡机器、千兆网互联场景：每步同步梯度的通信时间远超计算时间，
因此用梯度压缩降低每步要发的字节数。本项目用**多个进程 + 真实 TCP** 模拟
多卡/多机，全部逻辑只用 Python 与 numpy 实现，并附带一个离线 HTML 仪表盘。

## 实现的压缩方法与机制

| 需求点 | 实现 | 代码位置 |
| --- | --- | --- |
| 1. top-k 稀疏 + 误差累积 | 按绝对值选最大的 k 个元素发送 `(值,下标)`，未发部分累积进本地误差，下一步先回加再选 | `compressors.compress_topk` / `ef.EFBank` |
| 2. 符号压缩 + 误差累积 | 每元素 1 个正负号比特（位打包）+ 整层一个 RMS 缩放系数，同样带误差累积 | `compressors.compress_sign` |
| 3. 低秩压缩 | 梯度矩阵分解为 `P @ Qᵀ`，右因子 Q 用上一步结果热启动，每步只做一次幂迭代 | `compressors.LowRankState` / `compress_lowrank` |
| 4. 稀疏汇总正确性 | 服务端先交换各 worker 的下标和值，重建到同一 dense 缓冲再求和；下标互不相同也正确 | `compressors.merge_payloads`，测试 `SparseMergeTest` |
| 5. 小层/偏置不压缩 | 一维偏置、元素数低于阈值的二维矩阵自动走原样 dense，阈值可配 | `tensors.TensorRegistry` |
| 6. 小张量拼桶 | 多个小张量按配置的桶大小拼成一个通信单位，单载荷超桶则独占 | `framing.bucketize` |
| 7. 热身 + 逐步提高压缩 | 前 `warmup_steps` 步完全不压缩，之后在 `ramp_steps` 步内线性提高到目标压缩比例 | `schedule.CompressionSchedule` |
| 8. 动量/EF 两种顺序 | `ef_first`（经典 EF-SGD，先 EF 后动量，默认）与 `mom_first`（先动量后压缩）可切换 | `ef.EFBank` |

### 误差反馈（EF）的关键设计

- 残差 `e = v − 本地重建值` 在**通信前于本地**落盘，绝不能用服务端平均值
  （含其他 worker 信息，写进本地误差会破坏无偏性）。
- 服务端把各 worker 的重建值平均后下发，worker 用平均值更新参数，保持数据并行同步。
- `ef_norm_limit`：误差回加范数不超过当前梯度范数的若干倍，超出按比例收缩，
  防止强压缩（低秩/符号）下误差反馈正反馈发散。
- `use_ef=False` 时每步丢弃残差，用于“关掉 EF 明显变差”的对照。

## 目录结构

```
gradcomp/
  tensors.py      张量登记与压缩资格阈值
  schedule.py     热身/爬坡压缩调度
  compressors.py  四种载荷的编解码、跨进程合并、幂迭代
  ef.py           误差累积与动量的两种顺序
  framing.py      二进制分帧与拼桶（字节精确计数）
  transport.py    TCP 服务端/客户端 + 千兆网耗时模型；进程内参照通道
  comm.py         GradientComm：串起一次压缩 allreduce
  models.py       纯 numpy 的 MLP（手写反向传播）
  data.py         固定种子合成数据（含模拟污水处理出水指标数据集）
  training.py     训练循环、RunConfig、单进程参照
  runner.py       多进程编排、实验矩阵、校验
  charts.py       内联 SVG 绘图
  dashboard.py    离线单文件 HTML 仪表盘
tests/            23 个单元/集成测试
```

## 快速开始

依赖只有 numpy（见 `requirements.txt`）。建议用隔离模式运行，避免工作目录
被加入 `sys.path`：

```bash
# 1) 全部单元/集成测试（含多进程 TCP vs 单进程参照的逐位一致性）
PYTHONPATH=. python3 -P -m unittest discover -s tests -v

# 2) 独立校验入口（稀疏合并 + 四种方法的多进程/单进程一致性）
PYTHONPATH=. python3 -P -m gradcomp verify

# 3) 完整实验矩阵（12 组配置 × 每组 4 个真实进程）+ 生成仪表盘
PYTHONPATH=. python3 -P -m gradcomp experiment --out-dir results

# 4) 浏览器打开仪表盘
PYTHONPATH=. python3 -P -m gradcomp serve --dir results --port 8000
# 或直接用浏览器打开 results/dashboard.html（单文件、无外部依赖）

# 快速冒烟（单进程参照，步数短）
PYTHONPATH=. python3 -P -m gradcomp demo --out-dir results_demo
```

> `-P` 禁止隐式 cwd 注入、但保留显式 `PYTHONPATH`；`-I` 会连 `PYTHONPATH`
> 一起忽略，因此本项目命令统一用 `-P`。

## 仪表盘四个图块

1. **每层梯度压缩前后大小**：各层累计线上字节双柱对比 + 压缩比，偏置/小层显示为 dense 100%。
2. **每步通信耗时**：按 1 Gbps / RTT 0.2ms 建模的每步毫秒数，热身步明显更高。
3. **误差累积量范数变化**：开 EF 时残差随训练收敛回落，关 EF 时恒为 0。
4. **不同压缩方法的损失曲线对比**：两个数据集各自一张。
5. 另附最终损失/通信字节汇总表与全部自动化校验结果。

## 校验结论（默认 150 步、固定种子）

- **多进程 vs 单进程参照**：四种方法参数/损失曲线/字节数最大差均为 `0.00e+00`（逐位一致）。
- **EF 精度**：top-k、符号开 EF 后相对不压缩的损失偏差均在 15% 容差内（多数 < 5%）。
- **关 EF 明显变差**：符号关 EF 损失偏差约 +90%（generic）/ +195%（wastewater），低秩约 +59%。
- **通信节省**：符号压缩上行最低可到不压缩的 ~13%，top-k 约 39%~56%，低秩约 25%~47%。
- **低秩的结构性限制**：在 `rank 很小 + 每步仅一次幂迭代` 的硬约束下，大矩阵
  （如 48×24）近似误差偏大，这类运行标记为“非门控展示项”，不参与硬性精度判定，
  但通信节省与多进程一致性仍然成立。

## 千兆网耗时模型

真实 TCP 跑在 loopback 上，实测耗时反映系统调用开销但不反映真实链路带宽；
因此同时给出建模耗时（默认参数可在 `RunConfig` 覆盖）：

```
单步耗时 = RTT + (上行字节 + 下行字节) × 8 / 带宽bps
```

默认带宽 `1e9` bps、RTT `0.2ms`。实测与建模两个数值都记录在 `results.json`。
