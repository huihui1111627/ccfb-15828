"""gradcomp 的单元/集成测试，纯标准库 unittest，可 `python -m unittest -v` 运行。

覆盖：
- 分帧编解码、拼桶边界
- 三种压缩器的重建、残差与字节数
- 需求点4：各 worker 下标不同的稀疏合并 == 原始梯度平均
- 符号合并、低秩合并、dense 合并
- 压缩资格阈值（偏置/小层不压缩）
- 调度热身与爬坡
- EF 两种顺序的数值正确性 + 关闭 EF 行为
- InProcess 全链路：多“worker”训练后参数一致、损失下降
- 需求点4 端到端：多进程 TCP 与单进程参照逐位一致
"""

from __future__ import annotations

import unittest

import numpy as np

from gradcomp.compressors import (LowRankState, compress_dense,
                                  compress_lowrank, compress_sign,
                                  compress_topk, merge_payloads,
                                  pack_sign_bits, payload_from_wire)
from gradcomp.comm import GradientComm
from gradcomp.data import load_dataset, shard_indices
from gradcomp.ef import EFBank
from gradcomp.framing import bucketize, decode_message, encode_message
from gradcomp.models import MLP, mse_loss
from gradcomp.runner import run_multiprocess, verify_equivalence, \
    verify_sparse_merge
from gradcomp.schedule import CompressionSchedule
from gradcomp.tensors import TensorRegistry
from gradcomp.training import RunConfig, run_inprocess
from gradcomp.transport import InProcessClient, InProcessTransport


def _rng(seed=0):
    return np.random.default_rng(seed)


class FramingTest(unittest.TestCase):
    def test_roundtrip_all_kinds(self):
        g = _rng(1).standard_normal((12, 8)).astype(np.float32)
        p_dense, _ = compress_dense(g, 0)
        p_topk, _ = compress_topk(g, 1, 0.25)
        p_sign, _ = compress_sign(g, 2)
        state = LowRankState(2, seed=3)
        p_lr, _ = compress_lowrank(g, 3, state, 2)
        buckets = bucketize([p_dense, p_topk, p_sign, p_lr], 256)
        frame = encode_message(step=7, rank=2, world_size=4, buckets=buckets)
        msg = decode_message(frame)
        self.assertEqual(msg["step"], 7)
        self.assertEqual(msg["rank"], 2)
        flat = [p for b in msg["buckets"] for p in b]
        for got, want in zip(flat, [p_dense, p_topk, p_sign, p_lr]):
            self.assertEqual(got.kind, want.kind)
            np.testing.assert_allclose(got.to_dense(), want.to_dense(), rtol=1e-6)

    def test_bucketize_limits(self):
        g = np.zeros(100, dtype=np.float32)
        payloads = [compress_dense(g, i)[0] for i in range(5)]
        buckets = bucketize(payloads, bucket_bytes=250)
        # 每个载荷 400B，超过桶大小 -> 各自独占
        self.assertEqual(len(buckets), 5)
        buckets2 = bucketize(payloads, bucket_bytes=1000)
        self.assertEqual(len(buckets2), 3)  # 400+400 | 400+400 | 400
        self.assertEqual(bucketize(payloads, 0), [[p] for p in payloads])


class CompressorTest(unittest.TestCase):
    def setUp(self):
        self.g = _rng(2).standard_normal((20, 15)).astype(np.float32)

    def test_topk_count_and_residual(self):
        p, resid = compress_topk(self.g, 0, 0.1)
        self.assertEqual(len(p.indices), 30)  # ceil(300*0.1)
        self.assertEqual(len(np.unique(p.indices)), 30)
        # 残差 + 重建 == 原张量
        np.testing.assert_allclose(
            p.to_dense().ravel() + resid.ravel(),
            self.g.ravel(), atol=1e-6)
        # 选中的确实是绝对值最大的 k 个
        top_ref = np.sort(np.argpartition(-np.abs(self.g.ravel()), 29)[:30])
        np.testing.assert_array_equal(np.sort(p.indices), top_ref)

    def test_sign_packing(self):
        p, resid = compress_sign(self.g, 0)
        self.assertEqual(p.bits.nbytes, (self.g.size + 7) // 8)
        self.assertEqual(p.nbytes(), p.bits.nbytes + 4)
        np.testing.assert_allclose(p.to_dense() + resid,
                                   self.g, atol=1e-6)
        # 手工验证符号正确性
        signs = np.unpackbits(p.bits)[:self.g.size]
        self.assertTrue(np.all(signs == (self.g.ravel() >= 0)))

    def test_lowrank_one_power_iter(self):
        state = LowRankState(2, seed=9)
        p, resid = compress_lowrank(self.g, 0, state, 2)
        self.assertEqual(p.p.shape, (20, 2))
        self.assertEqual(p.q.shape, (15, 2))
        # 热启动：state.q 被更新为新的 Q
        np.testing.assert_allclose(state.q, p.q)
        np.testing.assert_allclose(p.to_dense() + resid, self.g, atol=1e-6)
        # 低秩近似应比零矩阵显著更接近原矩阵
        self.assertLess(np.linalg.norm(resid), np.linalg.norm(self.g))

    def test_lowrank_zero_matrix(self):
        state = LowRankState(1, seed=5)
        z = np.zeros((8, 8), dtype=np.float32)
        p, resid = compress_lowrank(z, 0, state, 1)  # 不应抛异常
        self.assertEqual(p.to_dense().shape, (8, 8))


class SparseMergeTest(unittest.TestCase):
    """需求点4：各进程挑中的位置不同，合并必须正确。"""

    def test_disjoint_indices_merge(self):
        rng = _rng(11)
        # 手工构造 4 个梯度，让 top-k 下标互不相交
        grads = []
        for r in range(4):
            g = np.zeros((4, 5), dtype=np.float32)
            idx = np.arange(r * 5, r * 5 + 5)
            g.ravel()[idx] = rng.standard_normal(5).astype(np.float32) + 10
            grads.append(g)
        payloads = [compress_topk(g, 0, density=0.25)[0] for g in grads]
        sets = [set(p.indices.tolist()) for p in payloads]
        self.assertEqual(len(set.intersection(*sets)), 0)  # 确实互不相交
        merged = merge_payloads(payloads, 4).to_dense()
        expected = sum(grads) / 4
        np.testing.assert_allclose(merged, expected, atol=1e-6)

    def test_random_merge_matches_average_on_union(self):
        rep = verify_sparse_merge(seed=4)
        self.assertTrue(rep["passed"], rep)
        self.assertLess(rep["shared_index_count"], rep["union_index_count"])

    def test_sign_and_dense_merge(self):
        rng = _rng(3)
        grads = [rng.standard_normal((10, 7)).astype(np.float32)
                 for _ in range(3)]
        signs = [compress_sign(g, 0)[0] for g in grads]
        merged_s = merge_payloads(signs, 3).to_dense()
        expected_s = sum(p.to_dense() for p in signs) / 3
        np.testing.assert_allclose(merged_s, expected_s, atol=1e-6)

        denses = [compress_dense(g, 0)[0] for g in grads]
        merged_d = merge_payloads(denses, 3).to_dense()
        np.testing.assert_allclose(merged_d, sum(grads) / 3, atol=1e-6)

    def test_lowrank_merge(self):
        rng = _rng(6)
        grads = [rng.standard_normal((12, 9)).astype(np.float32)
                 for _ in range(3)]
        payloads = [compress_lowrank(g, 0, LowRankState(2, seed=i), 2)[0]
                    for i, g in enumerate(grads)]
        merged = merge_payloads(payloads, 3).to_dense()
        expected = sum(p.to_dense() for p in payloads) / 3
        np.testing.assert_allclose(merged, expected, atol=1e-6)


class RegistryScheduleTest(unittest.TestCase):
    def test_compression_eligibility(self):
        reg = TensorRegistry([("W1", (32, 16)), ("b1", (16,)),
                              ("W2", (4, 4)), ("b2", (4,))],
                             min_compress_numel=256)
        self.assertTrue(reg.compressible[0])    # 512 元素二维
        self.assertFalse(reg.compressible[1])   # 偏置
        self.assertFalse(reg.compressible[2])   # 16 元素小矩阵
        self.assertFalse(reg.compressible[3])

    def test_schedule_warmup_ramp(self):
        sch = CompressionSchedule(warmup_steps=3, ramp_steps=10,
                                  target_density=0.1)
        for s in range(3):
            self.assertTrue(sch.is_dense_step(s))
            self.assertEqual(sch.topk_density(s), 1.0)
        self.assertAlmostEqual(sch.topk_density(3), 0.91)
        self.assertAlmostEqual(sch.topk_density(12), 0.1)
        self.assertEqual(sch.lowrank_rank(0, 4), 0)
        self.assertEqual(sch.lowrank_rank(12, 4), 4)
        self.assertFalse(sch.use_sign(0))
        self.assertTrue(sch.use_sign(5))


class EFTest(unittest.TestCase):
    def setUp(self):
        self.shape = (8, 6)
        self.rng = _rng(0)

    def _run_steps(self, order, use_ef, n=4):
        bank = EFBank([("W", self.shape)], momentum=0.9,
                      order=order, use_ef=use_ef)
        residuals = []
        for _ in range(n):
            g = self.rng.standard_normal(self.shape)
            v = bank.prepare("W", g)
            hat = np.zeros_like(v)  # 假装压缩把全部信息丢掉
            bank.commit_residual("W", v, hat)
            residuals.append(bank.residual_norm("W"))
        return bank, residuals

    def test_ef_first_accumulates(self):
        bank, norms = self._run_steps("ef_first", True)
        self.assertGreater(norms[-1], 0.0)
        # 残差应在累积（全丢弃场景下误差不断变大）
        self.assertGreater(norms[-1], norms[0])

    def test_ef_off_drops_residual(self):
        bank, norms = self._run_steps("ef_first", False)
        # 关闭 EF 后误差槽始终为 0
        self.assertTrue(all(n == 0.0 for n in norms))

    def test_two_orders_differ(self):
        b1, _ = self._run_steps("ef_first", True)
        b2, _ = self._run_steps("mom_first", True)
        # 同序列下两种顺序的动量/误差状态不应完全相同
        self.assertFalse(np.allclose(b1.states["W"].error,
                                     b2.states["W"].error))

    def test_ef_first_recovers_after_perfect_compression(self):
        bank = EFBank([("W", self.shape)], momentum=0.0,
                      order="ef_first", use_ef=True)
        g = self.rng.standard_normal(self.shape)
        v = bank.prepare("W", g)
        bank.commit_residual("W", v, v)  # 无损压缩 -> 残差为 0
        self.assertEqual(bank.residual_norm("W"), 0.0)


class InProcessEndToEndTest(unittest.TestCase):
    def _cfg(self, method, use_ef=True, order="ef_first"):
        # 60 步与正式实验一致：足够让 EF 的误差补偿收益显现
        return RunConfig(method=method, dataset="generic",
                         layer_sizes=[6, 32, 16, 1], steps=60,
                         batch_size=32, world_size=4, lr=0.05,
                         use_ef=use_ef, ef_order=order)

    def test_workers_stay_in_sync_and_loss_drops(self):
        cfg = self._cfg("topk")
        results = run_inprocess(cfg)
        self.assertEqual(len(results), 4)
        ref = results[0]["final_params"]
        for r in results[1:]:
            for k in ref:
                np.testing.assert_allclose(r["final_params"][k], ref[k],
                                           atol=1e-10)
        first_loss = results[0]["loss_curve"][0]
        last_loss = results[0]["final_train_loss"]
        self.assertLess(last_loss, first_loss * 0.5)

    def test_bytes_reduced_for_all_methods(self):
        none_res = run_inprocess(self._cfg("none"))[0]
        for method in ("topk", "sign", "lowrank"):
            res = run_inprocess(self._cfg(method))[0]
            self.assertLess(res["summary"]["uplink_bytes"],
                            none_res["summary"]["uplink_bytes"],
                            msg=method)

    def test_ef_on_beats_ef_off(self):
        # 低秩压缩的误差反馈收益最明显（单步幂迭代近似粗糙，靠 EF 补偿）
        on = run_inprocess(self._cfg("lowrank", use_ef=True))[0]
        off = run_inprocess(self._cfg("lowrank", use_ef=False))[0]
        self.assertLess(on["final_train_loss"], off["final_train_loss"])
        # 关闭 EF 时误差槽必须恒为 0
        self.assertTrue(all(
            all(t["residual_norm"] == 0.0 for t in rec["tensors"])
            for rec in off["step_records"]))
        # 开启 EF 后误差槽非零（残差确实在被累积与回加）
        self.assertGreater(
            max(s["total_residual_norm"] for s in on["step_records"]), 0.0)

    def test_biases_always_dense(self):
        cfg = self._cfg("topk")
        res = run_inprocess(cfg)[0]
        for rec in res["step_records"]:
            for t in rec["tensors"]:
                if t["name"].startswith("b"):
                    self.assertEqual(t["kind"], "dense")
                    self.assertEqual(t["compressed_bytes"], t["raw_bytes"])


class MultiprocessEquivalenceTest(unittest.TestCase):
    """需求点4 端到端：多进程 TCP 结果 == 单进程参照。"""

    def test_equivalence_all_methods(self):
        for method in ("none", "topk", "sign", "lowrank"):
            with self.subTest(method=method):
                cfg = RunConfig(method=method, dataset="generic",
                                layer_sizes=[6, 32, 16, 1], steps=10,
                                batch_size=32, world_size=3)
                rep = verify_equivalence(cfg)
                self.assertTrue(rep["passed"], rep)


class DeterminismTest(unittest.TestCase):
    def test_same_seed_same_result(self):
        cfg = RunConfig(method="topk", steps=12)
        r1 = run_inprocess(cfg)[0]["loss_curve"]
        r2 = run_inprocess(cfg)[0]["loss_curve"]
        np.testing.assert_array_equal(r1, r2)

    def test_shard_indices_partition(self):
        idx = np.concatenate([shard_indices(10, 4, r) for r in range(4)])
        np.testing.assert_array_equal(np.sort(idx), np.arange(10))


if __name__ == "__main__":
    unittest.main(verbosity=2)
