"""通信传输层。

两套实现，上层 GradientComm 不感知差异：

- SocketServer / SocketClient：多进程 + 真实 TCP（loopback）。
  每个 worker 每步新建一条连接；服务端每连接一个线程处理，收齐同一 step 的
  world_size 个请求后合并，再向所有等待者下发，因此每步都有真实的字节收发
  和可计时的系统调用开销。
- InProcessTransport / InProcessClient：单进程参照用。worker 用线程模拟，
  数据不经过网卡，耗时只按字节数用同一套千兆网模型估算，保证单进程参照
  与多进程走完全相同的“合并/分桶/字节计数”代码路径。

千兆网模型（可用参数覆盖）：
    耗时 = RTT + 字节数 * 8 / 带宽bps   （上行、下行分别计算后相加）
默认 1 Gbps、RTT 0.2 ms。
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .compressors import Payload, merge_payloads
from .framing import decode_message, encode_message, recv_frame, send_frame

GBPS = 1e9
DEFAULT_BANDWIDTH_BPS = 1.0 * GBPS
DEFAULT_RTT_SECONDS = 0.2e-3


# --------------------------------------------------------------------------- #
# 计时模型：字节 -> 模拟链路耗时
# --------------------------------------------------------------------------- #
def model_seconds(uplink_bytes: int, downlink_bytes: int,
                  bandwidth_bps: float = DEFAULT_BANDWIDTH_BPS,
                  rtt_seconds: float = DEFAULT_RTT_SECONDS) -> float:
    tx = (uplink_bytes + downlink_bytes) * 8.0 / bandwidth_bps
    return rtt_seconds + tx


# --------------------------------------------------------------------------- #
# 服务端共享的合并逻辑（socket 与 in-process 复用）
# --------------------------------------------------------------------------- #
def _merge_step(messages: list[dict]) -> tuple[list[list[Payload]], list[dict]]:
    """收齐一个 step 的全部 worker 消息，按桶位置对齐后逐张量合并。"""
    n_buckets = len(messages[0]["buckets"])
    world_size = len(messages)
    out_buckets: list[list[Payload]] = []
    per_tensor: list[dict] = []
    for bi in range(n_buckets):
        bucket_msgs = [m["buckets"][bi] for m in messages]
        n_tensors = len(bucket_msgs[0])
        merged_bucket: list[Payload] = []
        for ti in range(n_tensors):
            same_tensor = [bm[ti] for bm in bucket_msgs]
            payload = same_tensor[0]
            raw_sum = sum(p.nbytes() for p in same_tensor)
            merged = merge_payloads(same_tensor, world_size)
            per_tensor.append({
                "tid": merged.tid,
                "kind": payload.kind,
                "shape": list(merged.shape),
                "raw_bytes_per_worker": int(np.prod(merged.shape)) * 4,
                "uplink_bytes": raw_sum,
            })
            merged_bucket.append(merged)
        out_buckets.append(merged_bucket)
    return out_buckets, per_tensor


def encode_result(step: int, world_size: int,
                  buckets: list[list[Payload]], per_tensor: list[dict]) -> bytes:
    frame = encode_message(step, rank=-1, world_size=world_size, buckets=buckets)
    meta = {"step": step, "world_size": world_size, "per_tensor": per_tensor}
    meta_b = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    return struct.pack("<I", len(meta_b)) + meta_b + frame


def decode_result(frame: bytes) -> dict:
    (mlen,) = struct.unpack_from("<I", frame, 0)
    pos = 4
    meta = json.loads(frame[pos:pos + mlen].decode("utf-8"))
    pos += mlen
    msg = decode_message(frame[pos:])
    return {"meta": meta, "buckets": msg["buckets"]}


# --------------------------------------------------------------------------- #
# 真实 TCP 服务端（独立进程或同进程线程）
# --------------------------------------------------------------------------- #
class SocketServer:
    def __init__(self, host: str, port: int, world_size: int) -> None:
        self.host, self.port = host, port
        self.world_size = world_size
        self._sock: socket.socket | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._pending: dict[int, list[dict]] = {}
        self._results: dict[int, bytes] = {}
        self._threads: list[threading.Thread] = []

    def _listen(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(self.world_size * 2)
        self._sock.settimeout(0.2)

    def start_threaded(self) -> None:
        """同进程线程模式（测试用）。"""
        self._listen()
        t = threading.Thread(target=self.serve_forever, daemon=True)
        t.start()
        self._threads.append(t)

    def serve_forever(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            t = threading.Thread(target=self._handle_conn,
                                 args=(conn,), daemon=True)
            t.start()
            self._threads.append(t)

    def _handle_conn(self, conn: socket.socket) -> None:
        try:
            frame = recv_frame(conn)
            msg = decode_message(frame)
            step = msg["step"]
            with self._cond:
                bucket = self._pending.setdefault(step, [])
                bucket.append(msg)
                if len(bucket) < self.world_size:
                    while step not in self._results:
                        self._cond.wait(timeout=5.0)
                        if self._stop.is_set():
                            return
                    result = self._results[step]
                else:
                    messages = sorted(bucket, key=lambda m: m["rank"])
                    buckets, per_tensor = _merge_step(messages)
                    result = encode_result(step, self.world_size,
                                           buckets, per_tensor)
                    self._results[step] = result
                    self._pending.pop(step, None)
                    self._cond.notify_all()
            send_frame(conn, result)
        except (ConnectionError, OSError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def stop(self) -> None:
        with self._cond:
            self._stop.set()
            self._cond.notify_all()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        for t in self._threads:
            t.join(timeout=2.0)


def run_socket_server(host: str, port: int, world_size: int) -> None:
    """multiprocessing 子进程入口。"""
    server = SocketServer(host, port, world_size)
    server._listen()
    server.serve_forever()


# --------------------------------------------------------------------------- #
# 客户端与计时
# --------------------------------------------------------------------------- #
@dataclass
class StepTiming:
    step: int
    uplink_bytes: int
    downlink_bytes: int
    measured_seconds: float
    modeled_seconds: float
    per_tensor: list[dict] = field(default_factory=list)


class SocketClient:
    def __init__(self, host: str, port: int, rank: int,
                 bandwidth_bps: float = DEFAULT_BANDWIDTH_BPS,
                 rtt_seconds: float = DEFAULT_RTT_SECONDS) -> None:
        self.host, self.port, self.rank = host, port, rank
        self.bandwidth_bps = bandwidth_bps
        self.rtt_seconds = rtt_seconds

    def allreduce(self, step: int, world_size: int,
                  buckets: list[list[Payload]]
                  ) -> tuple[list[list[Payload]], StepTiming]:
        frame = encode_message(step, self.rank, world_size, buckets)
        with socket.create_connection((self.host, self.port), timeout=30) as conn:
            t0 = time.perf_counter()
            send_frame(conn, frame)
            result_frame = recv_frame(conn)
            measured = time.perf_counter() - t0
        result = decode_result(result_frame)
        modeled = model_seconds(len(frame), len(result_frame),
                                self.bandwidth_bps, self.rtt_seconds)
        timing = StepTiming(step, len(frame), len(result_frame),
                            measured, modeled, result["meta"]["per_tensor"])
        return result["buckets"], timing

    def close(self) -> None:
        pass


class InProcessClient:
    """单进程参照：线程内调用，走同一套合并代码，耗时只做模型估算。"""

    def __init__(self, transport: "InProcessTransport", rank: int,
                 bandwidth_bps: float = DEFAULT_BANDWIDTH_BPS,
                 rtt_seconds: float = DEFAULT_RTT_SECONDS) -> None:
        self.transport = transport
        self.rank = rank
        self.bandwidth_bps = bandwidth_bps
        self.rtt_seconds = rtt_seconds

    def allreduce(self, step: int, world_size: int,
                  buckets: list[list[Payload]]
                  ) -> tuple[list[list[Payload]], StepTiming]:
        frame = encode_message(step, self.rank, world_size, buckets)
        result_frame = self.transport.exchange(step, self.rank, frame)
        result = decode_result(result_frame)
        modeled = model_seconds(len(frame), len(result_frame),
                                self.bandwidth_bps, self.rtt_seconds)
        timing = StepTiming(step, len(frame), len(result_frame),
                            modeled, modeled, result["meta"]["per_tensor"])
        return result["buckets"], timing

    def close(self) -> None:
        pass


class InProcessTransport:
    """线程安全的进程内“网络”，按 step 收齐 world_size 帧后合并。"""

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self._cond = threading.Condition()
        self._pending: dict[int, list[tuple[int, bytes]]] = {}
        self._results: dict[int, bytes] = {}

    def exchange(self, step: int, rank: int, frame: bytes) -> bytes:
        with self._cond:
            bucket = self._pending.setdefault(step, [])
            bucket.append((rank, frame))
            if len(bucket) < self.world_size:
                while step not in self._results:
                    self._cond.wait()
                return self._results[step]
            bucket.sort(key=lambda x: x[0])
            messages = [decode_message(f) for _, f in bucket]
            buckets, per_tensor = _merge_step(messages)
            result = encode_result(step, self.world_size, buckets, per_tensor)
            self._results[step] = result
            self._pending.pop(step, None)
            self._cond.notify_all()
            return result
