"""线上二进制分帧与拼桶（需求点6）。

一帧 = 一个 JSON 头 + 若干原始字节缓冲区::

    [uint32 json 字节数][JSON(utf-8)][buf0 原始字节][buf1 原始字节]...

JSON 中记录每个缓冲区的 dtype/形状/长度，接收端零拷贝重建 numpy 数组。
小张量按 ``bucket_bytes`` 拼成一个“桶”，桶是一次通信的计费与耗时单位；
桶大小可在 GradientComm 中配置。
"""

from __future__ import annotations

import json
import struct

import numpy as np

from .compressors import Payload, payload_from_wire

_U32 = struct.Struct("<I")

# numpy dtype -> 线上标记
_DTYPE_CODE = {
    np.dtype(np.float32): "f4",
    np.dtype(np.int32): "i4",
    np.dtype(np.uint8): "u1",
}
_CODE_DTYPE = {v: k for k, v in _DTYPE_CODE.items()}


def _buf_to_wire(arr: np.ndarray) -> tuple[dict, np.ndarray]:
    arr = np.ascontiguousarray(arr)
    return {"dtype": _DTYPE_CODE[arr.dtype], "nbytes": int(arr.nbytes)}, arr


def encode_message(step: int, rank: int, world_size: int,
                   buckets: list[list[Payload]]) -> bytes:
    """把若干桶载荷编码为一帧完整字节。"""
    bucket_headers: list[dict] = []
    buffers: list[np.ndarray] = []
    offset = 0
    for bucket in buckets:
        tensors: list[dict] = []
        bucket_nbytes = 0
        for payload in bucket:
            ph, pbufs = payload.to_wire()
            buf_refs = []
            for b in pbufs:
                meta, raw = _buf_to_wire(b)
                buffers.append(raw)
                buf_refs.append({"offset": offset, **meta})
                offset += raw.nbytes
                bucket_nbytes += raw.nbytes
            ph["buffers"] = buf_refs
            tensors.append(ph)
        bucket_headers.append({"bucket_nbytes": bucket_nbytes, "tensors": tensors})

    header = {
        "step": step, "rank": rank, "world_size": world_size,
        "buckets": bucket_headers,
    }
    head = json.dumps(header, separators=(",", ":")).encode("utf-8")
    frame = bytearray(_U32.pack(len(head)))
    frame.extend(head)
    for b in buffers:
        frame.extend(b.tobytes())
    return bytes(frame)


def _recv_exact(sock, n: int) -> bytearray:
    chunks = bytearray()
    while len(chunks) < n:
        chunk = sock.recv(n - len(chunks))
        if not chunk:
            raise ConnectionError("对端关闭了连接")
        chunks.extend(chunk)
    return chunks


def send_frame(sock, frame: bytes) -> None:
    sock.sendall(_U32.pack(len(frame)) + frame)


def recv_frame(sock) -> bytes:
    (n,) = _U32.unpack(bytes(_recv_exact(sock, 4)))
    return bytes(_recv_exact(sock, n))


def decode_message(frame: bytes) -> dict:
    """解码一帧，返回 {step, rank, world_size, buckets: list[list[Payload]]}。"""
    head_len, = _U32.unpack(frame[:4])
    head_start = 4
    header = json.loads(frame[head_start:head_start + head_len].decode("utf-8"))
    # 缓冲区 offset 是相对“JSON 头之后第一个数据字节”计算的
    data_start = head_start + head_len
    all_bytes = frame[data_start:]

    buckets: list[list[Payload]] = []
    for bucket in header["buckets"]:
        payloads: list[Payload] = []
        for ph in bucket["tensors"]:
            buffers = []
            for ref in ph["buffers"]:
                start, nbytes = ref["offset"], ref["nbytes"]
                dtype = _CODE_DTYPE[ref["dtype"]]
                arr = np.frombuffer(all_bytes[start:start + nbytes],
                                    dtype=dtype).copy()
                buffers.append(arr)
            payloads.append(payload_from_wire(ph, buffers))
        buckets.append(payloads)
    return {
        "step": header["step"],
        "rank": header["rank"],
        "world_size": header["world_size"],
        "buckets": buckets,
    }


def bucketize(payloads: list[Payload], bucket_bytes: int) -> list[list[Payload]]:
    """需求点6：按配置的桶大小把小张量顺序拼桶。

    单个载荷超过桶大小时独占一个桶（不拆分）。
    """
    if bucket_bytes <= 0:
        return [[p] for p in payloads]
    buckets: list[list[Payload]] = []
    cur: list[Payload] = []
    cur_size = 0
    for p in payloads:
        size = p.nbytes()
        if cur and cur_size + size > bucket_bytes:
            buckets.append(cur)
            cur, cur_size = [], 0
        cur.append(p)
        cur_size += size
    if cur:
        buckets.append(cur)
    return buckets
