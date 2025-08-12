from typing import List
import torch

def encode_kv(X_fp16, group_size=16, headroom=1.01):
    device = X_fp16.device
    X = X_fp16.to(torch.float32, copy=False)
    _, H, S, D = X.shape
    assert D % group_size == 0
    G = D // group_size

    # Reshape
    X4 = X.view(1, H, S, G, group_size)

    # amax per (h,g)
    amax = X4.abs().amax(dim=(0, 2, 4))                                # [H,G]
    # raw scale requirement
    scale_raw = amax.mul_(headroom / 127.0)                             # in-place mul
    # exponent e so that 2^e >= scale_raw
    e_full = torch.ceil(torch.log2(scale_raw.clamp_min_(1e-12))).to(torch.int32)  # [H,G]

    # base + 2-bit delta
    base = torch.median(e_full, dim=1).values.to(torch.int32)           # [H]
    delta = (e_full - base[:, None]).clamp_(-2, 1).to(torch.int8)       # [H,G]

    # Quantize: Q = round(X / 2^e) using ldexp(X, -e)
    e = base[:, None] + delta.to(torch.int32)                           # [H,G]
    quant_scaled = torch.ldexp(X4, -e[None, :, None, :, None])          # float32
    Q4 = torch.round(quant_scaled).clamp_(-127, 127).to(torch.int8)     # int8
    Q = Q4.view(H, S, D).contiguous()

    # Pack metadata
    base_dtype = torch.int8 if (base.abs() <= 127).all() else torch.int16
    base_stored = base.to(base_dtype).cpu()

    # Vectorized 2-bit packing: map {-2,-1,0,1} -> {0,1,2,3} by +2
    codes = (delta.to(torch.int16) + 2).to(torch.uint8).view(-1)        # [H*G]
    n = codes.numel()
    pad = (-n) % 4
    if pad:
        codes = torch.nn.functional.pad(codes, (0, pad))
    codes = codes.view(-1, 4)
    packed_bytes = (codes[:, 0] |
                    (codes[:, 1] << 2) |
                    (codes[:, 2] << 4) |
                    (codes[:, 3] << 6)).to(torch.uint8)
    deltas_packed = bytes(packed_bytes.cpu().numpy().tobytes())

    meta = {
        "H": int(H), "S": int(S), "D": int(D),
        "group_size": int(group_size),
        "headroom_x100": int(round(headroom * 100)),
        "base_exp_dtype": "int8" if base_dtype is torch.int8 else "int16",
        "base_exp": base_stored.numpy().tobytes(),
        "deltas_2bit": deltas_packed,
        "device": str(device),
    }
    return Q, meta

def decode_kv(Q_int8, meta):
    H = meta["H"]; S = meta["S"]; D = meta["D"]
    g = meta["group_size"]; G = D // g
    device = torch.device(meta.get("device", "cpu"))

    # Unpack base
    if meta["base_exp_dtype"] == "int8":
        base = torch.frombuffer(memoryview(meta["base_exp"]), dtype=torch.int8).to(torch.int32)
    else:
        base = torch.frombuffer(memoryview(meta["base_exp"]), dtype=torch.int16).to(torch.int32)
    base = base.view(H).to(device)

    # Unpack 2-bit deltas
    n = H * G
    packed = torch.frombuffer(memoryview(meta["deltas_2bit"]), dtype=torch.uint8)
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8)
    expanded = ((packed.unsqueeze(1) >> shifts) & 0x3).view(-1)[:n].to(device)  # [H*G]
    delta = (expanded.to(torch.int16) - 2).to(torch.int8).view(H, G)            # [-2..1]

    e = base[:, None] + delta.to(torch.int32)                                   # [H,G]

    Q = Q_int8.view(H, S, D).to(device)
    Q4 = Q.view(H, S, G, g).to(torch.float32)
    # Dequantize: X = Q * 2^e  => ldexp(Q, e)
    Xhat = torch.ldexp(Q4, e[None, :, None, :, None]).view(H, S, D)
    return Xhat.unsqueeze(0)

def compress(keys: List[torch.Tensor], values: List[torch.Tensor]):
    compressed_keys = []
    compressed_values = []
    key_metas = []
    value_metas = []
    for k, v in zip(keys, values):
        k_q, k_meta = encode_kv(k)
        v_q, v_meta = encode_kv(v)
        compressed_keys.append(k_q)
        compressed_values.append(v_q)
        key_metas.append(k_meta)
        value_metas.append(v_meta)
    return (compressed_keys, compressed_values, key_metas, value_metas)

def decompress(compressed_keys, compressed_values, key_metas, value_metas):
    keys = []
    values = []
    for k_q, k_meta, v_q, v_meta in zip(compressed_keys, key_metas, compressed_values, value_metas):
        k = decode_kv(k_q, k_meta)
        v = decode_kv(v_q, v_meta)
        keys.append(k)
        values.append(v)
    return keys, values

def measure_compression_ratio(keys: List[torch.Tensor], values: List[torch.Tensor]):
    import numpy as np

    def tensor_size_mb(t: torch.Tensor):
        return t.numel() * t.element_size() / 1024 / 1024

    orig_keys_mb = sum(tensor_size_mb(k) for k in keys)
    orig_values_mb = sum(tensor_size_mb(v) for v in values)
    orig_total_mb = orig_keys_mb + orig_values_mb

    compressed_keys, compressed_values, key_metas, value_metas = compress(keys, values)
    comp_keys_mb = sum(k.numel() * k.element_size() / 1024 / 1024 for k in compressed_keys)
    comp_values_mb = sum(v.numel() * v.element_size() / 1024 / 1024 for v in compressed_values)
    meta_mb = (
        sum(len(m["base_exp"]) + len(m["deltas_2bit"]) for m in key_metas + value_metas) / 1024 / 1024
    )
    comp_total_mb = comp_keys_mb + comp_values_mb + meta_mb

    ratio = comp_total_mb / orig_total_mb if orig_total_mb > 0 else 0

    print(f"Original size: {orig_total_mb:.3f} MB")
    print(f"Compressed size: {comp_total_mb:.3f} MB")
    print(f"Compression ratio: {ratio:.3f}")

    return {
        "original_mb": orig_total_mb,
        "compressed_mb": comp_total_mb,
        "ratio": ratio,
    }
