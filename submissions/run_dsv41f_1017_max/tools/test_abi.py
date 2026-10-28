import ctypes, sys, numpy as np
lib = ctypes.CDLL("zig-out/lib/libbonsai.so")
lib.bonsai_abi_version.restype = ctypes.c_uint32
lib.bonsai_last_error.restype = ctypes.c_char_p
lib.bonsai_model_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
lib.bonsai_model_open.restype = ctypes.c_int32
lib.bonsai_n_vocab.argtypes = [ctypes.c_void_p]; lib.bonsai_n_vocab.restype = ctypes.c_int32
lib.bonsai_n_ctx_train.argtypes = [ctypes.c_void_p]; lib.bonsai_n_ctx_train.restype = ctypes.c_int32
lib.bonsai_tensor_count.argtypes = [ctypes.c_void_p]; lib.bonsai_tensor_count.restype = ctypes.c_int32
lib.bonsai_tensor_info.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int64)]
lib.bonsai_tensor_info.restype = ctypes.c_int32
lib.bonsai_tensor_dequant.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
lib.bonsai_tensor_dequant.restype = ctypes.c_int64
print("abi", lib.bonsai_abi_version())
m = ctypes.c_void_p()
rc = lib.bonsai_model_open(b"data/Ternary-Bonsai-2-4B-PTQ1_0.gguf", ctypes.byref(m))
print("open", rc, lib.bonsai_last_error())
if rc != 0: sys.exit(1)
print("vocab", lib.bonsai_n_vocab(m), "ctx", lib.bonsai_n_ctx_train(m), "ntensors", lib.bonsai_tensor_count(m))
name = ctypes.c_char_p(); gt = ctypes.c_int32(); rank = ctypes.c_int32(); ne = (ctypes.c_int64*4)()
for idx in [0, 60, 100]:
    rc = lib.bonsai_tensor_info(m, idx, ctypes.byref(name), ctypes.byref(gt), ctypes.byref(rank), ne)
    print("tinfo", idx, rc, name.value, gt.value, rank.value, list(ne))
# dequant compare with safetensors
from safetensors import safe_open
f = safe_open("data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors", framework="np")
for nm in [b"token_embd.weight", b"blk.0.attn_qkv.weight", b"blk.3.ffn_down.weight", b"blk.0.ssm_conv1d.weight"]:
    n = ctypes.c_int64(-1)
    ref = f.get_tensor(nm.decode())
    flat = ref.astype(np.float32).reshape(-1)
    # first find tensor index to get count
    cnt = lib.bonsai_tensor_count(m)
    nelem = None
    for idx in range(cnt):
        lib.bonsai_tensor_info(m, idx, ctypes.byref(name), ctypes.byref(gt), ctypes.byref(rank), ne)
        if name.value == nm:
            nelem = int(np.prod([ne[i] for i in range(rank.value)]))
            break
    out = (ctypes.c_float * nelem)()
    got = lib.bonsai_tensor_dequant(m, nm, out, nelem)
    arr = np.frombuffer(out, dtype=np.float32)
    ok = np.array_equal(arr, flat) if nm != b"blk.0.ssm_conv1d.weight" else np.array_equal(arr, flat)
    mx = np.max(np.abs(arr - flat))
    print(nm.decode(), "ret", got, "nelem", nelem, "exact", ok, "maxdiff", mx)
