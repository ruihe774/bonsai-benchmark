import ctypes, numpy as np, gguf
r = gguf.GGUFReader("data/Ternary-Bonsai-2-4B-PTQ1_0.gguf")
t = [x for x in r.tensors if x.name == "blk.0.ssm_conv1d.weight"][0]
from safetensors import safe_open
s = safe_open("data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors", framework="np")
w = s.get_tensor("blk.0.ssm_conv1d.weight").astype(np.float32)
print("st shape", w.shape, "w[0,:]", w[0, :])
lib = ctypes.CDLL("zig-out/lib/libbonsai.so")
lib.bonsai_model_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
lib.bonsai_model_open.restype = ctypes.c_int32
lib.bonsai_context_create.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p)]
lib.bonsai_context_create.restype = ctypes.c_int32
lib.bonsai_dbg_w.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
lib.bonsai_dbg_w.restype = ctypes.c_int32
m = ctypes.c_void_p(); c = ctypes.c_void_p()
print("open", lib.bonsai_model_open(b"data/Ternary-Bonsai-2-4B-PTQ1_0.gguf", ctypes.byref(m)))
print("ctx", lib.bonsai_context_create(m, 64, 1, ctypes.byref(c)))
rel = int(t.data_offset) - int(r.data_offset)
print("rel bytes", rel)
for off in (rel // 4, rel // 4 - 2, rel // 4 + 1):
    out = (ctypes.c_float * 16)()
    rc = lib.bonsai_dbg_w(off, out, 16)
    print("off", off, "rc", rc, "eng", np.frombuffer(out, dtype=np.float32))
