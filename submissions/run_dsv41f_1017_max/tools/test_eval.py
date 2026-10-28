import ctypes, sys, numpy as np, time
lib = ctypes.CDLL("zig-out/lib/libbonsai.so")
lib.bonsai_last_error.restype = ctypes.c_char_p
lib.bonsai_model_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
lib.bonsai_model_open.restype = ctypes.c_int32
lib.bonsai_n_vocab.argtypes = [ctypes.c_void_p]; lib.bonsai_n_vocab.restype = ctypes.c_int32
lib.bonsai_context_create.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p)]
lib.bonsai_context_create.restype = ctypes.c_int32
lib.bonsai_eval.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
lib.bonsai_eval.restype = ctypes.c_int32
lib.bonsai_context_reset.argtypes = [ctypes.c_void_p]; lib.bonsai_context_reset.restype = ctypes.c_int32
t0=time.time()
m = ctypes.c_void_p()
rc = lib.bonsai_model_open(b"data/Ternary-Bonsai-2-4B-PTQ1_0.gguf", ctypes.byref(m))
print("open", rc, lib.bonsai_last_error(), "t=%.1fs"%(time.time()-t0))
if rc: sys.exit(1)
nv = lib.bonsai_n_vocab(m)
ctx = ctypes.c_void_p()
rc = lib.bonsai_context_create(m, 512, 1, ctypes.byref(ctx))
print("ctx", rc, lib.bonsai_last_error(), "t=%.1fs"%(time.time()-t0))
if rc: sys.exit(1)
logits = (ctypes.c_float * nv)()
def run(toks, out):
    arr = (ctypes.c_int32 * len(toks))(*toks)
    t=time.time()
    rc = lib.bonsai_eval(ctx, arr, len(toks), logits, nv)
    print("eval", toks, "rc", rc, lib.bonsai_last_error(), "dt=%.2fs"%(time.time()-t))
    if rc: return
    a = np.frombuffer(logits, dtype=np.float32).copy()
    a.tofile(out)
    top = np.argsort(a)[-5:][::-1]
    print("top5", top.tolist(), a[top].tolist())
run([9419], "tools/vk_t1.bin")
lib.bonsai_context_reset(ctx)
run([248044,248046], "tools/vk_a.bin")
