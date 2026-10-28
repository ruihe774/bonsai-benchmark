import ctypes, time, glob, os, sys
lib = ctypes.CDLL("zig-out/lib/libbonsai.so")
lib.bonsai_last_error.restype = ctypes.c_char_p
lib.bonsai_model_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]; lib.bonsai_model_open.restype = ctypes.c_int32
lib.bonsai_context_create.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p)]; lib.bonsai_context_create.restype = ctypes.c_int32
lib.bonsai_eval.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.c_int32]; lib.bonsai_eval.restype = ctypes.c_int32
lib.bonsai_context_reset.argtypes = [ctypes.c_void_p]; lib.bonsai_context_reset.restype = ctypes.c_int32
lib.bonsai_n_vocab.argtypes = [ctypes.c_void_p]; lib.bonsai_n_vocab.restype = ctypes.c_int32
def vram():
    t = 0
    for f in glob.glob("/sys/class/drm/card*/device/mem_info_vram_used"):
        try: t += int(open(f).read())
        except Exception: pass
    return t
def hwm():
    for l in open("/proc/self/status"):
        if l.startswith("VmHWM"): return l.strip()
    return "?"
v0 = vram()
t0 = time.time()
m = ctypes.c_void_p()
print("open", lib.bonsai_model_open(b"data/Ternary-Bonsai-2-4B-PTQ1_0.gguf", ctypes.byref(m)), lib.bonsai_last_error(), "%.2fs" % (time.time() - t0))
nv = lib.bonsai_n_vocab(m)
v1 = vram()
print("vram after open %.1f MiB (delta %.1f MiB)" % (v1 / 2**20, (v1 - v0) / 2**20))
ctx = ctypes.c_void_p()
print("ctx16384", lib.bonsai_context_create(m, 16384, 1, ctypes.byref(ctx)), lib.bonsai_last_error())
v2 = vram()
print("vram after ctx %.1f MiB (delta %.1f MiB)" % (v2 / 2**20, (v2 - v1) / 2**20), hwm())
logits = (ctypes.c_float * nv)()
toks = (ctypes.c_int32 * 64)(*([198] * 64))
t = time.time(); rc = lib.bonsai_eval(ctx, toks, 64, logits, nv); dt = time.time() - t
print("eval64 rc", rc, "%.2fs -> %.1f tok/s (pp)" % (dt, 64 / dt))
lib.bonsai_context_reset(ctx)
toks2 = (ctypes.c_int32 * 1)(248046)
t = time.time()
for i in range(16):
    rc = lib.bonsai_eval(ctx, toks2, 1, logits, nv)
dt = time.time() - t
print("tg16 rc", rc, "%.2fs -> %.1f tok/s" % (dt, 16 / dt))
print(hwm(), "vram now %.1f MiB" % (vram() / 2**20))
