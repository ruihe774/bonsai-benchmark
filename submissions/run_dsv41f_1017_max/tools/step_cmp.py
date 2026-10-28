import ctypes, subprocess, numpy as np
lib = ctypes.CDLL("zig-out/lib/libbonsai.so")
lib.bonsai_last_error.restype = ctypes.c_char_p
lib.bonsai_model_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]; lib.bonsai_model_open.restype = ctypes.c_int32
lib.bonsai_context_create.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p)]; lib.bonsai_context_create.restype = ctypes.c_int32
lib.bonsai_context_reset.argtypes = [ctypes.c_void_p]; lib.bonsai_context_reset.restype = ctypes.c_int32
lib.bonsai_eval.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.c_int32]; lib.bonsai_eval.restype = ctypes.c_int32
lib.bonsai_n_vocab.argtypes = [ctypes.c_void_p]; lib.bonsai_n_vocab.restype = ctypes.c_int32
m = ctypes.c_void_p(); c = ctypes.c_void_p()
lib.bonsai_model_open(b"data/Ternary-Bonsai-2-4B-PTQ1_0.gguf", ctypes.byref(m))
nv = lib.bonsai_n_vocab(m); lib.bonsai_context_create(m, 512, 1, ctypes.byref(c))
p = subprocess.Popen(["ref/bin/bonsai-ref"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
def cmd(s):
    p.stdin.write(s + "\n"); p.stdin.flush(); return p.stdout.readline().strip()
cmd("load data/Ternary-Bonsai-2-4B-PTQ1_0.gguf"); cmd("ctx 0 512 vulkan")
seq = [248044, 248046, 198, 248045]
lib.bonsai_context_reset(c)
arr = (ctypes.c_int32 * len(seq))(*seq)
logits = (ctypes.c_float * nv)()
print("eval", lib.bonsai_eval(c, arr, len(seq), logits, nv), lib.bonsai_last_error())
eng = np.frombuffer(logits, dtype=np.float32).copy()
print("eng top3", np.argsort(eng)[-3:][::-1].tolist(), np.round(np.sort(eng)[-3:][::-1], 3).tolist())
print(cmd("eval 0 4 248044 248046 198 248045"))
print(cmd("logits 0 tools/ref_step.bin"))
ref = np.fromfile("tools/ref_step.bin", dtype=np.float32)
print("ref top3", np.argsort(ref)[-3:][::-1].tolist(), np.round(np.sort(ref)[-3:][::-1], 3).tolist())
n = min(len(eng), len(ref)); e = eng[:n]; r = ref[:n]
print("cos %.6f" % (float(np.dot(e, r) / (np.linalg.norm(e) * np.linalg.norm(r) + 1e-30))))
