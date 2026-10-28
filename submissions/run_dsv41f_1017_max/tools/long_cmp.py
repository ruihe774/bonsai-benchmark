import ctypes, subprocess, numpy as np, random
lib = ctypes.CDLL("zig-out/lib/libbonsai.so")
lib.bonsai_last_error.restype = ctypes.c_char_p
lib.bonsai_model_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]; lib.bonsai_model_open.restype = ctypes.c_int32
lib.bonsai_context_create.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p)]; lib.bonsai_context_create.restype = ctypes.c_int32
lib.bonsai_eval.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.c_int32]; lib.bonsai_eval.restype = ctypes.c_int32
lib.bonsai_n_vocab.argtypes = [ctypes.c_void_p]; lib.bonsai_n_vocab.restype = ctypes.c_int32
m = ctypes.c_void_p(); c = ctypes.c_void_p()
lib.bonsai_model_open(b"data/Ternary-Bonsai-2-4B-PTQ1_0.gguf", ctypes.byref(m))
nv = lib.bonsai_n_vocab(m); lib.bonsai_context_create(m, 512, 1, ctypes.byref(c))
random.seed(3)
seq = [random.randrange(1000, 200000) for _ in range(48)]
p = subprocess.Popen(["ref/bin/bonsai-ref"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
def cmd(s):
    p.stdin.write(s + "\n"); p.stdin.flush(); return p.stdout.readline().strip()
cmd("load data/Ternary-Bonsai-2-4B-PTQ1_0.gguf"); cmd("ctx 0 512 vulkan")
arr = (ctypes.c_int32 * len(seq))(*seq)
logits = (ctypes.c_float * nv)()
print("eval", lib.bonsai_eval(c, arr, len(seq), logits, nv), lib.bonsai_last_error())
eng = np.frombuffer(logits, dtype=np.float32).copy()
print(cmd("eval 0 %d %s" % (len(seq), " ".join(map(str, seq)))))
print(cmd("logits 0 tools/ref_long.bin"))
ref = np.fromfile("tools/ref_long.bin", dtype=np.float32)
n = min(len(eng), len(ref)); e = eng[:n]; r = ref[:n]
cos = float(np.dot(e, r) / (np.linalg.norm(e) * np.linalg.norm(r) + 1e-30))
pe = np.exp(e - e.max()); pe /= pe.sum(); pr = np.exp(r - r.max()); pr /= pr.sum()
kl = float(np.sum(pr * np.log((pr + 1e-30) / (pe + 1e-30))))
print("cos %.6f KL %.5f top1 eng %d ref %d" % (cos, kl, int(np.argmax(e)), int(np.argmax(r))))
