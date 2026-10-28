import ctypes, subprocess, sys, numpy as np
# reference generator
p = subprocess.Popen(["ref/bin/bonsai-ref"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
def cmd(s):
    p.stdin.write(s + "\n"); p.stdin.flush(); return p.stdout.readline().strip()
print(cmd("load data/Ternary-Bonsai-2-4B-PTQ1_0.gguf"))
print(cmd("ctx 0 512 vulkan"))
print(cmd("eval 0 2 248044 248046"))
ref = cmd("gen 0 8")
print("ref gen:", ref)
lib = ctypes.CDLL("zig-out/lib/libbonsai.so")
lib.bonsai_last_error.restype = ctypes.c_char_p
lib.bonsai_model_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]; lib.bonsai_model_open.restype = ctypes.c_int32
lib.bonsai_context_create.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p)]; lib.bonsai_context_create.restype = ctypes.c_int32
lib.bonsai_eval.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.c_int32]; lib.bonsai_eval.restype = ctypes.c_int32
lib.bonsai_generate_greedy.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_int32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_int32)]; lib.bonsai_generate_greedy.restype = ctypes.c_int32
lib.bonsai_n_vocab.argtypes = [ctypes.c_void_p]; lib.bonsai_n_vocab.restype = ctypes.c_int32
m = ctypes.c_void_p(); c = ctypes.c_void_p()
print("open", lib.bonsai_model_open(b"data/Ternary-Bonsai-2-4B-PTQ1_0.gguf", ctypes.byref(m)))
nv = lib.bonsai_n_vocab(m)
print("ctx", lib.bonsai_context_create(m, 512, 1, ctypes.byref(c)))
toks = (ctypes.c_int32 * 2)(248044, 248046)
out = (ctypes.c_int32 * 8)()
r = lib.bonsai_generate_greedy(c, toks, 2, 8, 0, out)
print("eng gen rc", r, [out[i] for i in range(8)])
