import ctypes, numpy as np
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
logits = (ctypes.c_float * nv)()
def step(toks):
    arr = (ctypes.c_int32 * len(toks))(*toks)
    rc = lib.bonsai_eval(c, arr, len(toks), logits, nv)
    a = np.frombuffer(logits, dtype=np.float32).copy()
    top = np.argsort(a)[-3:][::-1]
    print(toks, "rc", rc, "top3", top.tolist(), np.round(a[top], 3).tolist())
lib.bonsai_context_reset(c)
step([248044, 248046])
step([198])
step([248045])
step([846])
