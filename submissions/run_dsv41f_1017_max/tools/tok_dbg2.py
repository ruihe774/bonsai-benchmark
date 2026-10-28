import ctypes
lib = ctypes.CDLL("zig-out/lib/libbonsai.so")
lib.bonsai_last_error.restype = ctypes.c_char_p
lib.bonsai_model_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]; lib.bonsai_model_open.restype = ctypes.c_int32
lib.bonsai_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
lib.bonsai_tokenize.restype = ctypes.c_int32
m = ctypes.c_void_p()
lib.bonsai_model_open(b"data/Ternary-Bonsai-2-4B-PTQ1_0.gguf", ctypes.byref(m))
buf = (ctypes.c_int32 * 4096)()
for text in ["H", "i", "!", "Hi", "Hi!", "A", "A ", " q", "A q", " quick", "hello", "hello!", " hi", "32", "Hi5", "z"]:
    b = text.encode()
    need = lib.bonsai_tokenize(m, b, 0, None, 0)
    n = lib.bonsai_tokenize(m, b, 0, buf, need)
    print(repr(text), "need", need, "n", n, "err", lib.bonsai_last_error(), [buf[i] for i in range(max(0, min(n, 12)))])
