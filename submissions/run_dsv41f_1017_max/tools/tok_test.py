import ctypes, json, numpy as np
lib = ctypes.CDLL("zig-out/lib/libbonsai.so")
lib.bonsai_last_error.restype = ctypes.c_char_p
lib.bonsai_model_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]; lib.bonsai_model_open.restype = ctypes.c_int32
lib.bonsai_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
lib.bonsai_tokenize.restype = ctypes.c_int32
m = ctypes.c_void_p()
print("open", lib.bonsai_model_open(b"data/Ternary-Bonsai-2-4B-PTQ1_0.gguf", ctypes.byref(m)), lib.bonsai_last_error())
d = json.load(open("TOKENIZER-VECTORS.json"))
ok = 0
buf = (ctypes.c_int32 * 4096)()
for v in d:
    text = v["text"].encode("utf-8")
    need = lib.bonsai_tokenize(m, text, 0, None, 0)
    n = lib.bonsai_tokenize(m, text, 0, buf, need)
    got = [buf[i] for i in range(n)] if n >= 0 else []
    exp = v["ids"]
    good = got == exp
    ok += good
    if not good:
        print("FAIL", repr(v["text"]), "got", got, "exp", exp)
print(f"{ok}/{len(d)} vectors match")
