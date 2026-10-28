import ctypes, subprocess, random, sys
lib = ctypes.CDLL("zig-out/lib/libbonsai.so")
lib.bonsai_last_error.restype = ctypes.c_char_p
lib.bonsai_model_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]; lib.bonsai_model_open.restype = ctypes.c_int32
lib.bonsai_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
lib.bonsai_tokenize.restype = ctypes.c_int32
m = ctypes.c_void_p()
lib.bonsai_model_open(b"data/Ternary-Bonsai-2-4B-PTQ1_0.gguf", ctypes.byref(m))
p = subprocess.Popen(["ref/bin/bonsai-ref"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
def cmd(s):
    p.stdin.write(s + "\n"); p.stdin.flush(); return p.stdout.readline().strip()
cmd("load data/Ternary-Bonsai-2-4B-PTQ1_0.gguf")
fixed = ["Hi!", "  padded text  ", "\r\n\r\ntabs\ttabs\r\n", "<|im_end|>user\n<|im_start|>assistant",
         "ñ combining tilde: mañana", "¡Hola, señor!", "🥑 emoji 🧪 lab",
         "한국어 텍스트와 中文混合", "a b", " x ",
         "don't won't they're I've I'm we'll he'd", "<|im_start|>", "<|im_end|><|im_end|>", "x", "", " ", "  ",
         "line1\nline2\n\n\nline4", "code: `x = f(a[i]);`\n\treturn x;", "1234567890", "3.14159", "---___...",
         "éèêë", "́̂̃", "😀😀", "a\rb\r\nc", "　　",
         "word, word. word!", "  lead", "trail  ", "tabs\there\t", "¿Qué?"]
random.seed(7)
alphabet = "abcXYZ 0123.,;:'\"!?-_()[]{}\n\r\t<|>/\\@#$%^&*~é世🥑̃ "
gen = ["".join(random.choice(alphabet) for _ in range(random.randint(0, 40))) for _ in range(60)]
tests = fixed + gen
buf = (ctypes.c_int32 * 8192)()
fails = 0
for t in tests:
    b = t.encode("utf-8")
    need = lib.bonsai_tokenize(m, b, 0, None, 0)
    n = lib.bonsai_tokenize(m, b, 0, buf, need)
    got = [buf[i] for i in range(n)] if n >= 0 else None
    hexs = b.hex()
    rep = cmd(f"tokenize {hexs}") if hexs else cmd("tokenize ")
    exp = [int(x) for x in rep.split()[2:]] if rep.startswith("ok") else None
    if got != exp:
        fails += 1
        if fails <= 6:
            print("FAIL", repr(t), "got", got, "exp", exp, "need", need, "n", n)
print(f"{len(tests)-fails}/{len(tests)} match")
