import ctypes, sys, numpy as np, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nref
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
lib.bonsai_dbg_stop.argtypes = [ctypes.c_int32]
lib.bonsai_dbg_x.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
m = ctypes.c_void_p()
lib.bonsai_model_open(b"data/Ternary-Bonsai-2-4B-PTQ1_0.gguf", ctypes.byref(m))
nv = lib.bonsai_n_vocab(m)
ctx = ctypes.c_void_p()
lib.bonsai_context_create(m, 512, 1, ctypes.byref(ctx))
logits = (ctypes.c_float * nv)()
opts = dict(sign_first=True, mt='Tt', emb_mode='inv', conv_rev=False, kmap='tile',
            q_layout='chunked', gate_act='sigmoid', gdn_gate='silu', gdn_permute=True,
            q_scale=True, decay_first=True, decay_exp=True)
ref = nref.Model(opts)
ref.forward([248044,248046], ref.new_state(512))
toks = (ctypes.c_int32*2)(248044,248046)
out = (ctypes.c_float*2560)()
for L in [0,1,2,3,4,5,7,8,15,31]:
    lib.bonsai_context_reset(ctx)
    lib.bonsai_dbg_stop(L)
    rc = lib.bonsai_eval(ctx, toks, 2, logits, nv)
    if rc != 0:
        print("L", L, "eval rc", rc, lib.bonsai_last_error()); break
    lib.bonsai_dbg_x(ctx, out, 2560)
    eng = np.frombuffer(out, dtype=np.float32).copy()
    r = ref.rec[f"l_out_{L}"][1]
    cos = float(np.dot(eng,r)/(np.linalg.norm(eng)*np.linalg.norm(r)+1e-9))
    err = float(np.max(np.abs(eng-r)))
    print(f"L{L:2d} cos={cos:.6f} maxerr={err:.5f} sum_eng={eng.sum():.4f} sum_ref={r.sum():.4f} sum_dbg={(ref.rec[f'l_out_{L}'][1].sum()):.4f}")
lib.bonsai_dbg_stop(-1)
