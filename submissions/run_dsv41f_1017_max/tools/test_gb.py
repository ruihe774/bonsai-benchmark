import ctypes, sys, numpy as np, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nref
lib = ctypes.CDLL("zig-out/lib/libbonsai.so")
lib.bonsai_last_error.restype = ctypes.c_char_p
lib.bonsai_model_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]; lib.bonsai_model_open.restype = ctypes.c_int32
lib.bonsai_n_vocab.argtypes = [ctypes.c_void_p]; lib.bonsai_n_vocab.restype = ctypes.c_int32
lib.bonsai_context_create.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p)]; lib.bonsai_context_create.restype = ctypes.c_int32
lib.bonsai_eval.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.c_int32]; lib.bonsai_eval.restype = ctypes.c_int32
lib.bonsai_context_reset.argtypes = [ctypes.c_void_p]; lib.bonsai_context_reset.restype = ctypes.c_int32
lib.bonsai_dbg_stop.argtypes = [ctypes.c_int32]
lib.bonsai_dbg_x.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float), ctypes.c_int32]; lib.bonsai_dbg_x.restype = ctypes.c_int32
m = ctypes.c_void_p(); lib.bonsai_model_open(b"data/Ternary-Bonsai-2-4B-PTQ1_0.gguf", ctypes.byref(m))
nv = lib.bonsai_n_vocab(m); ctx = ctypes.c_void_p(); lib.bonsai_context_create(m, 512, 1, ctypes.byref(ctx))
logits = (ctypes.c_float * nv)()
opts = dict(sign_first=True, mt='Tt', emb_mode='inv', conv_rev=False, kmap='tile', q_layout='chunked',
            gate_act='sigmoid', gdn_gate='silu', gdn_permute=True, q_scale=True, decay_first=True, decay_exp=True)
ref = nref.Model(opts); ref.forward([248044], ref.new_state(512))
t1 = (ctypes.c_int32 * 1)(248044)
def dump(stop, off, n):
    lib.bonsai_context_reset(ctx); lib.bonsai_dbg_stop(stop)
    rc = lib.bonsai_eval(ctx, t1, 1, logits, nv)
    if rc: print("rc", rc, lib.bonsai_last_error()); return None
    out = (ctypes.c_float * n)()
    lib.bonsai_dbg_x(ctx, off, out, n)
    return np.frombuffer(out, dtype=np.float32).copy()
gate = dump(-9, 112640 // 4, 64)
print("gateE", np.round(gate[:8], 5), "betaE", np.round(gate[32:40], 5))
print("gateR", np.round(ref.rec["gate_0"][0][:8], 5), "betaR", np.round(ref.rec["beta_sigmoid_0"][0][:8], 5))
q2 = dump(-9, 63488 // 4, 4096)
qr, kr = ref.rec["qk_l2_0"]
qr = qr[0].reshape(-1); kr = kr[0].reshape(-1)
print("qE", np.round(q2[:4], 6), "qR", np.round(qr[:4], 6))
print("kE", np.round(q2[2048:2052], 6), "kR", np.round(kr[:4], 6))
print("qcos", float(np.dot(q2[:2048], qr) / (np.linalg.norm(q2[:2048]) * np.linalg.norm(qr) + 1e-9)))
print("kcos", float(np.dot(q2[2048:4096], kr) / (np.linalg.norm(q2[2048:4096]) * np.linalg.norm(kr) + 1e-9)))
gr = ref.rec["gdn_raw_0"][0].reshape(-1)
print("gdnrawR", np.round(gr[:4], 7), "sum", gr.sum())
gn = ref.rec["gdn_normed_0"][0].reshape(-1)
print("gdnnormR[0:4]", np.round(gn[:4], 7))
