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
lib.bonsai_dbg_x.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
m = ctypes.c_void_p(); lib.bonsai_model_open(b"data/Ternary-Bonsai-2-4B-PTQ1_0.gguf", ctypes.byref(m))
nv = lib.bonsai_n_vocab(m); ctx = ctypes.c_void_p(); lib.bonsai_context_create(m, 512, 1, ctypes.byref(ctx))
logits = (ctypes.c_float * nv)()
opts = dict(sign_first=True, mt='Tt', emb_mode='inv', conv_rev=False, kmap='tile', q_layout='chunked',
            gate_act='sigmoid', gdn_gate='silu', gdn_permute=True, q_scale=True, decay_first=True, decay_exp=True)
ref = nref.Model(opts); ref.forward([248044,248046], ref.new_state(512))
out = (ctypes.c_float*2560)()
def cos(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
# single token
lib.bonsai_dbg_stop(0); lib.bonsai_context_reset(ctx)
t1 = (ctypes.c_int32*1)(248044)
lib.bonsai_eval(ctx, t1, 1, logits, nv)
lib.bonsai_dbg_x(ctx, out, 2560); e0 = np.frombuffer(out, dtype=np.float32).copy()
r0 = ref.rec["l_out_0"][0]
print("1tok L0 cos vs ref tok0", cos(e0,r0), "sum", e0.sum(), r0.sum())
# two tokens
lib.bonsai_dbg_stop(0); lib.bonsai_context_reset(ctx)
t2 = (ctypes.c_int32*2)(248044,248046)
lib.bonsai_eval(ctx, t2, 2, logits, nv)
lib.bonsai_dbg_x(ctx, out, 2560); e1 = np.frombuffer(out, dtype=np.float32).copy()
print("2tok L0 cos vs ref tok1", cos(e1, ref.rec["l_out_0"][1]), "vs tok0", cos(e1, ref.rec["l_out_0"][0]))
print("eng", np.round(e1[:8],4).tolist())
print("ref1", np.round(ref.rec["l_out_0"][1][:8],4).tolist())
print("ref0", np.round(ref.rec["l_out_0"][0][:8],4).tolist())
# also dump after full 32 layers (stop=-1 => final X = l_out_31)
lib.bonsai_dbg_stop(-1); lib.bonsai_context_reset(ctx)
lib.bonsai_eval(ctx, t2, 2, logits, nv)
lib.bonsai_dbg_x(ctx, out, 2560); el = np.frombuffer(out, dtype=np.float32).copy()
print("full L31 cos vs ref", cos(el, ref.rec["l_out_31"][1]))
print("eng", np.round(el[:6],4).tolist())
print("ref", np.round(ref.rec["l_out_31"][1][:6],4).tolist())
