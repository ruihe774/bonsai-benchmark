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
ref = nref.Model(opts); ref.forward([248044], ref.new_state(512))
lib.bonsai_dbg_stop(-2); lib.bonsai_context_reset(ctx)
t1 = (ctypes.c_int32*1)(248044)
lib.bonsai_eval(ctx, t1, 1, logits, nv)
out = (ctypes.c_float*2560)()
lib.bonsai_dbg_x(ctx, out, 2560)
e = np.frombuffer(out, dtype=np.float32).copy()
r = ref.rec["model.input_embed"][0]
print("engine first3", np.round(e[:3],4).tolist(), "last3", np.round(e[-3:],4).tolist(), "sum", e.sum())
print("nref   first3", np.round(r[:3],4).tolist(), "last3", np.round(r[-3:],4).tolist(), "sum", r.sum())
print("dbg    first3 [0.0035, -0.0210, -0.0579] last3 [0.0159, -0.0178, -0.0004]")
print("cos", float(np.dot(e,r)/(np.linalg.norm(e)*np.linalg.norm(r)+1e-9)), "maxerr", float(np.max(np.abs(e-r))))
