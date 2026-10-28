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
lib.bonsai_dbg_x.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
lib.bonsai_dbg_x.restype = ctypes.c_int32
m = ctypes.c_void_p(); lib.bonsai_model_open(b"data/Ternary-Bonsai-2-4B-PTQ1_0.gguf", ctypes.byref(m))
nv = lib.bonsai_n_vocab(m); ctx = ctypes.c_void_p(); lib.bonsai_context_create(m, 512, 1, ctypes.byref(ctx))
logits = (ctypes.c_float * nv)()
opts = dict(sign_first=True, mt='Tt', emb_mode='inv', conv_rev=False, kmap='tile', q_layout='chunked',
            gate_act='sigmoid', gdn_gate='silu', gdn_permute=True, q_scale=True, decay_first=True, decay_exp=True)
ref = nref.Model(opts); ref.forward([248044], ref.new_state(512))
t1 = (ctypes.c_int32*1)(248044)
def dump(stop, off, n):
    lib.bonsai_context_reset(ctx); lib.bonsai_dbg_stop(stop)
    rc = lib.bonsai_eval(ctx, t1, 1, logits, nv)
    if rc: print("rc", rc, lib.bonsai_last_error()); return None
    out = (ctypes.c_float*n)()
    lib.bonsai_dbg_x(ctx, off, out, n)
    return np.frombuffer(out, dtype=np.float32).copy()
def cmp(name, e, r):
    if e is None: return
    e = e.reshape(-1); r = np.asarray(r).reshape(-1)
    c = float(np.dot(e[:min(len(e),len(r))], r[:min(len(e),len(r))])/(np.linalg.norm(e[:min(len(e),len(r))])*np.linalg.norm(r[:min(len(e),len(r))])+1e-9))
    print(f"{name:16s} cos={c:.8f} eng[0:4]={np.round(e[:4],4).tolist()} ref[0:4]={np.round(r[:4],4).tolist()} sumE={e.sum():.5f} sumR={r.sum():.5f}")
cmp("attn_norm", dump(-3, 10240//4, 2560), ref.rec["attn_norm_0"][0])
cmp("hf", dump(-4, 20480//4, 2560), ref.tf(ref.rec["attn_norm_0"][0]))
cmp("qkv", dump(-5, 30720//4, 8192), None if False else ref.rec["conv_output_silu_0"][0]*0 + np.concatenate([np.zeros(4096), ref.rec["conv_output_silu_0"][0][4096:]]))
# compare qkv projection against manually recomputed reference
hf_ref = ref.tf(ref.rec["attn_norm_0"][0]); W = ref.w("blk.0.attn_qkv.weight")
qkv_ref = hf_ref @ W.T
cmp("qkv_proj", dump(-5, 30720//4, 8192), qkv_ref)
conv_ref = ref.rec["conv_output_silu_0"][0]
cmp("conv_silu", dump(-9, 63488//4, 8192), conv_ref)
cmp("gdn_out", dump(-10, 233728//4, 4096), ref.rec["final_output_0"][0])
cmp("ssm_out", dump(-12, 112896//4, 2560), ref.rec["linear_attn_out_0"][0])
