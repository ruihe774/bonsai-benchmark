import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nref
m = nref.Model(dict())
toks=[248044,248046]
x = m.forward(toks, m.new_state(512))
ie = m.rec["model.input_embed"]
print("input_embed sum", ie.sum(), "dbg 0.378603")
print("row0 first/last", np.round(ie[0,:3],4), np.round(ie[0,-3:],4), "dbg [0.0035,-0.0210,-0.0579] [0.0159,-0.0178,-0.0004]")
print("row1 first/last", np.round(ie[1,:3],4), np.round(ie[1,-3:],4), "dbg [-0.0021,0.0075,-0.0187] [0.0042,0.0147,0.0067]")
an = m.rec["attn_norm_0"]
print("attn_norm sum", an.sum(), "dbg -80.665733")
print("attn_norm row0 first3", np.round(an[0,:3],4), "dbg [-1.8782,1.4719,1.6893]")
w = m.w("blk.0.attn_norm.weight")
vv = np.mean(ie*ie, axis=-1, keepdims=True)
print("rms", np.sqrt(vv[0,0]), "eps", m.cfg.eps)
n = ie / np.sqrt(vv + m.cfg.eps)
print("norm row0 first3", np.round(n[0,:3],4), "dbg [0.2770,-1.6635,-4.5853]")
print("w first3", w[:3])
print("n*w first3", np.round((n*w)[0,:3],4))
