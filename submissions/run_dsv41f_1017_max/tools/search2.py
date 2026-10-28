import numpy as np, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nref
oracle = np.fromfile("tools/oracle_logits_a.bin", dtype=np.float32).astype(np.float64)
def sm(x):
    x = x - x.max(); e=np.exp(x); return e/e.sum()
po = sm(oracle)
def metric(lg):
    lg = lg.astype(np.float64)
    cos = float(np.dot(lg,oracle)/(np.linalg.norm(lg)*np.linalg.norm(oracle)))
    top1 = int(np.argmax(lg)==np.argmax(oracle))
    p = sm(lg); kl = float(np.sum(po*np.log((po+1e-30)/(p+1e-30))))
    return cos, top1, kl
base = dict(sign_first=True, mt='Tt', emb_mode='inv', conv_rev=False, kmap='repeat',
            q_layout='chunked', gate_act='sigmoid', gdn_gate='silu', gdn_permute=True,
            q_scale=True, decay_first=False, decay_exp=True)
m = nref.Model({})
def run(o, label):
    m.o = o; m.rec = {}
    st = m.new_state(512)
    x = m.forward([248044,248046], st)
    lg = m.logits(x[-1:])[0]
    cos, top1, kl = metric(lg)
    raw = float(m.rec["gdn_raw_0"].sum())
    gated = float(m.rec["final_output_0"].sum())
    print(f"{label:35s} raw={raw:.6f}(dbg 0.876065) gated={gated:.4f}(dbg -19.2170) cos={cos:.5f} kl={kl:.4f}", flush=True)
run(dict(base), "base")
for name, vals in [('q_scale',[False]), ('kmap',['tile']), ('decay_first',[True]),
                   ('decay_exp',[False])]:
    for v in vals:
        o=dict(base); o[name]=v
        run(o, f"{name}={v}")
