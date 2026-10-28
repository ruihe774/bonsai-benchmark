import numpy as np, itertools, sys, json, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nref

ORACLE = "tools/oracle_logits_t1_cpu.bin"
oracle = np.fromfile(ORACLE, dtype=np.float32).astype(np.float64)
def sm(x):
    x = x - x.max(); e=np.exp(x); return e/e.sum()
po = sm(oracle)
def metric(lg):
    lg = lg.astype(np.float64)
    cos = float(np.dot(lg,oracle)/(np.linalg.norm(lg)*np.linalg.norm(oracle)))
    top1 = int(np.argmax(lg)==np.argmax(oracle))
    p = sm(lg)
    kl = float(np.sum(po*np.log((po+1e-30)/(p+1e-30))))
    return cos, top1, kl

m = nref.Model({})
st = m.new_state(512)
base = dict(sign_first=True, mt='Tt', emb_mode='inv', conv_rev=False, kmap='repeat', q_layout='per_head', gate_act='sigmoid', gdn_gate='silu')
results=[]
def run(opts, label):
    m.o = opts
    st = m.new_state(512)
    t0=time.time()
    x = m.forward([9419], st)
    lg = m.logits(x[-1:])[0]
    cos, top1, kl = metric(lg)
    results.append((cos, top1, kl, label))
    print(f"{label:55s} cos={cos:.5f} top1={top1} kl={kl:.4f} t={time.time()-t0:.1f}s", flush=True)

variants = []
for sign_first in (True, False):
    for mt in ('Tt','T'):
        o = dict(base); o['sign_first']=sign_first; o['mt']=mt
        variants.append((o, f"sign_first={sign_first} mt={mt}"))
for name in ('emb_mode','conv_rev','kmap','q_layout','gate_act'):
    vals = {'emb_mode':['inv','none','fwd'],'conv_rev':[False,True],'kmap':['repeat','tile'],'q_layout':['per_head','chunked'],'gate_act':['sigmoid','silu']}[name]
    for v in vals:
        if v == base.get(name): continue
        o=dict(base); o[name]=v
        variants.append((o, f"{name}={v}"))

sel = int(sys.argv[1]) if len(sys.argv)>1 else 0
if sel==0:
    for o,l in variants:
        run(o,l)
else:
    o,l = variants[sel-1]
    run(o,l)
print("BEST:", sorted(results, reverse=True)[:5])
