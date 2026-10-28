import numpy as np, re, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nref

DBG="tools/debug3.txt"
lines = open(DBG, errors='replace').read().splitlines()
name_re = re.compile(r'common_debug_cb_eval:\s+(.*?) = \((f32|i32)\)\s')
def get_preview(name, maxrows=200):
    start=None
    for i,l in enumerate(lines):
        m=name_re.match(l)
        if m and m.group(1).strip()==name:
            start=i; break
    if start is None: return None
    rows=[];
    for l in lines[start+1:start+6000]:
        if name_re.match(l): break
        s=l.strip()
        if s.startswith('sum'): break
        nums = re.findall(r'[-+]?\d+\.\d+', l)
        if nums: rows.append([float(x) for x in nums])
    return rows

opts = dict(sign_first=True, mt='Tt', emb_mode='inv', conv_rev=False, kmap='repeat',
            q_layout='chunked', gate_act='sigmoid', gdn_gate='silu', gdn_permute=True,
            q_scale=True, decay_first=False, decay_exp=True)
if len(sys.argv)>1: opts.update(json.loads(sys.argv[1]))
m = nref.Model(opts)
x = m.forward([248044,248046], m.new_state(512))

def cmp(name, arr, nh, hd, label):
    pv = get_preview(name)
    print("==",label,name,"rows", None if pv is None else len(pv), "sum", float(np.sum(arr)))
    if not pv: return 0
    a = arr.reshape(-1, nh, hd)
    heads = [0,1,2,nh-3,nh-2,nh-1] if len(pv) >= 12 else list(range(12))
    n=0; tot=0; maxerr=0
    for i,got in enumerate(pv):
        t = i // len(heads); h = heads[i % len(heads)]
        if t >= a.shape[0] or len(got)<6: continue
        mine=np.concatenate([a[t,h,:3],a[t,h,-3:]])
        err=np.abs(mine-np.array(got)).max()
        tot+=1; n+= err<1e-3; maxerr=max(maxerr,err)
        if err >= 1e-3 and tot < 30:
            print(f"   t{t} h{h} mine={np.round(mine,4).tolist()} dbg={got}")
    print(f"   rows compared {tot} match(<1e-3) {n} maxerr {maxerr:.5f}")

cmp("node_60", m.rec["gdn_normed_0"], 32, 128, "normed")
cmp("node_61", m.rec["final_output_0"], 32, 128, "swiglu")
