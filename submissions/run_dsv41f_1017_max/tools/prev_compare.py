import numpy as np, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nref

DBG="tools/debug3.txt"
lines = open(DBG, errors='replace').read().splitlines()
name_re = re.compile(r'common_debug_cb_eval:\s+(.*?) = \((f32|i32)\)\s')
def get_preview(name, maxrows=64):
    # first occurrence
    start=None
    for i,l in enumerate(lines):
        m=name_re.match(l)
        if m and m.group(1).strip()==name:
            start=i; break
    if start is None: return None
    rows=[]; cur=None
    for l in lines[start+1:start+4000]:
        if name_re.match(l): break
        if 'sum =' in l: break
        s=l.strip()
        if s in ('[','],','],',']'): continue
        nums = re.findall(r'[-+]?\d+\.\d+', l)
        if nums:
            rows.append([float(x) for x in nums])
        if len(rows)>=maxrows: break
    return rows

opts = dict(sign_first=True, mt='Tt', emb_mode='inv', conv_rev=False, kmap='repeat',
            q_layout='chunked', gate_act='sigmoid', gdn_gate='silu', gdn_permute=True,
            q_scale=True, decay_first=False, decay_exp=True)
if len(sys.argv)>1:
    import json; opts.update(json.loads(sys.argv[1]))
m = nref.Model(opts)
x = m.forward([248044,248046], m.new_state(512))

def show(name, arr, nh, hd, label):
    pv = get_preview(name)
    print("==", name, label, "preview rows", None if pv is None else len(pv))
    if pv is None: return
    # arr [T, nh*hd] -> rows per token per head
    a = arr.reshape(arr.shape[0], nh, hd)
    for t in range(a.shape[0]):
        for h in range(min(nh, 40)):
            if len(pv)<= t*nh+h: break
            got = pv[t*nh+h]
            if len(got)<6: continue
            mine = np.round(np.concatenate([a[t,h,:3], a[t,h,-3:]]),4).tolist()
            print(f" t{t} h{h:2d} mine={mine} dbg={got}")
show("attn_output-0", m.rec["gdn_raw_0"].reshape(2,-1), 32, 128, "raw gdn")
show("node_60", None if False else m.rec["gdn_raw_0"].reshape(2,-1), 32, 128, "(placeholder)")
