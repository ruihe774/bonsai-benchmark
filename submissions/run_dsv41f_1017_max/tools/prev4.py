import numpy as np, re, sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nref
DBG="tools/debug3.txt"
lines = open(DBG, errors='replace').read().splitlines()
name_re = re.compile(r'common_debug_cb_eval:\s+(.*?) = \((f32|i32)\)\s')
occ_by_name={}
def get_preview(name, which=0):
    occ=[i for i,l in enumerate(lines) if (lambda m: m and m.group(1).strip()==name)(name_re.match(l))]
    if len(occ)<=which: return None
    start=occ[which]
    rows=[]
    for l in lines[start+1:start+8000]:
        if name_re.match(l): break
        s=l.strip()
        if s.startswith('sum'): break
        nums = re.findall(r'[-+]?\d+\.\d+', l)
        if nums: rows.append([float(x) for x in nums])
    return rows, float(lines[[i for i in range(start,start+60) if lines[i].strip().startswith('sum')][0]].split('=')[1])

pv_raw, sum_raw = get_preview("attn_output-0")
pv_normed, _ = get_preview("node_60")
print("raw sum dbg", sum_raw, "rows", len(pv_raw))
heads=[0,1,2,29,30,31]
def score(arr, pv, nh=32, hd=128):
    a=arr.reshape(-1,nh,hd); n=0; tot=0; mx=0
    for i,got in enumerate(pv):
        t=i//6; h=heads[i%6]
        mine=np.concatenate([a[t,h,:3],a[t,h,-3:]])
        e=np.abs(mine-np.array(got)).max(); tot+=1; n+= e<2e-3; mx=max(mx,e)
    return n,tot,mx

base = dict(sign_first=True, mt='Tt', emb_mode='inv', conv_rev=False, kmap='repeat',
            q_layout='chunked', gate_act='sigmoid', gdn_gate='silu', gdn_permute=True,
            q_scale=True, decay_first=False, decay_exp=True)
variants=[]
for kmap in ['repeat','tile']:
    for sf in [True,False]:
        for qscale in [True,False]:
            for decay_first in [False,True]:
                variants.append(dict(base, kmap=kmap, sign_first=sf, q_scale=qscale, decay_first=decay_first))
res=[]
for v in variants:
    m=nref.Model(v)
    x=m.forward([248044,248046], m.new_state(512))
    raw=m.rec["gdn_raw_0"]
    n,tot,mx=score(raw,pv_raw)
    res.append((n,mx,{k:v[k] for k in ['kmap','sign_first','q_scale','decay_first']}, float(np.sum(raw))))
res.sort(key=lambda r:-r[0])
for r in res: print(r)
