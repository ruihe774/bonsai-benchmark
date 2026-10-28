import numpy as np, re, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nref
DBG="tools/debug3.txt"
lines = open(DBG, errors='replace').read().splitlines()
name_re = re.compile(r'common_debug_cb_eval:\s+(.*?) = \((f32|i32)\)\s')
def get_preview(name, which=0):
    occ=[i for i,l in enumerate(lines) if (lambda m: m and m.group(1).strip()==name)(name_re.match(l))]
    if len(occ)<=which: return None
    start=occ[which]
    rows=[]
    for l in lines[start+1:start+8000]:
        if name_re.match(l): break
        s=l.strip()
        if s.startswith('sum'): break
        if '[ ...' in l or '...' in s.replace('...,',''): pass
        nums = re.findall(r'[-+]?\d+\.\d+', l)
        if nums: rows.append([float(x) for x in nums])
    return rows

opts = dict(sign_first=True, mt='Tt', emb_mode='inv', conv_rev=False, kmap='repeat',
            q_layout='chunked', gate_act='sigmoid', gdn_gate='silu', gdn_permute=True,
            q_scale=True, decay_first=False, decay_exp=True)
if len(sys.argv)>1: opts.update(json.loads(sys.argv[1]))
m = nref.Model(opts)
x = m.forward([248044,248046], m.new_state(512))

def show(name, arr, label):
    pv = get_preview(name)
    print("==", label, name, "rows", None if pv is None else len(pv))
    if not pv: return
    a = arr.reshape(-1, arr.shape[-1]) if arr.ndim>1 else arr[None]
    for i,got in enumerate(pv):
        t = i // 2 if len(pv) >= 2*len(a) else 0
        if t >= a.shape[0]: t = 0
        mine = np.concatenate([a[t,:3], a[t,-3:]])
        print(f"  row{i} mine={np.round(mine,4).tolist()} dbg={got}")

show("conv_output_silu-0", m.rec["conv_output_silu_0"], "conv_silu")
# raw q,k before scale: q stored scaled in rec; k normalized
q,k = m.rec["qk_l2_0"]
# undo q scale
qs = q / (128 ** -0.5)
print("qk layout test: rows = 3 first + 3 last per token, 32 rows per token")
pv = get_preview("qk_conv_l2-0")
print("pv rows", len(pv))
for t in range(2):
    print(f"-- token {t}")
    for j,h in enumerate([0,1,2,29,30,31]):
        got = pv[t*6+j] if len(pv)>=12 else pv[j]
        if h<16:
            chunked = np.concatenate([qs[t,h,:3], qs[t,h,-3:]])
            inter_q = np.concatenate([qs[t,h//2,:3], qs[t,h//2,-3:]]) if h % 2 == 0 else None
            inter_k = np.concatenate([k[t,h//2,:3], k[t,h//2,-3:]]) if h % 2 == 1 else None
            alt = inter_q if inter_q is not None else inter_k
        else:
            chunked = np.concatenate([k[t,h-16,:3], k[t,h-16,-3:]])
            hh=(h-16)//2
            alt = np.concatenate([qs[t,hh,:3], qs[t,hh,-3:]]) if (h-16)%2==0 else np.concatenate([k[t,hh,:3], k[t,hh,-3:]])
        print(f"  r{h:2d} chunked={np.round(chunked,4).tolist()} inter={np.round(alt,4).tolist()} dbg={got}")
