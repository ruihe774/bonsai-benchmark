import sys, os, re, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nref
lines = open("tools/debug3.txt", errors="replace").read().splitlines()
name_re = re.compile(r"common_debug_cb_eval:\s+(.*?)\s+= \((f32|i32)\)\s")
info = {}
cur = None
for l in lines:
    m = name_re.match(l)
    if m:
        cur = m.group(1).strip(); info[cur] = {}
        continue
    s = l.strip()
    if cur and s.startswith("sum = "): info[cur]["sum"] = float(s[6:]); cur = None
def prev(name, maxrows=3):
    idx = [i for i, l in enumerate(lines) if (lambda m: m and m.group(1).strip() == name)(name_re.match(l))]
    if not idx: return None
    out = []
    for l in lines[idx[0]+1:idx[0]+200]:
        if name_re.match(l) or l.strip().startswith("sum"): break
        if "[" in l:
            nums = re.findall(r"[-+]?\d+\.\d+", l)
            if nums: out.append([float(x) for x in nums])
        if len(out) >= maxrows: break
    return out
opts = dict(sign_first=True, mt='Tt', emb_mode='inv', conv_rev=False, kmap='tile', q_layout='chunked',
            gate_act='sigmoid', gdn_gate='silu', gdn_permute=True, q_scale=True, decay_first=True, decay_exp=True)
ref = nref.Model(opts)
_ = ref.logits(ref.forward([248044, 248046], ref.new_state(512)))
for nm, key in [("Qcur_full-3", "Qcur_full_3"), ("Qcur_normed-3", "Qcur_normed_3"), ("Kcur_normed-3", "Kcur_normed_3"), ("Qcur-3", "Qcur_3")]:
    a = np.asarray(ref.rec[key], dtype=np.float64)
    print(nm, "dump", info.get(nm, {}).get("sum"), "nref", float(a.sum()), "nref absmean %.5f" % float(np.abs(a).mean()))
    p = prev(nm)
    print("  dump prev", p)
    print("  nref row0", np.round(a.reshape(-1, a.shape[-1])[0][:6], 4).tolist())
