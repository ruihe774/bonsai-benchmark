import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nref
opts = dict(sign_first=True, mt='Tt', emb_mode='inv', conv_rev=False, kmap='tile', q_layout='chunked',
            gate_act='sigmoid', gdn_gate='silu', gdn_permute=True, q_scale=True, decay_first=True, decay_exp=True)
ref = nref.Model(opts)
_ = ref.logits(ref.forward([248044, 248046], ref.new_state(512)))
qg = np.asarray(ref.rec["Qcur_full_3"], dtype=np.float64)  # [2, 8192]
sig = lambda v: 1.0/(1.0+np.exp(-v))
chunk_gate = np.concatenate([qg[t, 4096:] for t in range(2)])
perhead_gate = np.concatenate([qg[t].reshape(16, 512)[:, 256:].reshape(-1) for t in range(2)])
print("dump gate_sigmoid sum 609.325073")
print("chunked gate sigmoid sum", sig(chunk_gate).sum(), " gate raw sum", chunk_gate.sum())
print("perhead gate sigmoid sum", sig(perhead_gate).sum(), " gate raw sum", perhead_gate.sum())
# also compare the preview of gate_sigmoid-3 rows if present
lines = open("tools/debug3.txt", errors="replace").read().splitlines()
import re
name_re = re.compile(r"common_debug_cb_eval:\s+(.*?)\s+= \((f32|i32)\)\s")
idx = [i for i, l in enumerate(lines) if (lambda m: m and m.group(1).strip() == "gate_sigmoid-3")(name_re.match(l))]
rows = []
for l in lines[idx[0]+1:idx[0]+80]:
    if name_re.match(l): break
    if "[" in l:
        nums = re.findall(r"[-+]?\d+\.\d+", l)
        if nums: rows.append([float(x) for x in nums])
print("dump gate_sigmoid preview", rows[:3])
print("nref chunked row0", np.round(sig(chunk_gate).reshape(-1, 256)[0][:3], 4).tolist())
print("nref perhead row0", np.round(sig(perhead_gate).reshape(-1, 256)[0][:3], 4).tolist())
