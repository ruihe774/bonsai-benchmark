import numpy as np, re, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nref
lines = open("tools/debug3.txt", errors='replace').read().splitlines()
op_re = re.compile(r'common_debug_cb_eval:\s+(.*?) = \((f32|i32)\)\s')
want={"ffn_gate-0","ffn_up-0","ffn_swiglu-0","ffn_out-0","attn_post_norm-0"}
sums={}
i=0
while i < 37400:
    m=op_re.match(lines[i])
    if m and m.group(1).strip() in want and m.group(1).strip() not in sums:
        nm=m.group(1).strip()
        for j in range(i+1, i+2000):
            s=lines[j].strip()
            if s.startswith('sum'):
                sums[nm]=float(s.split('=')[1]); break
            if op_re.match(lines[j]): break
    i+=1
print("dbg", sums)
base = dict(sign_first=True, mt='Tt', emb_mode='inv', conv_rev=False, kmap='tile',
            q_layout='chunked', gate_act='sigmoid', gdn_gate='silu', gdn_permute=True,
            q_scale=True, decay_first=True, decay_exp=True)
variants = [("base",{}), ("swap",{"ffn_swap":True}), ("noffntf",{"ffn_tf":False}),
            ("ffn_mt_T",{"ffn_mt":"T"}), ("ffn_inv",{"ffn_tf":"inv"})]
for name, ex in variants:
    o=dict(base); o.update(ex)
    m=nref.Model(o)
    m.forward([248044,248046], m.new_state(512))
    r=m.rec
    print(name, "postnorm", float(np.sum(r["attn_norm_0"])) if False else "",
          "gate", float(np.sum(r["ffn_gate_0"])), "up", float(np.sum(r["ffn_up_0"])),
          "swiglu", float(np.sum(r["ffn_swiglu_0"])), "out", float(np.sum(r["ffn_out_0"])))
