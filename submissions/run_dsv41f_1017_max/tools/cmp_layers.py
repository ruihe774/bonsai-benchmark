import numpy as np, re, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nref
lines = open("tools/debug3.txt", errors='replace').read().splitlines()
# parse first eval only: stop at second occurrence of node_60 op line
occ=[i for i,l in enumerate(lines) if re.match(r'common_debug_cb_eval:\s+node_60 = \(f32\)', l)]
lim = occ[1] if len(occ)>1 else len(lines)
op_re = re.compile(r'common_debug_cb_eval:\s+(.*?) = \((f32|i32)\)\s')
sums={}
i=0
while i < lim:
    m=op_re.match(lines[i])
    if m:
        nm=m.group(1).strip()
        for j in range(i+1, min(i+8000, lim)):
            s=lines[j].strip()
            if s.startswith('sum') and '=' in s:
                if nm not in sums:
                    try: sums[nm]=float(s.split('=')[1])
                    except: pass
                break
            if op_re.match(lines[j]): break
    i+=1
opts = dict(sign_first=True, mt='Tt', emb_mode='inv', conv_rev=False, kmap='tile',
            q_layout='chunked', gate_act='sigmoid', gdn_gate='silu', gdn_permute=True,
            q_scale=True, decay_first=True, decay_exp=True)
if len(sys.argv)>1: opts.update(json.loads(sys.argv[1]))
m=nref.Model(opts)
x=m.forward([248044,248046], m.new_state(512))
rec=m.rec
def chk(name, arr):
    if name not in sums: return None
    return (sums[name], float(np.sum(arr)))
checks=[]
checks.append(("model.input_embed", rec["model.input_embed"]))
for il in range(32):
    checks.append((f"attn_norm-{il}", rec[f"attn_norm_{il}"]))
    if il % 4 != 3:
        checks.append((f"attn_output-{il}", rec[f"gdn_raw_{il}"]))
        if il==0:
            checks.append(("node_60", rec["gdn_normed_0"]))
            checks.append(("node_61", rec["final_output_0"]))
        checks.append((f"final_output-{il}", rec[f"final_output_{il}"]))
        checks.append((f"linear_attn_out-{il}", rec[f"linear_attn_out_{il}"]))
    checks.append((f"ffn_swiglu-{il}", rec[f"ffn_swiglu_{il}"]))
    checks.append((f"ffn_out-{il}", rec[f"ffn_out_{il}"]))
    checks.append((f"l_out-{il}", rec[f"l_out_{il}"]))
for name, arr in checks:
    r = chk(name, arr)
    if r is None:
        print(f"{name:24s} MISSING")
        continue
    d, v = r
    rel = abs(d-v)/(abs(d)+1e-9)
    flag = "OK " if rel < 2e-3 else "BAD"
    print(f"{name:24s} dbg={d:14.6f} mine={v:14.6f} rel={rel:.2e} {flag}")
