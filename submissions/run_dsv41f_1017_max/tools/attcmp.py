import sys, os, re, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nref
lines = open("tools/debug3.txt", errors="replace").read().splitlines()
name_re = re.compile(r"common_debug_cb_eval:\s+(.*?)\s+= \((f32|i32)\)\s")
sums = {}
cur = None
for l in lines:
    m = name_re.match(l)
    if m:
        cur = m.group(1).strip(); continue
    s = l.strip()
    if cur and s.startswith("sum = "): sums.setdefault(cur, float(s[6:])); cur = None
opts = dict(sign_first=True, mt='Tt', emb_mode='inv', conv_rev=False, kmap='tile', q_layout='chunked',
            gate_act='sigmoid', gdn_gate='silu', gdn_permute=True, q_scale=True, decay_first=True, decay_exp=True)
ref = nref.Model(opts)
_ = ref.logits(ref.forward([248044, 248046], ref.new_state(512)))
def s(k):
    a = np.asarray(ref.rec[k], dtype=np.float64); return float(a.sum())
for il in (3, 7):
    for dn, rk in [(f"attn_pregate-{il}", f"attn_pregate_{il}"), (f"gate_sigmoid-{il}", f"gate_sigmoid_{il}"),
                   (f"attn_gated-{il}", f"attn_gated_{il}"), (f"attn_out-{il}", f"attn_out_{il}"),
                   (f"ffn_gate-{il}", f"ffn_gate_{il}"), (f"ffn_out-{il}", f"ffn_out_{il}"), (f"l_out-{il}", f"l_out_{il}")]:
        d = sums.get(dn); r = s(rk) if rk in ref.rec else None
        print(f"{dn:22s} dump={d} nref={r}")
print("Qcur_full-3", sums.get("Qcur_full-3"))
a = np.asarray(ref.rec["Qcur_full_3"], dtype=np.float64)
print("nref Qcur_full_3 sum", a.sum(), "first row first6", np.round(a.reshape(-1, a.shape[-1])[0][:6], 4).tolist())
print("nref Qcur_full_3 token0 first4096 sum", a[0, :4096].sum(), "gate sum", a[0, 4096:].sum())
