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
        cur = m.group(1).strip()
        continue
    s = l.strip()
    if cur and s.startswith("sum = "):
        sums.setdefault(cur, float(s[6:]))
        cur = None
opts = dict(sign_first=True, mt='Tt', emb_mode='inv', conv_rev=False, kmap='tile', q_layout='chunked',
            gate_act='sigmoid', gdn_gate='silu', gdn_permute=True, q_scale=True, decay_first=True, decay_exp=True)
ref = nref.Model(opts)
_ = ref.logits(ref.forward([248044, 248046], ref.new_state(512)))
pairs = [("model.input_embed", "model.input_embed")]
for il in (0, 1, 2):
    pairs += [(f"attn_norm-{il}", f"attn_norm_{il}"),
              (f"conv_output_silu-{il}", f"conv_output_silu_{il}"),
              (f"gate-{il}", f"gate_{il}"),
              (f"beta_sigmoid-{il}", f"beta_sigmoid_{il}"),
              (f"attn_output-{il}", f"attn_output_{il}"),
              (f"ffn_gate-{il}", f"ffn_gate_{il}"),
              (f"ffn_up-{il}", f"ffn_up_{il}"),
              (f"ffn_swiglu-{il}", f"ffn_swiglu_{il}"),
              (f"ffn_out-{il}", f"ffn_out_{il}"),
              (f"l_out-{il}", f"l_out_{il}")]
for il in (3,):
    pairs += [(f"Qcur_normed-{il}", f"Qcur_normed_{il}"),
              (f"Kcur_normed-{il}", f"Kcur_normed_{il}"),
              (f"Qcur-{il}", f"Qcur_{il}"),
              (f"l_out-{il}", f"l_out_{il}")]
for dn, rk in pairs:
    d = sums.get(dn)
    r = ref.rec.get(rk)
    if d is None or r is None:
        print(f"{dn:24s} dump={d} nref={'NA' if r is None else 'ok'}")
        continue
    rs = float(np.asarray(r, dtype=np.float64).sum())
    print(f"{dn:24s} dump={d:12.5f} nref={rs:12.5f} diff={rs-d:10.5f} rel={abs(rs-d)/(abs(d)+1e-9):.2e}")
# exact-value check of q/k l2 norms of token 0 against dump previews
print("extra dump names:", [k for k in sums if k.endswith("-0") and ("predelta" in k or "new_state" in k)][:10])
for nm in ("q_conv_predelta-0", "k_conv_predelta-0", "new_state-0", "conv_output_raw-0"):
    print(nm, sums.get(nm))
