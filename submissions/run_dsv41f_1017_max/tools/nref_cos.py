import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nref
opts = dict(sign_first=True, mt='Tt', emb_mode='inv', conv_rev=False, kmap='tile', q_layout='per_head',
            gate_act='sigmoid', gdn_gate='silu', gdn_permute=True, q_scale=True, decay_first=True, decay_exp=True)
ref = nref.Model(opts)
for toks, orc in [([9419], "tools/oracle_logits_t1_cpu.bin"), ([248044, 248046], "tools/oracle_logits_a.bin")]:
    st = ref.new_state(512)
    out = ref.logits(ref.forward(toks, st))
    x = np.asarray(out, dtype=np.float32).reshape(-1)
    y = np.fromfile(orc, dtype=np.float32)
    n = min(len(x), len(y)); x = x[:n]; y = y[:n]
    c = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30))
    print(toks, "nref cos %.6f" % c, "sum", float(x.sum()), "orcsum", float(y.sum()))
    print("  top5 nref", np.argsort(x)[-5:][::-1].tolist(), np.round(np.sort(x)[-5:][::-1], 3).tolist())
    print("  top5 orc ", np.argsort(y)[-5:][::-1].tolist(), np.round(np.sort(y)[-5:][::-1], 3).tolist())
