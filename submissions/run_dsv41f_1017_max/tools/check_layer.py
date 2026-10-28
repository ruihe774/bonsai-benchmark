import numpy as np, re, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nref

DBG = sys.argv[1] if len(sys.argv) > 1 else "tools/debug3.txt"
TOKENS = [248044, 248046]
opts = dict(sign_first=True, mt='Tt', emb_mode='inv', conv_rev=False, kmap='repeat',
            q_layout='chunked', gate_act='sigmoid', gdn_gate='silu', gdn_permute=True,
            q_scale=True, decay_first=False, decay_exp=True)
if len(sys.argv) > 2:
    opts.update(json.loads(sys.argv[2]))

# parse debug
lines = open(DBG, errors='replace').read().splitlines()
entries = {}
cur = None
for i, l in enumerate(lines):
    m = re.match(r'common_debug_cb_eval:\s+(.*?) = \((f32|i32)\)\s', l)
    if m:
        cur = m.group(1).strip()
        entries[cur] = {'line': i, 'sum': None, 'preview': []}
        continue
    if cur is None: continue
    sm = re.match(r'\s+sum = ([-0-9.eE+]+)', l)
    if sm and entries[cur]['sum'] is None and not l.strip().startswith('['):
        entries[cur]['sum'] = float(sm.group(1))
    nums = re.findall(r'[-+]?\d+\.\d+', l)
    if nums and len(entries[cur]['preview']) < 8 and 'sum =' not in l:
        entries[cur]['preview'].append([float(x) for x in nums])

m = nref.Model(opts)
st = m.new_state(512)
x = m.forward(TOKENS, st)

def gsum(a):
    return float(np.sum(a.astype(np.float64)))

def report(name, arr, dbgname=None):
    dn = dbgname or name
    e = entries.get(dn)
    if e is None:
        print(f"{name:24s} MISSING in debug")
        return
    s = gsum(arr)
    ds = e['sum']
    d = abs(s - ds)
    rel = d / (abs(ds) + 1e-6)
    flat = np.asarray(arr).reshape(arr.shape[0], -1)
    row0 = flat[0]
    pv = e['preview'][0] if e['preview'] else []
    pvs = pv[:3] + pv[-3:] if len(pv) >= 6 else pv
    mine = list(np.round(row0[:3], 4)) + list(np.round(row0[-3:], 4))
    print(f"{name:24s} sum={s:14.6f} dbg={ds:14.6f} rel={rel:.2e} preview={mine} dbg={pvs}")

for il in (0, 1, 3):
    print("==== layer", il)
    report(f"attn_norm_{il}", m.rec[f"attn_norm_{il}"], f"attn_norm-{il}")
    if f"conv_output_silu_{il}" in m.rec:
        report(f"conv_output_silu_{il}", m.rec[f"conv_output_silu_{il}"], f"conv_output_silu-{il}")
        report(f"gate_{il}", m.rec[f"gate_{il}"], f"gate-{il}")
        report(f"beta_sigmoid_{il}", m.rec[f"beta_sigmoid_{il}"], f"beta_sigmoid-{il}")
        report(f"attn_output_{il}", m.rec[f"attn_output_{il}"], f"attn_output-{il}")
        report(f"final_output_{il}", m.rec[f"final_output_{il}"], f"final_output-{il}")
        report(f"linear_attn_out_{il}", m.rec[f"linear_attn_out_{il}"], f"linear_attn_out-{il}")
    else:
        report(f"Qcur_full_{il}", m.rec[f"Qcur_full_{il}"], f"Qcur_full-{il}")
        report(f"Qcur_normed_{il}", m.rec[f"Qcur_normed_{il}"], f"Qcur_normed-{il}")
        report(f"Qcur_{il}", m.rec[f"Qcur_{il}"], f"Qcur-{il}")
        report(f"Kcur_normed_{il}", m.rec[f"Kcur_normed_{il}"], f"Kcur_normed-{il}")
    report(f"l_out_{il}", m.rec.get(f"l_out_{il}", np.zeros(1)), f"l_out-{il}")
