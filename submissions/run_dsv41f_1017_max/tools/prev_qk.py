import numpy as np, re, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nref
lines = open("tools/debug3.txt", errors='replace').read().splitlines()
name_re = re.compile(r'common_debug_cb_eval:\s+(.*?) = \((f32|i32)\)\s')
def get_preview(name):
    start=None
    for i,l in enumerate(lines):
        m=name_re.match(l)
        if m and m.group(1).strip()==name: start=i; break
    rows=[]
    for l in lines[start+1:start+6000]:
        if name_re.match(l): break
        if l.strip().startswith('sum'): break
        nums = re.findall(r'[-+]?\d+\.\d+', l)
        if nums: rows.append([float(x) for x in nums])
    return rows
pv = get_preview("qk_conv_l2-0")
opts = dict(q_scale=True);
if len(sys.argv)>1: opts.update(json.loads(sys.argv[1]))
m = nref.Model(opts)
x = m.forward([248044,248046], m.new_state(512))
q,k = m.rec["qk_l2_0"]
print("nrows", len(pv))
# rows: token0 [0,1,2,...,29,30,31] -> 6? actually {128,32,2}: 32 rows per token, printed 3 + ... + 3
heads = [0,1,2,29,30,31]
for t in range(2):
    for j,h in enumerate(heads):
        got = pv[t*6+j]
        mine = np.round(np.concatenate([q[t,h,:3], q[t,h,-3:]]),4).tolist()
        minek = np.round(np.concatenate([k[t,h,:3], k[t,h,-3:]]),4).tolist()
        print(f"t{t} row{h:2d} dbg={got} q={mine} k={minek}")
