import re, sys
lines = open("tools/debug3.txt", errors='replace').read().splitlines()
name_re = re.compile(r'common_debug_cb_eval:\s+(.*?) = \((f32|i32)\)\s')
want = sys.argv[1] if len(sys.argv)>1 else None
cur=None; sums=[]; out=[]
for i,l in enumerate(lines):
    m = name_re.match(l)
    if m:
        cur = m.group(1).strip(); sums.append([cur, i+1, None])
        continue
    if cur and sums and sums[-1][2] is None:
        sm = re.match(r'\s+sum = ([-0-9.eE+]+)', l)
        if sm: sums[-1][2] = float(sm.group(1))
from collections import defaultdict
d = defaultdict(list)
for nm, ln, s in sums:
    d[nm].append((ln, s))
for nm in (want.split(',') if want else list(d)[:60]):
    print(nm, d.get(nm))
