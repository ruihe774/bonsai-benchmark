import os
for root in ["docs", "/opt/bonsai"]:
    for dp, dn, fns in os.walk(root):
        if '/.git' in dp: continue
        for fn in fns:
            p = os.path.join(dp, fn)
            try:
                if os.path.getsize(p) > 60_000_000: continue
                data = open(p,'rb').read()
            except Exception: continue
            for pat in [b'PTQ', b'PQ2', b'TQ1_0', b'ternary', b'Bonsai', b'prism.hadamard']:
                if pat in data:
                    print(p, pat)
