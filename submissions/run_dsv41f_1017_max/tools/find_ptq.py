import os, re
roots = ["/opt/bonsai", "/usr/local", "/opt"]
seen=set()
for root in roots:
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath in seen: continue
        seen.add(dirpath)
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            if os.path.getsize(p) > 20_000_000: continue
            try:
                data = open(p, 'rb').read()
            except Exception:
                continue
            if b'PTQ1_0' in data or b'ptq1' in data.lower():
                print("HIT", p)
