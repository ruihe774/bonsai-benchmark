import sys,re
p = sys.argv[1]
pat = sys.argv[2] if len(sys.argv)>2 else None
lines = open(p, errors='replace').read().splitlines()
if pat is None:
    for i,l in enumerate(lines):
        if 'print_info' in l or 'tensor' in l.lower() and ('[' in l):
            print(i+1, l[:250])
else:
    rx = re.compile(pat)
    for i,l in enumerate(lines):
        if rx.search(l):
            print(i+1, l[:300])
