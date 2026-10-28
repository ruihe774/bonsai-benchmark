import re
p="/opt/bonsai/gguf-py/gguf/quants.py"
src=open(p).read()
lines=src.splitlines()
idx=[i for i,l in enumerate(lines) if 'TQ1_0' in l or 'TQ2_0' in l or 'Q1_0' in l or 'PTQ' in l or 'PQ2' in l]
print("lines with refs:", idx[:40])
# print functions containing them
for i in idx:
    # find enclosing def upward
    j=i
    while j>0 and not re.match(r'^\s*(def|class) ', lines[j]): j-=1
    print("---", j+1, lines[j])
    for k in range(j, min(j+60, len(lines))):
        print(k+1, lines[k])
    print()
