import numpy as np
import gguf

r = gguf.GGUFReader("data/Ternary-Bonsai-2-4B-PTQ1_0.gguf")
t = [x for x in r.tensors if x.name == "blk.0.attn_qkv.weight"][0]
b = np.frombuffer(np.asarray(t.data).tobytes(), dtype=np.uint8)
n = len(b)//28
b = b[:n*28].reshape(n,28)
hit5 = np.unique(b[:,0])
hit4 = np.unique(b[:,24])
print("hit5 count", len(hit5), "hit4 count", len(hit4))
print("hit4", hit4.tolist())
print("hit5 first", hit5[:30].tolist())
missing5 = sorted(set(range(256))-set(hit5.tolist()))
print("missing5", missing5)

def test_map(name, fn, hit, domain):
    got = set(int(fn(v)) for v in domain)
    if got == set(hit.tolist()):
        print("MATCH", name)
        return True
    return False

found=False
for a in range(0, 400):
    if test_map(f"f(v)=(v*256+{a})//243", lambda v,a=a: (v*256+a)//243, hit5, range(243)): found=True; break
if not found: print("no match f1")
for a in range(-300,300):
    if test_map(f"round(v*256/243)+{a}", lambda v,a=a: int(np.floor(v*256/243+0.5))+a, hit5, range(243)): found=True; break
if not found: print("no match f2")
for m in range(240, 280):
    for s in (8,9,10):
        for a in range(0, 1<<s):
            got = set((v*m+a)>>s for v in range(243))
            if got == set(hit5.tolist()):
                print("MATCH mulshift", m, s, a); found=True; break
        if found: break
    if found: break
# 4-trit: domain 81
for a in range(0, 300):
    if test_map(f"g(v)=(v*256+{a})//81", lambda v,a=a: (v*256+a)//81, hit4, range(81)): found=True; break
for a in range(0, 300):
    if test_map(f"g(v)=(v*255+{a})//80", lambda v,a=a: (v*255+a)//80, hit4, range(81)): found=True; break
for m in range(240, 340):
    for s in (8,9,10):
        got = set((v*m)>>s for v in range(81))
        if got == set(hit4.tolist()):
            print("MATCH4 mulshift", m, s)
