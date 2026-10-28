import numpy as np, struct, itertools
from safetensors import safe_open
import gguf

r = gguf.GGUFReader("data/Ternary-Bonsai-2-4B-PTQ1_0.gguf")
t = [x for x in r.tensors if x.name == "blk.0.attn_gate.weight"][0]
raw = np.asarray(t.data).tobytes()
f = safe_open("data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors", framework="np")
arr = f.get_tensor("blk.0.attn_gate.weight").astype(np.float32).reshape(-1)

NBLK = 30
Q = np.zeros((NBLK,128), np.int8)
for blk in range(NBLK):
    seg = arr[blk*128:(blk+1)*128]
    s = struct.unpack('<e', raw[blk*28+26:blk*28+28])[0]
    Q[blk] = (np.isclose(seg,s,rtol=0,atol=0).astype(np.int8)
              - np.isclose(seg,-s,rtol=0,atol=0).astype(np.int8))

perms = list(itertools.permutations([-1,0,1]))

def score(pred, blk):
    return int(np.sum(pred == Q[blk]))

def try_digits(digits, name, blocks=5):
    # digits: list of ints 0..2 length>=128, aligned at start
    for p in perms:
        mp = np.array(p)
        tot = 0
        for blk in range(blocks):
            pass
    return None

# generic: build candidate digit array per block from a function
def test(name, fn, blocks=10):
    best = (0, None)
    for p in perms:
        mp = np.array([p[0],p[1],p[2]])
        ok = 0
        for blk in range(blocks):
            d = fn(blk)  # array of digits len 128
            pred = mp[d]
            ok += int(np.sum(pred == Q[blk]))
        if ok > best[0]:
            best = (ok, p)
    print(f"{name}: best {best[0]}/{128*blocks} perm {best[1]}")
    return best

def data_bytes(blk, lo, hi):
    return raw[blk*28+lo:blk*28+hi]

def bigint_digits(bs, outlen, byteorder='little', msb_first_trits=False):
    N = int.from_bytes(bs, byteorder)
    ds = []
    for i in range(outlen):
        ds.append(N % 3); N //= 3
    if msb_first_trits: ds = ds[::-1]
    return np.array(ds, np.int64)

def byte5_digits(bs, outlen, rev=False, byteorder='little'):
    ds = []
    for b in bs:
        v = b
        for _ in range(5):
            ds.append(v % 3); v //= 3
        if v != 0:
            return None  # invalid
    if len(ds) < outlen: return None
    ds = np.array(ds[:outlen], np.int64)
    if rev: ds = ds[::-1]
    return ds

def word_digits(bs, outlen, wsize=2, byteorder='little', trit_rev=False):
    ds = []
    for i in range(0, len(bs)-wsize+1, wsize):
        v = int.from_bytes(bs[i:i+wsize], byteorder)
        for _ in range(10 if wsize==2 else 0):
            ds.append(v % 3); v //= 3
    if len(ds) < outlen: return None
    ds = np.array(ds[:outlen], np.int64)
    if trit_rev: ds = ds[::-1]
    return ds

# test bigint over several ranges
for lo, hi in [(0,26),(2,28),(0,28),(0,24),(2,26),(4,28)]:
    for bo in ('little','big'):
        for rev in (False, True):
            try:
                test(f"bigint[{lo}:{hi}] {bo} rev={rev}", lambda b, lo=lo, hi=hi, bo=bo, rev=rev: bigint_digits(data_bytes(b,lo,hi),128,bo,rev))
            except Exception as e:
                pass
