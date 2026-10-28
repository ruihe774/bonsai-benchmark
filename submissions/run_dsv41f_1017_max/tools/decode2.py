import numpy as np, struct, itertools
from safetensors import safe_open
import gguf

r = gguf.GGUFReader("data/Ternary-Bonsai-2-4B-PTQ1_0.gguf")
t = [x for x in r.tensors if x.name == "blk.0.attn_gate.weight"][0]
raw = np.asarray(t.data).tobytes()
f = safe_open("data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors", framework="np")
arr = f.get_tensor("blk.0.attn_gate.weight").astype(np.float32).reshape(-1)
NB=20
Q=np.zeros((NB,128),np.int8)
for blk in range(NB):
    seg=arr[blk*128:(blk+1)*128]
    s=struct.unpack('<e', raw[blk*28+26:blk*28+28])[0]
    Q[blk]=(np.isclose(seg,s,rtol=0,atol=0).astype(np.int8)-np.isclose(seg,-s,rtol=0,atol=0).astype(np.int8))

def digits5(b):
    v=(int(b)*243)>>8
    return [(v//(3**(4-p)))%3 for p in range(5)]
def digits4(b):
    v=(int(b)*81)>>8
    return [(v//(3**(3-p)))%3 for p in range(4)]

def build(blk):
    bs=raw[blk*28:blk*28+28]
    dA=[digits5(bs[c]) for c in range(24)]
    dB=[digits4(bs[24+c]) for c in range(2)]
    vals=[0]*128
    for c in range(24):
        for p in range(5): vals[p*24+c]=dA[c][p]
    for c in range(2):
        for p in range(4): vals[120+p*2+c]=dB[c][p]
    return np.array(vals,np.int64)-1  # mapping 0->-1,1->0,2->1

P=np.array([build(b) for b in range(NB)])
match=(P==Q)
print("total", match.sum(), "/", match.size)
per=match.mean(axis=0)
print("first 120 mean", per[:120].mean(), "last8", per[120:].mean())
print("per index:", np.round(per,2).tolist())
# check if Q has a permutation relation: for each block, find mapping by matching sorted multisets
for blk in range(3):
    print("blk",blk)
    print(" pred", P[blk].tolist())
    print(" true", Q[blk].tolist())
