import numpy as np, struct, itertools
from safetensors import safe_open
import gguf

r = gguf.GGUFReader("data/Ternary-Bonsai-2-4B-PTQ1_0.gguf")
t = [x for x in r.tensors if x.name == "blk.0.attn_gate.weight"][0]
raw = np.asarray(t.data).tobytes()
f = safe_open("data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors", framework="np")
arr = f.get_tensor("blk.0.attn_gate.weight").astype(np.float32).reshape(-1)

NB = 50
Q = np.zeros((NB,128), np.int8)
for blk in range(NB):
    seg = arr[blk*128:(blk+1)*128]
    s = struct.unpack('<e', raw[blk*28+26:blk*28+28])[0]
    Q[blk] = (np.isclose(seg,s,rtol=0,atol=0).astype(np.int8)
              - np.isclose(seg,-s,rtol=0,atol=0).astype(np.int8))

def digits5(b):
    v = (int(b)*243)>>8
    return [(v//(3**(4-p)))%3 for p in range(5)]

def digits4(b):
    v = (int(b)*81)>>8
    return [(v//(3**(3-p)))%3 for p in range(4)]

def build(blk, mode, digit_order):
    bs = raw[blk*28:blk*28+28]
    # qs: bytes 0..23 ; qh bytes 24..25
    dA = [digits5(bs[c]) for c in range(24)]
    dB = [digits4(bs[24+c]) for c in range(2)]
    vals = [None]*128
    if mode == 'stride24_qh_last':
        for c in range(24):
            for p in range(5):
                idx = p*24+c
                vals[idx] = dA[c][p]
        for c in range(2):
            for p in range(4):
                idx = 120 + p*2 + c
                vals[idx] = dB[c][p]
    elif mode == 'stride24_qh_first':
        for c in range(24):
            for p in range(5):
                idx = 8 + p*24 + c
                vals[idx] = dA[c][p]
        for c in range(2):
            for p in range(4):
                idx = p*2 + c
                vals[idx] = dB[c][p]
    elif mode == 'contig_qh_last':
        for c in range(24):
            for p in range(5):
                idx = c*5+p
                vals[idx] = dA[c][p]
        for c in range(2):
            for p in range(4):
                vals[idx] = 0
        for c in range(2):
            for p in range(4):
                vals[120+c*4+p] = dB[c][p]
    elif mode == 'contig_qh_first':
        for c in range(24):
            for p in range(5):
                idx = 8+c*5+p
                vals[idx] = dA[c][p]
        for c in range(2):
            for p in range(4):
                vals[c*4+p] = dB[c][p]
    arrd = np.array(vals, np.int64)
    if digit_order == 'rev':
        arrd = arrd.reshape(-1,5)[:, ::-1].reshape(-1) if mode.startswith('contig') else arrd
    return arrd

def test(name, arrd):
    # try all 6 mappings code->value
    best=(0,None)
    for p in itertools.permutations([-1,0,1]):
        mp=np.array(p)
        for blk in range(min(NB,10)):
            pass
        tot=0
        for blk in range(min(NB,10)):
            pred=mp[build(blk,name[0],name[1])]
            tot += int(np.sum(pred==Q[blk]))
        if tot>best[0]: best=(tot,p)
    print(name, "best", best[0], "/", 128*10, "perm", best[1])

# digit order variants: in digits5, p=0 is most significant (weight 81)
# try forward and reversed assignment by rewriting
def build_v(blk, mode, rev):
    bs = raw[blk*28:blk*28+28]
    dA = [digits5(bs[c]) for c in range(24)]
    dB = [digits4(bs[24+c]) for c in range(2)]
    if rev:
        dA = [d[::-1] for d in dA]
        dB = [d[::-1] for d in dB]
    vals=[0]*128
    if mode=='stride24_qh_last':
        for c in range(24):
            for p in range(5): vals[p*24+c]=dA[c][p]
        for c in range(2):
            for p in range(4): vals[120+p*2+c]=dB[c][p]
    elif mode=='stride24_qh_first':
        for c in range(24):
            for p in range(5): vals[8+p*24+c]=dA[c][p]
        for c in range(2):
            for p in range(4): vals[p*2+c]=dB[c][p]
    elif mode=='contig_qh_last':
        for c in range(24):
            for p in range(5): vals[c*5+p]=dA[c][p]
        for c in range(2):
            for p in range(4): vals[120+c*4+p]=dB[c][p]
    elif mode=='contig_qh_first':
        for c in range(24):
            for p in range(5): vals[8+c*5+p]=dA[c][p]
        for c in range(2):
            for p in range(4): vals[c*4+p]=dB[c][p]
    return np.array(vals,np.int64)

for mode in ['stride24_qh_last','stride24_qh_first','contig_qh_last','contig_qh_first']:
    for rev in (False,True):
        best=(0,None)
        for p in itertools.permutations([-1,0,1]):
            mp=np.array(p); tot=0
            for blk in range(10):
                tot += int(np.sum(mp[build_v(blk,mode,rev)]==Q[blk]))
            if tot>best[0]: best=(tot,p)
        print(mode, "rev" if rev else "fwd", "best",best[0],"/1280 perm",best[1])
