import numpy as np, struct
from safetensors import safe_open
import gguf

gguf_path = "data/Ternary-Bonsai-2-4B-PTQ1_0.gguf"
st_path = "data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors"
r = gguf.GGUFReader(gguf_path)
t = [x for x in r.tensors if x.name == "blk.0.attn_gate.weight"][0]
raw = np.asarray(t.data).tobytes()
f = safe_open(st_path, framework="np")
arr = f.get_tensor("blk.0.attn_gate.weight").astype(np.float32).reshape(-1)

# q for first 200 blocks
q = np.zeros(128*200, dtype=np.int8)
for blk in range(200):
    seg = arr[blk*128:(blk+1)*128]
    s = struct.unpack('<e', raw[blk*28+26:blk*28+28])[0]
    qq = np.zeros(128, np.int8)
    qq[np.isclose(seg, s, atol=0, rtol=0)] = 1
    qq[np.isclose(seg, -s, atol=0, rtol=0)] = -1
    # verify all match
    err = np.sum(~(np.isclose(seg,0)|np.isclose(seg,s)|np.isclose(seg,-s)))
    if err: print("block", blk, "err", err, np.unique(seg)[:5])
    q[blk*128:(blk+1)*128] = qq

print("counts +:", np.sum(q==1), " -:", np.sum(q==-1), " 0:", np.sum(q==0), "total", q.size)
for blk in range(3):
    print("blk", blk, "q0..47", q[blk*128:blk*128+48].tolist())
    b = raw[blk*28:(blk+1)*28]
    print("  bytes0-1", b[0:2].hex(), " mid", b[2:26].hex(), " last", b[26:28].hex())

# hypothesis: base-3 5 trits per byte, little endian over some byte region
b = raw[0:28]
def trits5(bs):
    out=[]
    for x in bs:
        v=x
        for _ in range(5):
            out.append(v%3-1); v//=3
        # if v!=0 flag
    return out

for start in (0,2):
    tr = trits5(b[start:start+24])
    print("start",start,"match24", np.sum(np.array(tr[:120])==q[:120]), "/120")
    tr2 = trits5(b[start:start+26])
    print("start",start,"match26", np.sum(np.array(tr2[:128])==q[:128]), "/128")
