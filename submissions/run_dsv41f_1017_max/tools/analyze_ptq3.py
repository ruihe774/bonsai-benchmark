import numpy as np, struct
from safetensors import safe_open
import gguf

r = gguf.GGUFReader("data/Ternary-Bonsai-2-4B-PTQ1_0.gguf")
t = [x for x in r.tensors if x.name == "blk.0.attn_gate.weight"][0]
raw = np.asarray(t.data).tobytes()
f = safe_open("data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors", framework="np")
arr = f.get_tensor("blk.0.attn_gate.weight").astype(np.float32).reshape(-1)

NB = 3000
q = np.zeros(NB*128, np.int8)
scales = np.zeros(NB, np.float32)
firsts = np.zeros(NB, np.float32)
bad = 0
for blk in range(NB):
    seg = arr[blk*128:(blk+1)*128]
    s = struct.unpack('<e', raw[blk*28+26:blk*28+28])[0]
    scales[blk] = s
    firsts[blk] = struct.unpack('<e', raw[blk*28:blk*28+2])[0]
    nz = np.isclose(seg, s, rtol=0, atol=0)
    nzneg = np.isclose(seg, -s, rtol=0, atol=0)
    zz = (seg == 0)
    if np.sum(nz)+np.sum(nzneg)+np.sum(zz) != 128:
        bad += 1
        if bad < 3:
            print("bad block", blk, np.unique(seg)[:8], s)
    q[blk*128:(blk+1)*128] = nz.astype(np.int8) - nzneg.astype(np.int8)
print("bad", bad)
print("sum per block unique:", np.unique(np.add.reduceat(q.astype(np.int32), np.arange(0,NB*128,128)))[:20])
print("zeros per block min/max:", np.min(np.sum(q.reshape(NB,128)==0,axis=1)), np.max(np.sum(q.reshape(NB,128)==0,axis=1)))
print("firsts stats", np.min(firsts), np.max(firsts), np.mean(np.abs(firsts)))
print("scales stats", np.min(scales), np.max(scales))
print("corr first/scale", np.corrcoef(np.abs(firsts), np.abs(scales))[0,1])
print("sample firsts", firsts[:10])
print("sample scales", scales[:10])
# relation between first f16 raw bits and the first values?
b0 = np.frombuffer(raw, dtype=np.uint8).reshape(-1,28)[:NB]
print("byte hist per position:")
for i in range(28):
    print(i, np.min(b0[:,i]), np.max(b0[:,i]), np.mean(b0[:,i]))
