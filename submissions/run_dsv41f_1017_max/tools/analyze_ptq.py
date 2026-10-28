import numpy as np, struct, sys
from safetensors import safe_open
import gguf

gguf_path = "data/Ternary-Bonsai-2-4B-PTQ1_0.gguf"
st_path = "data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors"

r = gguf.GGUFReader(gguf_path)
t = [x for x in r.tensors if x.name == "blk.0.attn_gate.weight"][0]
print("shape", t.shape, "type", t.tensor_type, "nbytes", t.n_bytes)
raw = np.asarray(t.data).tobytes()
print("raw len", len(raw), "expect elems", int(np.prod(t.shape)))
# first block 28 bytes
for blk in range(3):
    b = raw[blk*28:(blk+1)*28]
    print("block", blk, b.hex())
    print("  first f16:", struct.unpack('<e', b[0:2]), " last f16:", struct.unpack('<e', b[26:28]))

f = safe_open(st_path, framework="np")
print(f.keys()[:5])
key = "blk.0.attn_gate.weight"
arr = f.get_tensor(key)
print("st shape", arr.shape, arr.dtype)
a = arr.reshape(-1)[:384].astype(np.float32)
for blk in range(3):
    seg = a[blk*128:(blk+1)*128]
    u = np.unique(seg)
    print("block", blk, "unique n", len(u), u[:10])
