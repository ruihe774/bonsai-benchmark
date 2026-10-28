import numpy as np
from safetensors import safe_open
f = safe_open("data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors", framework="np")
E = f.get_tensor("token_embd.weight")
print(E.dtype, E.shape)
for v in (248044, 248046, 9419):
    r = E[v].astype(np.float32)
    print(v, "sum", r.sum(), "unique", np.unique(r)[:5], "nuniq", len(np.unique(r)), "first", r[:6], "last", r[-3:])
