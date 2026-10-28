import numpy as np
from safetensors import safe_open
f = safe_open("data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors", framework="np")
E = f.get_tensor("token_embd.weight").astype(np.float32)  # [248320, 2560]
print(E.shape)
previews = [
 ([-0.0202,0.0202,-0.0202],[-0.0124,-0.0124,0.0000]),
 ([-0.0147,0.0147,-0.0147],[-0.0123,0.0123,-0.0123]),
]
for first,last in previews:
    cand = []
    for v in range(E.shape[0]):
        r = E[v]
        if (abs(r[0]-first[0])<6e-5 and abs(r[1]-first[1])<6e-5 and abs(r[2]-first[2])<6e-5
            and abs(r[-3]-last[0])<6e-5 and abs(r[-2]-last[1])<6e-5 and abs(r[-1]-last[2])<6e-5):
            cand.append(v)
    print(first, last, "candidates", cand[:20], "count", len(cand))
