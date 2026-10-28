import numpy as np
from safetensors import safe_open
f = safe_open("data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors", framework="np")
E = f.get_tensor("token_embd.weight").astype(np.float64)
x = E[[248044, 248046]]
i = np.arange(512)
par = np.zeros(512, np.int64)
for b in range(9): par = par + ((i>>b)&1)
H = np.where((par[:,None]^par[None,:])==0, 1.0, -1.0)
Hn = H/np.sqrt(512.0)
X = x.reshape(2,5,512)
Y = np.einsum('ij,bkj->bki', Hn, X)   # [2,5,512]
for t in range(2):
    for b in range(5):
        print(t, b, np.round(Y[t,b,[0,1,2,509,510,511]],4).tolist(), "sum", Y[t,b].sum().round(4))
print("total sum", Y.sum())
