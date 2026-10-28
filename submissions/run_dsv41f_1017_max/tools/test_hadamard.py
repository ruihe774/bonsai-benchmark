import numpy as np
from safetensors import safe_open
f = safe_open("data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors", framework="np")
E = f.get_tensor("token_embd.weight").astype(np.float32)
x = E[[248044, 248046]]  # [2,2560]
# debug node_0 sum=-0.415237, node_2 sum=-2.272893, model.input_embed sum=0.378603
# previews of node_2 row0 first3: -0.0035, 0.0210, 0.0579 ; row1: 0.0021,-0.0075,0.0187
# model.input_embed row0 first3: 0.0035,-0.0210,-0.0579; row1: -0.0021,0.0075,-0.0187
i = np.arange(512)
par = np.zeros(512, np.int64)
for b in range(9): par = par + ((i>>b)&1)
H = np.where((par[:,None]^par[None,:])==0, 1.0, -1.0).astype(np.float32)
Hn = H/np.sqrt(512.0)
nov = x.reshape(2,5,512)
for name, M in [("unnorm", H), ("norm", Hn), ("normT", Hn.T), ("unnormT", H.T)]:
    y = np.einsum('ij,bkj->bki', M, nov).reshape(2,2560)
    print(name, "sum", y.sum(), "row0[:3]", np.round(y[0,:3],4), "row1[:3]", np.round(y[1,:3],4),
          "row0[-3:]", np.round(y[0,-3:],4))
print("scale needed: -2.272893 / sum_norm =", -2.272893/ (x.sum()/np.sqrt(512)))
