import numpy as np
from safetensors import safe_open
f = safe_open("data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors", framework="np")
E = f.get_tensor("token_embd.weight").astype(np.float32)
x = E[[248044, 248046]]  # [2,2560]
X = x.reshape(2,5,512)

# debug node_2 values (ne0 index 0,1,2,509,510,511) for each block, token
dbg = np.array([
 # token0 block0..4 first3 last3
 [[-0.0035,0.0210,0.0579,-0.0028,0.0147,0.0136],
  [-0.0172,-0.0063,0.0080,-0.0108,0.0012,0.0063],
  [0.0097,0.0080,0.0056,-0.0026,0.0150,0.0069],
  [0.0047,-0.0093,-0.0054,-0.0014,-0.0019,-0.0065],
  [-0.0180,0.0168,-0.0049,0.0159,-0.0178,0.0004]],
 [[0.0021,-0.0075,0.0187,-0.0185,0.0125,0.0045],
  [-0.0098,0.0083,-0.0012,-0.0115,0.0167,-0.0053],
  [0.0105,0.0157,-0.0058,0.0058,0.0108,0.0017],
  [0.0046,-0.0035,-0.0128,0.0113,0.0206,0.0080],
  [-0.0015,0.0209,0.0045,0.0042,0.0147,-0.0067]],
], dtype=np.float64)
idxs = [0,1,2,509,510,511]

i = np.arange(512)
par = np.zeros(512, np.int64)
for b in range(9): par = par + ((i>>b)&1)
H = np.where((par[:,None]^par[None,:])==0, 1.0, -1.0)  # [i,j]
bitrev = np.zeros(512, dtype=np.int64)
for v in range(512):
    r=0
    for b in range(9): r = (r<<1) | ((v>>b)&1)
    bitrev[v]=r
gray = i ^ (i>>1)
perms = {
 'id': i, 'bitrev': bitrev, 'gray': gray,
 'invbitrev': np.argsort(bitrev), 'invgray': np.argsort(gray),
}
best=None
for rn, rp in perms.items():
    for cn, cp in perms.items():
        M = H[np.ix_(rp, cp)]
        for scale_name, s in (("1/sqrt", 1/np.sqrt(512.0)), ("1/512", 1/512.0), ("1", 1.0)):
            pred = np.einsum('ij,bkj->bki', M, X)[:, :, :]  # [2,5,512]
            got = pred[:, :, idxs] * s
            err = np.abs(got - dbg).max()
            if best is None or err < best[0]:
                best = (err, rn, cn, scale_name)
            if err < 0.002:
                print("MATCH", rn, cn, scale_name, err)
print("best", best)
