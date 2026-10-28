import numpy as np, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nref
opts = dict(sign_first=True, mt='Tt', emb_mode='inv', conv_rev=False, kmap='tile',
            q_layout='chunked', gate_act='sigmoid', gdn_gate='silu', gdn_permute=True,
            q_scale=True, decay_first=True, decay_exp=True)
m = nref.Model(opts)
T=2
st = m.new_state(512)
x = m.forward([248044,248046], st)
q16,k16 = m.rec["qk_l2_0"]      # q scaled, k normalized  [T,16,128]
q = np.tile(q16, (1,2,1))
k = np.tile(k16, (1,2,1))
conv = m.rec["conv_output_silu_0"]  # [T,8192]
v = conv[:,4096:].reshape(T,32,128)
beta = m.rec["beta_sigmoid_0"]  # [T,32]
g = m.rec["gate_0"]             # [T,32]
target = 0.876065
def run(decay_exp=True, decay_first=True, beta_on_k=True, upd_after_decay=True, extra=None):
    S = np.zeros((32,128,128), np.float32)
    outs = np.zeros((T,32,128), np.float32)
    for t in range(T):
        dec = np.exp(g[t]).astype(np.float32) if decay_exp else g[t].astype(np.float32)
        if decay_first:
            S = S * dec[:,None,None]
        dot = np.einsum('hk,hkv->hv', k[t], S)
        delta = v[t] - dot
        if not beta_on_k:
            delta = beta[t][:,None]*v[t] - dot
        else:
            delta = delta * beta[t][:,None]
        if not decay_first:
            S = S * dec[:,None,None]
        S = S + k[t][:,:,None]*delta[:,None,:]
        if extra == 'mult_after':
            S = S * dec[:,None,None]
        outs[t] = np.einsum('hk,hkv->hv', q[t], S)
    return float(outs.sum())
print("target", target)
for de in [True, False]:
    for df in [True, False]:
        for bk in [True, False]:
            for ex in [None, 'mult_after']:
                s = run(de, df, bk, True, ex)
                print(f"decay_exp={de} decay_first={df} beta_on_k={bk} extra={ex} sum={s:.6f} diff={s-target:.6f}")
