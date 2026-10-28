"""Numpy reference implementation of the bonsai qwen35 forward pass (dev only)."""
import numpy as np, struct, sys, json
from safetensors import safe_open

ST = "data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors"

class Cfg:
    n_layer=32; n_embd=2560; n_ff=9216; n_head=16; n_head_kv=4; head_dim=256
    rope_dim=64; rope_theta=1e7; eps=1e-6; n_vocab=248320
    conv_k=4; ssm_state=128; n_k_heads=16; n_v_heads=32; inner=4096
    full_attn_interval=4

class Model:
    def __init__(self, opts):
        self.o = opts
        self.f = safe_open(ST, framework="np")
        self.cache = {}
        self.rec = {}
        self.cfg = Cfg()
        self.sign_widths = [2560,4096,9216]
        # sign vectors from GGUF metadata
        import gguf
        r = gguf.GGUFReader("data/Ternary-Bonsai-2-4B-PTQ1_0.gguf")
        f = r.fields
        sv = np.array([int(x) for x in f["prism.hadamard.sign_values"].contents()], dtype=np.float32)
        self.signs = {}
        off=0
        for w in self.sign_widths:
            self.signs[w] = sv[off:off+w].copy(); off+=w
        # H512
        i = np.arange(512)
        par = np.zeros(512, dtype=np.int64)
        for b in range(9):
            par = par + ((i>>b)&1)
        # H[i,j] = (-1)^popcount(i & j): parity of bitwise AND
        pc = np.zeros((512,512), np.int64)
        for b in range(9):
            pc += ((i[:,None] & i[None,:])>>b)&1
        H = np.where((pc & 1)==0, 1.0, -1.0).astype(np.float32)/np.sqrt(512.0)
        self.H = H

    def w(self, name):
        if name not in self.cache:
            a = self.f.get_tensor(name)
            self.cache[name] = np.ascontiguousarray(a.astype(np.float32))
        return self.cache[name]

    def tf(self, x, width=None, inverse=False):
        """Activation transform T (or inverse T^T) for sign vector of given width."""
        width = x.shape[-1]
        if self.o.get('had', True) is False:
            return x
        s = self.signs[width]
        # matmul storage mode: 'Tt' -> W' = W T^T (activation transform T)
        #                    'T'  -> W' = W T   (activation transform T^T)
        if self.o.get('mt', 'Tt') == 'T' and not inverse:
            inverse = True
        elif self.o.get('mt', 'Tt') == 'T' and inverse:
            inverse = False
        if self.o.get('emb_mode') == 'none' and inverse:
            return x
        if self.o.get('emb_mode') == 'fwd' and inverse:
            inverse = False
        sign_first = self.o.get('sign_first', True)
        # T = H S if sign_first else S H
        def fwd(x):
            if sign_first: x = x * s
            x = self.wht(x)
            if not sign_first: x = x * s
            return x
        def inv(x):
            if sign_first: x = self.wht(x) * s   # S H
            else: x = self.wht(x * s)            # H S
            return x
        return inv(x) if inverse else fwd(x)

    def wht(self, x):
        shp = x.shape
        y = x.reshape(-1, shp[-1]//512, 512).copy()
        for b in range(9):
            step = 1<<b
            y = y.reshape(-1, shp[-1]//512, 512//(2*step), 2, step)
            a = y[:,:,:,0,:].copy(); c = y[:,:,:,1,:].copy()
            y[:,:,:,0,:] = a+c; y[:,:,:,1,:] = a-c
            y = y.reshape(-1, shp[-1]//512, 512)
        y *= 1.0/np.sqrt(512.0)
        return y.reshape(shp)

    def rmsnorm(self, x, wname):
        w = self.w(wname)
        v = np.mean(x*x, axis=-1, keepdims=True)
        return x / np.sqrt(v + self.cfg.eps) * w

    def matmul_t(self, x, wname):
        W = self.w(wname)          # [out, in]
        return x @ W.T

    def embed(self, tok):
        E = self.f.get_tensor("token_embd.weight")
        e = E[tok].astype(np.float32)
        return self.tf(e, inverse=True)

    def logits(self, x, toks=None):
        xn = self.rmsnorm(x, "output_norm.weight")
        xf = self.tf(xn)
        E = self.f.get_tensor("token_embd.weight")
        if toks is not None:
            rows = E[toks].astype(np.float32)
            return xf @ rows.T
        return xf @ E.astype(np.float32).T

    def softplus(self, x):
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    def gdn(self, x, il, state):
        o = self.o
        c = self.cfg
        h = self.rmsnorm(x, f"blk.{il}.attn_norm.weight")
        self.rec[f"attn_norm_{il}"] = h.copy()
        hf = self.tf(h)
        qkv = hf @ self.w(f"blk.{il}.attn_qkv.weight").T     # [T, 8192]
        z = hf @ self.w(f"blk.{il}.attn_gate.weight").T       # [T, 4096]
        a = h @ self.w(f"blk.{il}.ssm_alpha.weight").T        # [T, 32]
        b = h @ self.w(f"blk.{il}.ssm_beta.weight").T         # [T, 32]
        Tn = qkv.shape[0]
        # conv1d over channels, causal kernel 4
        cw = self.w(f"blk.{il}.ssm_conv1d.weight").reshape(c.inner*2, c.conv_k)  # [8192,4]
        hist = state['conv']   # [conv_k-1, 8192]
        seq = np.concatenate([hist, qkv], axis=0)  # [T+3, 8192]
        out = np.zeros_like(qkv)
        # out[t] = sum_k cw[:,k]*seq[t+k]
        for k in range(c.conv_k):
            kk = (c.conv_k-1-k) if o.get('conv_rev') else k
            out += seq[k:k+Tn] * cw[:, kk][None, :]
        state['conv'] = seq[-(c.conv_k-1):].copy()
        qkv = out / (1.0 + np.exp(-out))  # silu
        self.rec[f"conv_output_silu_{il}"] = qkv.copy()
        q = qkv[:, :2048].reshape(Tn, c.n_k_heads, 128)
        k = qkv[:, 2048:4096].reshape(Tn, c.n_k_heads, 128)
        v = qkv[:, 4096:].reshape(Tn, c.n_v_heads, 128)
        qn = np.sqrt(np.sum(q*q, axis=-1, keepdims=True)) + 1e-6
        kn = np.sqrt(np.sum(k*k, axis=-1, keepdims=True)) + 1e-6
        q = q / qn * ((128 ** -0.5) if o.get('q_scale', True) else 1.0)
        k = k / kn
        beta = 1.0/(1.0+np.exp(-b))
        ssm_a = self.w(f"blk.{il}.ssm_a")
        dtb = self.w(f"blk.{il}.ssm_dt.bias")
        g = ssm_a[None,:] * self.softplus(a + dtb[None,:])   # [T,32]
        self.rec[f"gate_{il}"] = g.copy()
        self.rec[f"beta_sigmoid_{il}"] = beta.copy()
        self.rec[f"alpha_{il}"] = a.copy()
        self.rec[f"qk_l2_{il}"] = (q, k)
        # expand k to v heads
        rep = c.n_v_heads // c.n_k_heads
        kmap = o.get('kmap', 'repeat')
        if kmap == 'repeat':
            kk = np.repeat(k, rep, axis=1)  # [T,32,128]
            qq = np.repeat(q, rep, axis=1)
        else:
            kk = np.tile(k, (1, rep, 1))
            qq = np.tile(q, (1, rep, 1))
        S = state['S']  # [32,128,128]
        outs = np.zeros((Tn, c.n_v_heads, 128), dtype=np.float32)
        decay_first = o.get('decay_first', False)
        for t in range(Tn):
            gt = np.exp(g[t])[:, None, None] if o.get('decay_exp', True) else g[t][:,None,None]  # [32,1,1]
            kt = kk[t]  # [32,128]
            if decay_first:
                S = S * gt
            # delta = (v - k^T S) * beta
            delta = v[t] - np.einsum('hk,hkv->hv', kt, S)
            delta = delta * beta[t][:,None]
            if not decay_first:
                S = S * gt
            S = S + kt[:,:,None]*delta[:,None,:]
            outs[t] = np.einsum('hk,hkv->hv', qq[t], S)
        state['S'] = S
        self.rec[f"gdn_raw_{il}"] = outs.copy()
        # gated rms norm
        zz = z.reshape(Tn, c.n_v_heads, 128)
        gate = zz / (1.0+np.exp(-zz)) if o.get('gdn_gate', 'silu') == 'silu' else 1.0/(1.0+np.exp(-zz))
        nv = np.mean(outs*outs, axis=-1, keepdims=True)
        outs = outs / np.sqrt(nv + c.eps) * self.w(f"blk.{il}.ssm_norm.weight")[None,None,:]
        self.rec[f"gdn_normed_{il}"] = outs.copy()
        if o.get('norm_gate_order', 'after') == 'after':
            outs = outs * gate
        else:
            outs = outs * gate
        self.rec[f"attn_output_{il}"] = outs.reshape(Tn, 4096).copy()
        if o.get('gdn_permute', True):
            # new[d + 128*(b + 2a)] = old[d + 128*(a + 16b)]
            o2 = np.empty_like(outs)
            o2[:, 0::2, :] = outs[:, :16, :]
            o2[:, 1::2, :] = outs[:, 16:, :]
            outs = o2
        self.rec[f"final_output_{il}"] = outs.reshape(Tn, 4096).copy()
        y = self.tf(outs.reshape(Tn, 4096)) @ self.w(f"blk.{il}.ssm_out.weight").T
        self.rec[f"linear_attn_out_{il}"] = y.copy()
        return x + y

    def full_attn(self, x, il, state):
        o = self.o; c = self.cfg
        h = self.rmsnorm(x, f"blk.{il}.attn_norm.weight")
        self.rec[f"attn_norm_{il}"] = h.copy()
        hf = self.tf(h)
        Tn = h.shape[0]
        qg = hf @ self.w(f"blk.{il}.attn_q.weight").T   # [T, 8192]
        k = hf @ self.w(f"blk.{il}.attn_k.weight").T    # [T, 1024]
        v = hf @ self.w(f"blk.{il}.attn_v.weight").T
        qg = qg.reshape(Tn, 2, c.n_head, c.head_dim)
        if o.get('q_layout','chunked') == 'per_head':
            q = qg.reshape(Tn, c.n_head, 2*c.head_dim)[:,:, :c.head_dim]
            gate = qg.reshape(Tn, c.n_head, 2*c.head_dim)[:,:, c.head_dim:]
        else:
            q = qg[:,0]; gate = qg[:,1]
        q = q.reshape(Tn*c.n_head, c.head_dim)
        qn = self.rmsnorm(q, f"blk.{il}.attn_q_norm.weight")
        kk = self.rmsnorm(k.reshape(Tn*c.n_head_kv, c.head_dim), f"blk.{il}.attn_k_norm.weight")
        q = qn.reshape(Tn, c.n_head, c.head_dim)
        kk = kk.reshape(Tn, c.n_head_kv, c.head_dim)
        v = v.reshape(Tn, c.n_head_kv, c.head_dim)
        # rope partial
        pos = state['pos'] + np.arange(Tn)
        def rope(t):
            rd = c.rope_dim
            half = rd//2
            freq = 1.0/(c.rope_theta ** (np.arange(0, rd, 2, dtype=np.float32)/rd))  # [32]
            ang = pos[:,None].astype(np.float32) * freq[None,:]
            cos = np.cos(ang).astype(np.float32)[:, None, :]; sin = np.sin(ang).astype(np.float32)[:, None, :]
            x1 = t[..., :half].copy(); x2 = t[..., half:rd].copy()
            t[..., :half] = x1*cos - x2*sin
            t[..., half:rd] = x1*sin + x2*cos
            return t
        self.rec[f"Qcur_full_{il}"] = qg.reshape(Tn, -1).copy()
        self.rec[f"Qcur_normed_{il}"] = q.reshape(Tn, -1).copy()
        self.rec[f"Kcur_normed_{il}"] = kk.reshape(Tn, -1).copy()
        q = rope(q); kk = rope(kk)
        self.rec[f"Qcur_{il}"] = q.reshape(Tn, -1).copy()
        self.rec[f"k_cache_in_{il}"] = kk.copy()
        # kv cache
        kc = state['k']; vc = state['v']
        kc = np.concatenate([kc, kk], axis=0)
        vc = np.concatenate([vc, v], axis=0)
        state['k'] = kc; state['v'] = vc
        npast = kc.shape[0]
        scale = c.head_dim ** -0.5
        rep = c.n_head // c.n_head_kv
        attn = np.zeros((Tn, c.n_head, c.head_dim), dtype=np.float32)
        for hh in range(c.n_head):
            kvh = hh // rep
            sc = q[:, hh, :] @ kc[:, kvh, :].T * scale  # [T, npast]
            # causal
            qpos = pos[:,None]
            kpos = np.arange(npast)[None,:]
            sc = np.where(kpos <= qpos, sc, -np.inf)
            sc = sc - np.max(sc, axis=-1, keepdims=True)
            e = np.exp(sc); e /= np.sum(e, axis=-1, keepdims=True)
            attn[:, hh, :] = e @ vc[:, kvh, :]
        gflat = gate.reshape(Tn, -1)
        self.rec[f"attn_pregate_{il}"] = attn.reshape(Tn, -1).copy()
        gact = 1.0/(1.0+np.exp(-gflat)) if o.get('gate_act','sigmoid') == 'sigmoid' else gflat/(1.0+np.exp(-gflat))
        self.rec[f"gate_sigmoid_{il}"] = gact.copy()
        attn = attn.reshape(Tn, c.n_head*c.head_dim) * gact
        self.rec[f"attn_gated_{il}"] = attn.copy()
        state['pos'] = npast
        y = self.tf(attn) @ self.w(f"blk.{il}.attn_output.weight").T
        self.rec[f"attn_out_{il}"] = y.copy()
        return x + y

    def ffn(self, x, il):
        h = self.rmsnorm(x, f"blk.{il}.post_attention_norm.weight")
        self.rec[f"attn_post_norm_{il}"] = h.copy()
        tmode = self.o.get('ffn_tf', True)
        if tmode is False:
            hf = h
        elif tmode == 'inv':
            hf = self.tf(h, inverse=True)
        else:
            hf = self.tf(h)
        g = hf @ self.w(f"blk.{il}.ffn_gate.weight").T
        u = hf @ self.w(f"blk.{il}.ffn_up.weight").T
        self.rec[f"ffn_gate_{il}"] = g.copy()
        self.rec[f"ffn_up_{il}"] = u.copy()
        if self.o.get('ffn_swap', False):
            a = (u/(1.0+np.exp(-u))) * g
        else:
            a = (g/(1.0+np.exp(-g))) * u
        self.rec[f"ffn_swiglu_{il}"] = a.copy()
        y = self.tf(a) @ self.w(f"blk.{il}.ffn_down.weight").T
        self.rec[f"ffn_out_{il}"] = y.copy()
        return x + y

    def new_state(self, n_ctx):
        c = self.cfg
        st = []
        for il in range(c.n_layer):
            if (il+1) % c.full_attn_interval == 0:
                st.append({'kind':'attn','k':np.zeros((0,c.n_head_kv,c.head_dim),np.float32),
                           'v':np.zeros((0,c.n_head_kv,c.head_dim),np.float32),'pos':0})
            else:
                st.append({'kind':'gdn','S':np.zeros((c.n_v_heads,c.ssm_state,c.ssm_state),np.float32),
                           'conv':np.zeros((c.conv_k-1, 2*c.inner),np.float32)})
        return st

    def forward(self, toks, st):
        x = np.stack([self.embed(t) for t in toks])
        self.rec["model.input_embed"] = x.copy()
        for il in range(self.cfg.n_layer):
            if st[il]['kind'] == 'gdn':
                x = self.gdn(x, il, st[il])
            else:
                x = self.full_attn(x, il, st[il])
            x = self.ffn(x, il)
            self.rec[f"l_out_{il}"] = x.copy()
        return x


def main():
    opts = dict(sign_first=True, decay_first=False, q_layout='per_head')
    if len(sys.argv) > 1:
        opts.update(json.loads(sys.argv[1]))
    m = Model(opts)
    toks = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [9419]
    out = sys.argv[3] if len(sys.argv) > 3 else "tools/nref_logits.bin"
    st = m.new_state(512)
    x = m.forward(toks, st)
    lg = m.logits(x[-1:])[0]
    lg.astype(np.float32).tofile(out)
    top = np.argsort(lg)[-10:][::-1]
    print("opts", opts, "toks", toks)
    print("top10", top.tolist(), lg[top].tolist())

if __name__ == "__main__":
    main()
