import numpy as np
import gguf
r = gguf.GGUFReader("data/Ternary-Bonsai-2-4B-PTQ1_0.gguf")
want = ["blk.0.ssm_a","blk.0.ssm_dt.bias","blk.0.ssm_norm.weight","blk.0.ssm_conv1d.weight","blk.0.ssm_alpha.weight","blk.0.ssm_beta.weight","blk.0.attn_norm.weight","blk.0.post_attention_norm.weight","output_norm.weight","blk.0.attn_k_norm.weight","blk.3.attn_q_norm.weight"]
for nm in want:
    t=[x for x in r.tensors if x.name==nm][0]
    a=np.frombuffer(np.asarray(t.data).tobytes(), dtype=np.float32 if t.tensor_type.name=='F32' else np.float16)
    print(nm, t.tensor_type.name, a.shape, "min",float(np.nanmin(a)),"max",float(np.nanmax(a)),"mean",float(np.nanmean(a)))
    if a.size <= 64:
        print("  values", np.round(a.astype(np.float32),4).tolist())
    else:
        print("  head", np.round(a.astype(np.float32)[:8],4).tolist())
