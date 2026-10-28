import numpy as np
import gguf

r = gguf.GGUFReader("data/Ternary-Bonsai-2-4B-PTQ1_0.gguf")
names = ["token_embd.weight", "blk.0.attn_qkv.weight", "blk.3.attn_q.weight"]
for nm in names:
    t = [x for x in r.tensors if x.name == nm][0]
    a = np.asarray(t.data).tobytes()
    b = np.frombuffer(a, dtype=np.uint8)
    n = len(b)//28
    b = b[:n*28].reshape(n,28)
    print(nm, "blocks", n)
    for pos in [0,1,2,22,23,24,25,26,27]:
        col = b[:,pos]
        u = np.unique(col)
        missing = sorted(set(range(256)) - set(u.tolist()))
        print(" pos",pos,"min",col.min(),"max",col.max(),"mean",col.mean(),"nuniq",len(u),"missing_sample",missing[:12], "missing_count", len(missing))
    v16 = b[:,24].astype(np.uint32) | (b[:,25].astype(np.uint32)<<8)
    print(" u16@24 max", v16.max(), "min", v16.min(), "mean", v16.mean())
    v2 = b[:,0].astype(np.uint32) | (b[:,1].astype(np.uint32)<<8)
    print(" u16@0 max", v2.max(), "min", v2.min(), "mean", v2.mean())
    f16 = np.frombuffer(b[:,26:28].copy().tobytes(), dtype=np.float16)
    print(" f16@26 min/max", np.nanmin(f16), np.nanmax(f16), "nan", np.sum(np.isnan(f16)))
