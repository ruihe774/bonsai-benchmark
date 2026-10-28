import numpy as np, struct
from safetensors import safe_open
import gguf

r = gguf.GGUFReader("data/Ternary-Bonsai-2-4B-PTQ1_0.gguf")
names=["blk.0.attn_gate.weight","blk.0.attn_qkv.weight","blk.3.attn_q.weight","blk.0.ssm_out.weight"]
f = safe_open("data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors", framework="np")
for nm in names:
    t=[x for x in r.tensors if x.name==nm][0]
    raw=np.asarray(t.data).tobytes()
    arr=f.get_tensor(nm).astype(np.float32).reshape(-1)
    NB=200
    tot=0; bad=0
    for blk in range(NB):
        bs=raw[blk*28:blk*28+28]
        s=struct.unpack('<e', bs[26:28])[0]
        seg=arr[blk*128:(blk+1)*128]
        true=(np.isclose(seg,s,rtol=0,atol=0).astype(np.int8)-np.isclose(seg,-s,rtol=0,atol=0).astype(np.int8))
        def d5(b):
            v=(int(b)*243)>>8
            return [(v//(3**(4-p)))%3-1 for p in range(5)]
        def d4(b):
            v=(int(b)*81)>>8
            return [(v//(3**(3-p)))%3-1 for p in range(4)]
        pred=[0]*128
        for c in range(16):
            dd=d5(bs[c])
            for p in range(5): pred[p*16+c]=dd[p]
        for c in range(8):
            dd=d5(bs[16+c])
            for p in range(5): pred[80+p*8+c]=dd[p]
        for c in range(2):
            dd=d4(bs[24+c])
            for p in range(4): pred[120+p*2+c]=dd[p]
        pred=np.array(pred,np.int8)
        m=int(np.sum(pred==true)); tot+=m
        if m!=128 and bad<3:
            bad+=1
            print(nm,"blk",blk,"match",m)
            print(" pred",pred.tolist())
            print(" true",true.tolist())
    print(nm,"match",tot,"/",NB*128, "frac", tot/(NB*128))
