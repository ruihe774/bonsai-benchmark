import numpy as np
from safetensors import safe_open
f = safe_open("data/Ternary-Bonsai-2-4B-PTQ1_0-f16.safetensors", framework="np")
E = f.get_tensor("token_embd.weight").astype(np.float64)
x = E[[248044, 248046]]  # [2,2560]
print("sum x", x.sum(), "dbg node_0 sum", -0.415237)
x0 = x.reshape(2,5,512)[:, :, 0]
print("x[block][0] sum", x0.sum(), "pred norm", 22.627417*x0.sum(), "pred unnorm", 512*x0.sum())
print("dbg node_2 sum", -2.272893)
print("ratio", -2.272893/(x0.sum()))
# maybe sum is over abs? no. Try: sum of input_embed
# model.input_embed sum dbg 0.378603
signs = None
