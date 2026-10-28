import numpy as np, sys
a = np.fromfile(sys.argv[1], dtype=np.float32)
b = np.fromfile(sys.argv[2], dtype=np.float32)
n = min(a.size, b.size)
a = a[:n].astype(np.float64); b = b[:n].astype(np.float64)
cos = float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
top1 = int(np.argmax(a)==np.argmax(b))
# KL(a||b) over softmax
def sm(x):
    x = x - x.max(); e = np.exp(x); return e/e.sum()
pa = sm(a); pb = sm(b)
kl = float(np.sum(pa*np.log((pa+1e-30)/(pb+1e-30))))
print("cos", cos, "top1", top1, "argmax", int(np.argmax(a)), int(np.argmax(b)), "kl", kl)
print("top10 a", np.argsort(a)[-10:][::-1].tolist())
print("top10 b", np.argsort(b)[-10:][::-1].tolist())
