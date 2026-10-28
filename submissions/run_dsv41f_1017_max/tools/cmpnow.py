import numpy as np
for name, a, b in [("t1 vs cpu oracle", "tools/vk_t1.bin", "tools/oracle_logits_t1_cpu.bin"),
                   ("t2 vs oracle_a", "tools/vk_a.bin", "tools/oracle_logits_a.bin")]:
    x = np.fromfile(a, dtype=np.float32); y = np.fromfile(b, dtype=np.float32)
    n = min(len(x), len(y)); x = x[:n]; y = y[:n]
    c = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30))
    p = np.exp(x - x.max()); p /= p.sum(); q = np.exp(y - y.max()); q /= q.sum()
    kl = float(np.sum(q * np.log((q + 1e-30) / (p + 1e-30))))
    topx = np.argsort(x)[-5:][::-1]; topy = np.argsort(y)[-5:][::-1]
    print(f"{name}: cos={c:.6f} KL(p||q)={kl:.4f} top1 eng={topx[0]} gold={topy[0]}")
    print("  eng top5", topx.tolist(), np.round(x[topx], 3).tolist())
    print("  gld top5", topy.tolist(), np.round(y[topy], 3).tolist())
