import gguf, json
r = gguf.GGUFReader("data/Ternary-Bonsai-2-4B-PTQ1_0.gguf")
f = r.fields
def arr(name):
    x = f[name]
    try:
        return x.contents()
    except Exception:
        return x.data
wn = arr("prism.hadamard.weight_names")
sw = arr("prism.hadamard.sign_widths")
sv = arr("prism.hadamard.sign_values")
iw = arr("prism.hadamard.inverse_weight_names")
print("n weight_names", len(wn))
print("names:", [str(x) for x in wn])
print("inverse:", [str(x) for x in iw])
print("sign_widths", [int(x) for x in sw])
print("n sign_values", len(sv), "first 40", [int(x) for x in sv[:40]])
# check sum of widths vs len
print("sum widths?", sum(int(x) for x in sw), "len", len(sv))
