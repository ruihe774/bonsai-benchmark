import gguf
r = gguf.GGUFReader("data/Ternary-Bonsai-2-4B-PTQ1_0.gguf")
f = r.fields
toks = [str(x) for x in f["tokenizer.ggml.tokens"].contents()]
for i in (220, 256, 42417, 79, 16336, 1414, 12675, 0, 32, 3841):
    print(i, repr(toks[i]))
