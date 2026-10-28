import gguf, sys, json

path = sys.argv[1]
r = gguf.GGUFReader(path)
print("n_fields", len(r.fields))
for k, f in r.fields.items():
    if k.startswith("tokenizer.ggml.tokens") or k.startswith("tokenizer.ggml.merges") or k.startswith("tokenizer.ggml.token_type"):
        try:
            print(k, f.types, len(f.parts), f.data.shape if hasattr(f.data, 'shape') else None)
        except Exception as e:
            print(k, "?", e)
        continue
    try:
        v = f.contents()
    except Exception:
        v = f.data
    s = str(v)
    if len(s) > 300:
        s = s[:300] + "..."
    print(k, f.types, s)

print("== tensors ==", len(r.tensors))
for t in r.tensors:
    print(t.name, t.tensor_type.name, list(t.shape), t.n_bytes, t.data_offset)
