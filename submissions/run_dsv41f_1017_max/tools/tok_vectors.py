import json
d = json.load(open("TOKENIZER-VECTORS.json"))
print(len(d))
for i, v in enumerate(d):
    print(i, repr(v["text"]), v["ids"])
