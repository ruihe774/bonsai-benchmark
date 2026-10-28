import re
p="/opt/bonsai/gguf-py/gguf/constants.py"
lines=open(p).read().splitlines()
for i,l in enumerate(lines):
    if 'PTQ' in l or 'PQ2' in l or 'Q1_0' in l or 'Q2_0' in l:
        print(i+1, l)
print("----")
# print blocks around GGML_QUANT_SIZES
for i,l in enumerate(lines):
    if 'GGML_QUANT_SIZES' in l or 'quant_sizes' in l.lower():
        print("SECTION at", i+1)
        for j in range(i, min(i+140, len(lines))):
            print(j+1, lines[j])
        break
