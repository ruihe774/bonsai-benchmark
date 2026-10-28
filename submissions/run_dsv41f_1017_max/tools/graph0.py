import re
lines = open("tools/debug3.txt", errors='replace').read().splitlines()
name_re = re.compile(r'common_debug_cb_eval:\s+(.*?) = \(f32\)\s+(.*)')
# first eval block: between first 'llama_context: graph' maybe. Use lines up to second 'node_60' occurrence (line ~38000)
out = []
count = 0
for i,l in enumerate(lines[:34000]):
    m = name_re.match(l)
    if m:
        nm = m.group(1).strip()
        op = m.group(2).strip()
        out.append(f"{i}: {nm} :: {op}")
open("tools/graph0.txt","w").write("\n".join(out))
print(len(out))
