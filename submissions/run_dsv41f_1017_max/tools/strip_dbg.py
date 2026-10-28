src = open("src/main.zig").read()
lines = src.split("\n")
out = []
i = 0
while i < len(lines):
    l = lines[i]
    if l.startswith("export fn bonsai_dbg_"):
        # skip to the line that is exactly "}" at top level
        j = i
        while j < len(lines) and lines[j].rstrip() != "}":
            j += 1
        i = j + 1
        continue
    out.append(l)
    i += 1
res = "\n".join(out)
res = "\n".join(l for l in res.split("\n") if "g_stop_layer" not in l and "g_dbg_model" not in l)
open("src/main.zig", "w").write(res)
print("remaining dbg refs:", res.count("dbg_"), "stop refs:", res.count("g_stop_layer"))
