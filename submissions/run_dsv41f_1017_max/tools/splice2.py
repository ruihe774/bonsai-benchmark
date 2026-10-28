main = open("src/main.zig").read()
new = open("tools/evalone.zig").read()
start = main.index("fn evalOne(")
end = main.index("export fn bonsai_context_create(")
main = main[:start] + new + "\n" + main[end:]
open("src/main.zig","w").write(main)
print("ok", len(main))
