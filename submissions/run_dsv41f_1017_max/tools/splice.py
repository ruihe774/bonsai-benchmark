import re
main = open("src/main.zig").read()
tail = open("tools/tail.zig").read()
idx = main.index("// Vulkan engine state is created lazily")
head = main[:idx]
# fix: all bufPrint locals must be const
tail = tail.replace("            var qkv_name = std.fmt", "            const qkv_name = std.fmt")
tail = tail.replace("            var gname = std.fmt", "            const gname = std.fmt")
tail = tail.replace("            var alname = std.fmt", "            const alname = std.fmt")
tail = tail.replace("            var betaname = std.fmt", "            const betaname = std.fmt")
tail = tail.replace("                var a_name = std.fmt", "                const a_name = std.fmt")
tail = tail.replace("                var dt_name = std.fmt", "                const dt_name = std.fmt")
tail = tail.replace("                var cw_name = std.fmt", "                const cw_name = std.fmt")
tail = tail.replace("                var nw_name = std.fmt", "                const nw_name = std.fmt")
tail = tail.replace("            var oname = std.fmt", "            const oname = std.fmt")
tail = tail.replace("            var qname = std.fmt", "            const qname = std.fmt")
tail = tail.replace("            var kname = std.fmt", "            const kname = std.fmt")
tail = tail.replace("            var vname = std.fmt", "            const vname = std.fmt")
tail = tail.replace("            var qn = std.fmt", "            const qn = std.fmt")
tail = tail.replace("            var kn = std.fmt", "            const kn = std.fmt")
tail = tail.replace("        var pn = std.fmt", "        const pn = std.fmt")
tail = tail.replace("        var fg = std.fmt", "        const fg = std.fmt")
tail = tail.replace("        var fu = std.fmt", "        const fu = std.fmt")
tail = tail.replace("        var fd = std.fmt", "        const fd = std.fmt")
tail = tail.replace("        var an = std.fmt", "        const an = std.fmt")
tail = tail.replace("""    if (want_logits) {
        const dstf: [*]f32 = undefined;
        _ = dstf;
    }
""", "")
# rewrite generate loop cleanly
old_gen = tail[tail.index("    var nv: usize = @intCast(m.n_vocab);"):tail.index("    _ = &nv;\n    return gen;\n}")+len("    _ = &nv;\n    return gen;\n}")]
new_gen = """    const nv: usize = @intCast(m.n_vocab);
    const buf = alloc.alloc(f32, nv) catch return -2;
    defer alloc.free(buf);
    if (n_prompt > 0) {
        const rc = bonsai_eval(ctx, prompt, n_prompt, buf.ptr, m.n_vocab);
        if (rc != 0) return rc;
    }
    var gen: i32 = 0;
    while (gen < n_predict) {
        var best: f32 = -std.math.inf(f32);
        var bi: i32 = 0;
        for (buf[0..nv]) |v, idx| {
            if (v > best) {
                best = v;
                bi = @intCast(idx);
            }
        }
        out_ids.?[@intCast(gen)] = bi;
        gen += 1;
        if ((flags & 1) == 0 and bi == m.eos_id) break;
        const tok_arr: [1]i32 = .{bi};
        const rc = bonsai_eval(ctx, &tok_arr, 1, buf.ptr, m.n_vocab);
        if (rc != 0) return rc;
    }
    return gen;
}"""
tail = tail.replace(old_gen, new_gen)
tail = tail.replace("    _ = flags;\n", "")
open("src/main.zig","w").write(head + tail)
print("spliced", len(head+tail))
