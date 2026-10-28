src = open("src/main.zig").read()
# generalize dump export
old = src[src.index("export fn bonsai_dbg_x("):]
old = old[:old.index("\n}\n")+3]
new = """export fn bonsai_dbg_x(ctx: ?*ContextOpaque, off: u32, out: ?[*]f32, n: i32) callconv(.c) i32 {
    if (ctx == null or out == null or n < 1 or n > 200000) return -1;
    const c: *Context = @ptrCast(@alignCast(ctx.?));
    const bytes: u64 = @intCast(n * 4);
    vk.beginCommands(&g_ctx) catch return -2;
    const tmp = vk.createBuffer(&g_ctx, bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT, vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) catch return -3;
    vk.copyBufferO(&g_ctx, c.scratch, @as(u64, off) * 4, tmp, 0, bytes);
    vk.endSubmitWait(&g_ctx) catch return -4;
    const p = vk.mapBuffer(&g_ctx, tmp) catch return -5;
    const srcs: [*]const f32 = @ptrCast(@alignCast(p.ptr));
    @memcpy(out.?[0..@intCast(n)], srcs[0..@intCast(n)]);
    return 0;
}
"""
src = src.replace(old, new)
# add stop points in gdn branch
src = src.replace("""            matvecPtq(c, SCR_ATTNF, @intCast(tout.offset / 4), INNER, N_EMBD, SCR_Y);
            ewDispatch(c, SCR_X, SCR_Y, SCR_X, N_EMBD, 0);""",
"""            matvecPtq(c, SCR_ATTNF, @intCast(tout.offset / 4), INNER, N_EMBD, SCR_Y);
            if (g_stop_layer == -12) break;
            ewDispatch(c, SCR_X, SCR_Y, SCR_X, N_EMBD, 0);""")
src = src.replace("""        rmsDispatch(c, SCR_X, SCR_H, N_EMBD, try f32OffBlk(m, "blk.{d}.attn_norm.weight", il), 1, 1);
        tfDispatch(c, SCR_H, SCR_HF, N_EMBD, SIGNS_2560, 0, 1);
        if (!full) {""",
"""        rmsDispatch(c, SCR_X, SCR_H, N_EMBD, try f32OffBlk(m, "blk.{d}.attn_norm.weight", il), 1, 1);
        if (g_stop_layer == -3 and il == 0) break;
        tfDispatch(c, SCR_H, SCR_HF, N_EMBD, SIGNS_2560, 0, 1);
        if (g_stop_layer == -4 and il == 0) break;
        if (!full) {""")
src = src.replace("""            matvecPtq(c, SCR_HF, @intCast(tqkv.offset / 4), N_EMBD, QKV_DIM, SCR_QKV);
            const tgate""",
"""            matvecPtq(c, SCR_HF, @intCast(tqkv.offset / 4), N_EMBD, QKV_DIM, SCR_QKV);
            if (g_stop_layer == -5 and il == 0) break;
            const tgate""")
src = src.replace("""                const p = pcBuf(&.{ 32, 0, 0, 0 });
                dispatch(g_pipes.l2norm, &.{.{ .b = c.scratch, .off = SCR_QKV2 }}, &p, 32, 1, 1);
            }""",
"""                const p = pcBuf(&.{ 32, 0, 0, 0 });
                dispatch(g_pipes.l2norm, &.{.{ .b = c.scratch, .off = SCR_QKV2 }}, &p, 32, 1, 1);
            }
            if (g_stop_layer == -9 and il == 0) break;""")
src = src.replace("""                }, &p, 32, 1, 1);
            }
            tfDispatch(c, SCR_ATTN, SCR_ATTNF, INNER, SIGNS_4096, 0, 1);""",
"""                }, &p, 32, 1, 1);
            }
            if (g_stop_layer == -10 and il == 0) break;
            tfDispatch(c, SCR_ATTN, SCR_ATTNF, INNER, SIGNS_4096, 0, 1);""")
open("src/main.zig","w").write(src)
print("ok")
