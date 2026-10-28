src = open("src/main.zig").read()
# add globals
src = src.replace("var g_sets_buf: [1024]vk.VkDescriptorSet = undefined;",
"""var g_sets_buf: [1024]vk.VkDescriptorSet = undefined;
var g_stop_layer: i32 = -1;""")
# break after layer
src = src.replace("""        ewDispatch(c, SCR_X, SCR_Y, SCR_X, N_EMBD, 0);
    }
    const on = m.find("output_norm.weight")""",
"""        ewDispatch(c, SCR_X, SCR_Y, SCR_X, N_EMBD, 0);
        if (g_stop_layer >= 0 and @as(i32, @intCast(il)) >= g_stop_layer) break;
    }
    const on = m.find("output_norm.weight")""")
# add debug exports at end
src += """

export fn bonsai_dbg_stop(layer: i32) callconv(.c) void {
    g_stop_layer = layer;
}

export fn bonsai_dbg_x(ctx: ?*ContextOpaque, out: ?[*]f32, cap: i32) callconv(.c) i32 {
    if (ctx == null or out == null or cap < 2560) return -1;
    const c: *Context = @ptrCast(@alignCast(ctx.?));
    vk.beginCommands(&g_ctx) catch return -2;
    const tmp = vk.createBuffer(&g_ctx, 2560 * 4, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT, vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) catch return -3;
    vk.copyBufferO(&g_ctx, c.scratch, SCR_X, tmp, 0, 2560 * 4);
    vk.endSubmitWait(&g_ctx) catch return -4;
    const p = vk.mapBuffer(&g_ctx, tmp) catch return -5;
    const srcs: [*]const f32 = @ptrCast(@alignCast(p.ptr));
    @memcpy(out.?[0..2560], srcs[0..2560]);
    return 0;
}
"""
open("src/main.zig","w").write(src)
print("ok")
