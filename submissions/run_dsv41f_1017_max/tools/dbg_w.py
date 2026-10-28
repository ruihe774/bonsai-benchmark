src = open("src/main.zig").read()
src += """

export fn bonsai_dbg_w(off: u32, out: ?[*]f32, n: i32) callconv(.c) i32 {
    if (out == null or n < 1 or n > 4096) return -1;
    const m = g_dbg_model orelse return -2;
    vk.beginCommands(&g_ctx) catch return -3;
    const bytes: u64 = @intCast(n * 4);
    const tmp = vk.createBuffer(&g_ctx, bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT, vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) catch return -4;
    vk.copyBufferO(&g_ctx, m.weights, @as(u64, off) * 4, tmp, 0, bytes);
    vk.endSubmitWait(&g_ctx) catch return -5;
    const p = vk.mapBuffer(&g_ctx, tmp) catch return -6;
    const srcs: [*]const f32 = @ptrCast(@alignCast(p.ptr));
    @memcpy(out.?[0..@intCast(n)], srcs[0..@intCast(n)]);
    return 0;
}
"""
src = src.replace("var g_stop_layer: i32 = -1;", "var g_stop_layer: i32 = -1;\nvar g_dbg_model: ?*Model = null;")
src = src.replace("    if (!m.have_gpu) {\n        setErr(\"model has no Vulkan weights", "    g_dbg_model = m;\n    if (!m.have_gpu) {\n        setErr(\"model has no Vulkan weights")
open("src/main.zig","w").write(src)
print("ok")
