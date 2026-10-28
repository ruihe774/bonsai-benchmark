fn findBlk(m: *const Model, comptime fmt: []const u8, il: u32) !*const TensorInfo {
    var buf: [128]u8 = undefined;
    const name = std.fmt.bufPrint(&buf, fmt, .{il}) catch return error.Name;
    return m.find(name) orelse error.NoTensor;
}

fn f32OffBlk(m: *const Model, comptime fmt: []const u8, il: u32) !u32 {
    var buf: [128]u8 = undefined;
    const name = std.fmt.bufPrint(&buf, fmt, .{il}) catch return error.Name;
    const t = m.find(name) orelse return error.NoTensor;
    return @intCast(t.offset / 4);
}

fn evalOne(c: *Context, tok: i32, want_logits: bool) !void {
    const m = c.model;
    const tok_u: u32 = @intCast(tok);
    _ = g_ctx.api.reset_descriptor_pool(g_ctx.device, g_ctx.pool, 0);
    var layouts: [1024]vk.VkDescriptorSetLayout = undefined;
    for (&layouts) |*l| l.* = g_ctx.dsl;
    var ai = vk.VkDescriptorSetAllocateInfo{ .descriptorPool = g_ctx.pool, .descriptorSetCount = 1024, .pSetLayouts = &layouts };
    if (g_ctx.api.allocate_descriptor_sets(g_ctx.device, &ai, &g_sets_buf) != vk.VK_SUCCESS) return error.Descriptors;
    g_set_count = 1024;
    g_set_idx = 0;

    try vk.beginCommands(&g_ctx);
    const emb = m.find("token_embd.weight") orelse return error.NoTensor;
    {
        const p = pcBuf(&.{ N_EMBD, @intCast(emb.offset / 4), tok_u * (N_EMBD / 128), 0 });
        dispatch(g_pipes.dequant_row, &.{ .{ .b = m.weights, .off = 0 }, .{ .b = c.scratch, .off = SCR_X } }, &p, (N_EMBD + 255) / 256, 1, 1);
    }
    tfDispatch(c, SCR_X, SCR_X, N_EMBD, SIGNS_2560, 1, 1);

    var il: u32 = 0;
    while (il < N_LAYER) : (il += 1) {
        const full = (il % FULL_ATTN_INTERVAL) == (FULL_ATTN_INTERVAL - 1);
        rmsDispatch(c, SCR_X, SCR_H, N_EMBD, try f32OffBlk(m, "blk.{d}.attn_norm.weight", il), 1, 1);
        tfDispatch(c, SCR_H, SCR_HF, N_EMBD, SIGNS_2560, 0, 1);
        if (!full) {
            const tqkv = try findBlk(m, "blk.{d}.attn_qkv.weight", il);
            matvecPtq(c, SCR_HF, @intCast(tqkv.offset / 4), N_EMBD, QKV_DIM, SCR_QKV);
            const tgate = try findBlk(m, "blk.{d}.attn_gate.weight", il);
            matvecPtq(c, SCR_HF, @intCast(tgate.offset / 4), N_EMBD, INNER, SCR_Z);
            const al_off = try f32OffBlk(m, "blk.{d}.ssm_alpha.weight", il);
            const beta_off = try f32OffBlk(m, "blk.{d}.ssm_beta.weight", il);
            matvecF32(c, SCR_HF, al_off, N_EMBD, 32, SCR_GB);
            matvecF32(c, SCR_HF, beta_off, N_EMBD, 32, SCR_GB + 128);
            {
                const p = pcBuf(&.{ al_off, beta_off, try f32OffBlk(m, "blk.{d}.ssm_a", il), try f32OffBlk(m, "blk.{d}.ssm_dt.bias", il) });
                dispatch(g_pipes.gates, &.{ .{ .b = m.weights, .off = 0 }, .{ .b = c.scratch, .off = SCR_GB } }, &p, 1, 1, 1);
            }
            {
                const p = pcBuf(&.{ QKV_DIM, try f32OffBlk(m, "blk.{d}.ssm_conv1d.weight", il), 0, 0 });
                dispatch(g_pipes.conv_silu, &.{ .{ .b = c.scratch, .off = SCR_QKV }, .{ .b = m.weights, .off = 0 }, .{ .b = c.convhist, .off = @as(u64, slotOf(il)) * 3 * QKV_DIM * 4 }, .{ .b = c.scratch, .off = SCR_QKV2 } }, &p, QKV_DIM / 256, 1, 1);
            }
            {
                const p = pcBuf(&.{ 32, 0, 0, 0 });
                dispatch(g_pipes.l2norm, &.{.{ .b = c.scratch, .off = SCR_QKV2 }}, &p, 32, 1, 1);
            }
            {
                const p = pcBuf(&.{ try f32OffBlk(m, "blk.{d}.ssm_norm.weight", il), 0, 0, 0 });
                dispatch(g_pipes.gdn, &.{
                    .{ .b = c.scratch, .off = SCR_QKV2 },
                    .{ .b = c.scratch, .off = SCR_GB },
                    .{ .b = c.states, .off = @as(u64, slotOf(il)) * 32 * 128 * 128 * 4 },
                    .{ .b = c.scratch, .off = SCR_Z },
                    .{ .b = m.weights, .off = 0 },
                    .{ .b = c.scratch, .off = SCR_ATTN },
                }, &p, 32, 1, 1);
            }
            tfDispatch(c, SCR_ATTN, SCR_ATTNF, INNER, SIGNS_4096, 0, 1);
            const tout = try findBlk(m, "blk.{d}.ssm_out.weight", il);
            matvecPtq(c, SCR_ATTNF, @intCast(tout.offset / 4), INNER, N_EMBD, SCR_Y);
            ewDispatch(c, SCR_X, SCR_Y, SCR_X, N_EMBD, 0);
        } else {
            const fl = il / FULL_ATTN_INTERVAL;
            const tq = try findBlk(m, "blk.{d}.attn_q.weight", il);
            matvecPtq(c, SCR_HF, @intCast(tq.offset / 4), N_EMBD, QKV_DIM, SCR_QKV);
            const tk = try findBlk(m, "blk.{d}.attn_k.weight", il);
            matvecPtq(c, SCR_HF, @intCast(tk.offset / 4), N_EMBD, 1024, SCR_KRAW);
            const tv = try findBlk(m, "blk.{d}.attn_v.weight", il);
            matvecPtq(c, SCR_HF, @intCast(tv.offset / 4), N_EMBD, 1024, SCR_VRAW);
            rmsDispatch(c, SCR_QKV, SCR_QKV, 256, try f32OffBlk(m, "blk.{d}.attn_q_norm.weight", il), 1, 16);
            rmsDispatch(c, SCR_KRAW, SCR_KRAW, 256, try f32OffBlk(m, "blk.{d}.attn_k_norm.weight", il), 1, 4);
            {
                const p = pcBuf(&.{ @intCast(c.pos), 16, 0, 0 });
                dispatch(g_pipes.rope, &.{ .{ .b = c.scratch, .off = SCR_QKV }, .{ .b = c.scratch, .off = SCR_QROT } }, &p, 16, 1, 1);
            }
            {
                const p = pcBuf(&.{ @intCast(c.pos), 4, 0, 0 });
                dispatch(g_pipes.rope, &.{ .{ .b = c.scratch, .off = SCR_KRAW }, .{ .b = c.scratch, .off = SCR_KROT } }, &p, 4, 1, 1);
            }
            const kbase = @as(u64, fl) * 2 * c.kv_layer_stride;
            {
                const p = pcBuf(&.{ @intCast(c.pos), 1024, 0, 0 });
                dispatch(g_pipes.store_kv, &.{
                    .{ .b = c.scratch, .off = SCR_KROT },
                    .{ .b = c.scratch, .off = SCR_VRAW },
                    .{ .b = c.kv, .off = kbase },
                    .{ .b = c.kv, .off = kbase + c.kv_layer_stride },
                }, &p, 4, 1, 1);
            }
            {
                const p = pcBuf(&.{ @intCast(c.pos), 0, 0, 0 });
                dispatch(g_pipes.attention, &.{
                    .{ .b = c.scratch, .off = SCR_QROT },
                    .{ .b = c.kv, .off = kbase },
                    .{ .b = c.kv, .off = kbase + c.kv_layer_stride },
                    .{ .b = c.scores, .off = 0 },
                    .{ .b = c.scratch, .off = SCR_ATTN },
                }, &p, 16, 1, 1);
            }
            ewDispatch(c, SCR_ATTN, SCR_QKV + 16384, SCR_ATTN, INNER, 2);
            tfDispatch(c, SCR_ATTN, SCR_ATTNF, INNER, SIGNS_4096, 0, 1);
            const to = try findBlk(m, "blk.{d}.attn_output.weight", il);
            matvecPtq(c, SCR_ATTNF, @intCast(to.offset / 4), INNER, N_EMBD, SCR_Y);
            ewDispatch(c, SCR_X, SCR_Y, SCR_X, N_EMBD, 0);
        }
        rmsDispatch(c, SCR_X, SCR_H, N_EMBD, try f32OffBlk(m, "blk.{d}.post_attention_norm.weight", il), 1, 1);
        tfDispatch(c, SCR_H, SCR_HF, N_EMBD, SIGNS_2560, 0, 1);
        const tfg = try findBlk(m, "blk.{d}.ffn_gate.weight", il);
        matvecPtq(c, SCR_HF, @intCast(tfg.offset / 4), N_EMBD, N_FF, SCR_FF);
        const tfu = try findBlk(m, "blk.{d}.ffn_up.weight", il);
        matvecPtq(c, SCR_HF, @intCast(tfu.offset / 4), N_EMBD, N_FF, SCR_FU);
        ewDispatch(c, SCR_FF, SCR_FU, SCR_FA, N_FF, 1);
        tfDispatch(c, SCR_FA, SCR_FA, N_FF, SIGNS_9216, 0, 1);
        const tfd = try findBlk(m, "blk.{d}.ffn_down.weight", il);
        matvecPtq(c, SCR_FA, @intCast(tfd.offset / 4), N_FF, N_EMBD, SCR_Y);
        ewDispatch(c, SCR_X, SCR_Y, SCR_X, N_EMBD, 0);
    }
    const on = m.find("output_norm.weight") orelse return error.NoTensor;
    rmsDispatch(c, SCR_X, SCR_H, N_EMBD, @intCast(on.offset / 4), 1, 1);
    tfDispatch(c, SCR_H, SCR_HF, N_EMBD, SIGNS_2560, 0, 1);
    if (want_logits) {
        matvecPtq(c, SCR_HF, @intCast(emb.offset / 4), N_EMBD, @intCast(m.n_vocab), SCR_LOGITS);
        vk.copyBufferO(&g_ctx, c.scratch, SCR_LOGITS, c.readback, 0, @as(u64, @intCast(m.n_vocab)) * 4);
    }
    try vk.endSubmitWait(&g_ctx);
}
