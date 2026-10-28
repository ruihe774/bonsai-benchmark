// ---------------- Vulkan compute engine ----------------
const SCR_X: u64 = 0;
const SCR_H: u64 = 10240;
const SCR_HF: u64 = 20480;
const SCR_QKV: u64 = 30720;
const SCR_QKV2: u64 = 63488;
const SCR_Z: u64 = 96256;
const SCR_GB: u64 = 112640;
const SCR_Y: u64 = 112896;
const SCR_FF: u64 = 123136;
const SCR_FU: u64 = 160000;
const SCR_FA: u64 = 196864;
const SCR_ATTN: u64 = 233728;
const SCR_ATTNF: u64 = 250112;
const SCR_QROT: u64 = 266496;
const SCR_KRAW: u64 = 282880;
const SCR_VRAW: u64 = 286976;
const SCR_KROT: u64 = 291072;
const SCR_LOGITS: u64 = 295168;
const SCR_TOTAL: u64 = 1288448;

const SIGNS_2560: u32 = 0;
const SIGNS_4096: u32 = 2560;
const SIGNS_9216: u32 = 6656;

var g_ctx: vk.Ctx = .{};
var g_vk_ready: bool = false;
var g_pipes_ready: bool = false;
var g_sets_buf: [1024]vk.VkDescriptorSet = undefined;
var g_set_count: usize = 0;
var g_set_idx: usize = 0;

const Pipes = struct {
    matvec_ptq: vk.VkPipeline = 0,
    matvec_f32: vk.VkPipeline = 0,
    rmsnorm: vk.VkPipeline = 0,
    tf: vk.VkPipeline = 0,
    conv_silu: vk.VkPipeline = 0,
    l2norm: vk.VkPipeline = 0,
    gates: vk.VkPipeline = 0,
    gdn: vk.VkPipeline = 0,
    rope: vk.VkPipeline = 0,
    store_kv: vk.VkPipeline = 0,
    attention: vk.VkPipeline = 0,
    ew: vk.VkPipeline = 0,
    dequant_row: vk.VkPipeline = 0,
};
var g_pipes: Pipes = .{};

fn makePipeline(spv_bytes: []const u8) !vk.VkPipeline {
    const nwords = (spv_bytes.len + 3) / 4;
    const buf = try alloc.alloc(u32, nwords);
    defer alloc.free(buf);
    @memcpy(std.mem.sliceAsBytes(buf)[0..spv_bytes.len], spv_bytes);
    var smci = vk.VkShaderModuleCreateInfo{ .codeSize = spv_bytes.len, .pCode = buf.ptr };
    var mod: vk.VkShaderModule = 0;
    if (g_ctx.api.create_shader_module(g_ctx.device, &smci, null, &mod) != vk.VK_SUCCESS) return error.ShaderModule;
    defer g_ctx.api.destroy_shader_module(g_ctx.device, mod, null);
    const stage = vk.VkPipelineShaderStageCreateInfo{ .module = mod, .pName = "main" };
    var ci = vk.VkComputePipelineCreateInfo{ .stage = stage, .layout = g_ctx.pl };
    var pipe: vk.VkPipeline = 0;
    if (g_ctx.api.create_compute_pipelines(g_ctx.device, 0, 1, @ptrCast(&ci), null, @ptrCast(&pipe)) != vk.VK_SUCCESS) return error.Pipeline;
    return pipe;
}

fn createPipelines() !void {
    g_pipes.matvec_ptq = try makePipeline(@embedFile("spv/matvec_ptq.spv"));
    g_pipes.matvec_f32 = try makePipeline(@embedFile("spv/matvec_f32.spv"));
    g_pipes.rmsnorm = try makePipeline(@embedFile("spv/rmsnorm.spv"));
    g_pipes.tf = try makePipeline(@embedFile("spv/tf_hadamard.spv"));
    g_pipes.conv_silu = try makePipeline(@embedFile("spv/conv_silu.spv"));
    g_pipes.l2norm = try makePipeline(@embedFile("spv/l2norm_qk.spv"));
    g_pipes.gates = try makePipeline(@embedFile("spv/gates.spv"));
    g_pipes.gdn = try makePipeline(@embedFile("spv/gdn.spv"));
    g_pipes.rope = try makePipeline(@embedFile("spv/rope.spv"));
    g_pipes.store_kv = try makePipeline(@embedFile("spv/store_kv.spv"));
    g_pipes.attention = try makePipeline(@embedFile("spv/attention.spv"));
    g_pipes.ew = try makePipeline(@embedFile("spv/ew.spv"));
    g_pipes.dequant_row = try makePipeline(@embedFile("spv/dequant_row.spv"));
}

fn ensureVk() !void {
    if (!g_vk_ready) {
        try vk.init(&g_ctx);
        g_vk_ready = true;
    }
    if (!g_pipes_ready) {
        try createPipelines();
        g_pipes_ready = true;
    }
}

fn gpuSetupModel(m: *Model) !void {
    try ensureVk();
    const usage_w: u32 = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    m.weights = try vk.createBuffer(&g_ctx, m.data_size, usage_w, vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    const chunk: u64 = 32 * 1024 * 1024;
    const staging = try vk.createBuffer(&g_ctx, chunk, vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    const map = try vk.mapBuffer(&g_ctx, staging);
    var off: u64 = 0;
    while (off < m.data_size) {
        const n = @min(chunk, m.data_size - off);
        const nread = pread(m.fd, map.ptr, @intCast(n), @intCast(m.data_start + off));
        if (nread != @as(isize, @intCast(n))) return error.ReadFailed;
        try vk.beginCommands(&g_ctx);
        vk.copyBuffer(&g_ctx, staging, m.weights, off, n);
        try vk.endSubmitWait(&g_ctx);
        off += n;
    }
    // signs buffer (host visible, tiny)
    const usage_s: u32 = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    m.signs_buf = try vk.createBuffer(&g_ctx, @max(64, m.signs.len * 4), usage_s, vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    const smap = try vk.mapBuffer(&g_ctx, m.signs_buf);
    const sb: []f32 = @alignCast(std.mem.bytesAsSlice(f32, smap));
    for (m.signs, 0..) |v, i| sb[i] = v;
    m.have_gpu = true;
}

fn slotOf(il: u32) u32 {
    return il - (il + 1) / 4;
}

fn pcBuf(vals: []const u32) [16]u32 {
    var p = [_]u32{0} ** 16;
    for (vals, 0..) |v, i| {
        if (i < 16) p[i] = v;
    }
    return p;
}

fn fbits(v: f32) u32 {
    return @bitCast(v);
}

const Binding = struct { b: vk.Buffer, off: u64 };

fn dispatch(pipe: vk.VkPipeline, bufs: []const Binding, pc: *const [16]u32, gx: u32, gy: u32, gz: u32) void {
    if (g_set_idx >= g_set_count) return;
    const set = g_sets_buf[g_set_idx];
    g_set_idx += 1;
    var infos: [8]vk.VkDescriptorBufferInfo = undefined;
    var writes: [8]vk.VkWriteDescriptorSet = undefined;
    var i: usize = 0;
    while (i < 8) : (i += 1) {
        const b = if (i < bufs.len) bufs[i] else bufs[bufs.len - 1];
        infos[i] = .{ .buffer = b.b.buf, .offset = b.off, .range = 0xFFFFFFFFFFFFFFFF };
        writes[i] = .{ .dstSet = set, .dstBinding = @intCast(i), .descriptorCount = 1, .descriptorType = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &infos[i] };
    }
    g_ctx.api.update_descriptor_sets(g_ctx.device, 8, &writes, 0, null);
    g_ctx.api.cmd_bind_pipeline(g_ctx.cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    const one: [1]vk.VkDescriptorSet = .{set};
    g_ctx.api.cmd_bind_descriptor_sets(g_ctx.cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, g_ctx.pl, 0, 1, &one, 0, null);
    g_ctx.api.cmd_push_constants(g_ctx.cmd, g_ctx.pl, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, 64, @ptrCast(pc));
    g_ctx.api.cmd_dispatch(g_ctx.cmd, gx, gy, gz);
    vk.barrier(&g_ctx);
}

const Context = struct {
    model: *Model,
    n_ctx: i32,
    pos: i32 = 0,
    scratch: vk.Buffer = .{},
    kv: vk.Buffer = .{},
    states: vk.Buffer = .{},
    convhist: vk.Buffer = .{},
    scores: vk.Buffer = .{},
    readback: vk.Buffer = .{},
    rb_ptr: [*]u8 = undefined,
    kv_layer_stride: u64 = 0,
};

fn tensorOff(m: *const Model, name: []const u8) !u32 {
    const t = m.find(name) orelse return error.NoTensor;
    return @intCast(t.offset);
}

fn tensorF32Off(m: *const Model, name: []const u8) !u32 {
    const t = m.find(name) orelse return error.NoTensor;
    return @intCast(t.offset / 4);
}

fn tfDispatch(c: *Context, src: u64, dst: u64, width: u32, sign_base: u32, inverse: u32, rows: u32) void {
    const nblk = width / 512;
    const p = pcBuf(&.{ width, sign_base, inverse, nblk });
    dispatch(g_pipes.tf, &.{ .{ .b = c.scratch, .off = src }, .{ .b = c.model.signs_buf, .off = 0 }, .{ .b = c.scratch, .off = dst } }, &p, nblk, rows, 1);
}

fn rmsDispatch(c: *Context, src: u64, dst: u64, n: u32, woff: u32, has_w: u32, rows: u32) void {
    const p = pcBuf(&.{ n, fbits(1e-6), woff, has_w });
    dispatch(g_pipes.rmsnorm, &.{ .{ .b = c.scratch, .off = src }, .{ .b = c.model.weights, .off = 0 }, .{ .b = c.scratch, .off = dst } }, &p, rows, 1, 1);
}

fn matvecPtq(c: *Context, x: u64, w_off_words: u32, n_in: u32, n_out: u32, y: u64) void {
    const p = pcBuf(&.{ n_in, w_off_words, 0, 0 });
    dispatch(g_pipes.matvec_ptq, &.{ .{ .b = c.scratch, .off = x }, .{ .b = c.model.weights, .off = 0 }, .{ .b = c.scratch, .off = y } }, &p, n_out, 1, 1);
}

fn matvecF32(c: *Context, x: u64, w_off_elems: u32, n_in: u32, n_out: u32, y: u64) void {
    const p = pcBuf(&.{ n_in, w_off_elems, 0, 0 });
    dispatch(g_pipes.matvec_f32, &.{ .{ .b = c.scratch, .off = x }, .{ .b = c.model.weights, .off = 0 }, .{ .b = c.scratch, .off = y } }, &p, n_out, 1, 1);
}

fn ewDispatch(c: *Context, a: u64, b: u64, y: u64, n: u32, op: u32) void {
    const p = pcBuf(&.{ n, op, 0, 0 });
    dispatch(g_pipes.ew, &.{ .{ .b = c.scratch, .off = a }, .{ .b = c.scratch, .off = b }, .{ .b = c.scratch, .off = y } }, &p, (n + 255) / 256, 1, 1);
}

fn evalOne(c: *Context, tok: i32, want_logits: bool) !void {
    const m = c.model;
    const tok_u: u32 = @intCast(tok);
    // reset descriptor pool and allocate a fresh batch of sets for this token
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

    var nb: [128]u8 = undefined;
    var il: u32 = 0;
    while (il < N_LAYER) : (il += 1) {
        const full = (il % FULL_ATTN_INTERVAL) == (FULL_ATTN_INTERVAL - 1);
        var an = std.fmt.bufPrint(&nb, "blk.{d}.attn_norm.weight", .{il}) catch return error.Name;
        rmsDispatch(c, SCR_X, SCR_H, N_EMBD, try tensorF32Off(m, an), 1, 1);
        tfDispatch(c, SCR_H, SCR_HF, N_EMBD, SIGNS_2560, 0, 1);
        if (!full) {
            var qkv_name = std.fmt.bufPrint(&nb, "blk.{d}.attn_qkv.weight", .{il}) catch return error.Name;
            var t = m.find(qkv_name) orelse return error.NoTensor;
            matvecPtq(c, SCR_HF, @intCast(t.offset / 4), N_EMBD, QKV_DIM, SCR_QKV);
            var gname = std.fmt.bufPrint(&nb, "blk.{d}.attn_gate.weight", .{il}) catch return error.Name;
            t = m.find(gname) orelse return error.NoTensor;
            matvecPtq(c, SCR_HF, @intCast(t.offset / 4), N_EMBD, INNER, SCR_Z);
            var alname = std.fmt.bufPrint(&nb, "blk.{d}.ssm_alpha.weight", .{il}) catch return error.Name;
            var betaname = std.fmt.bufPrint(&nb, "blk.{d}.ssm_beta.weight", .{il}) catch return error.Name;
            matvecF32(c, SCR_HF, try tensorF32Off(m, alname), N_EMBD, 32, SCR_GB);
            matvecF32(c, SCR_HF, try tensorF32Off(m, betaname), N_EMBD, 32, SCR_GB + 128);
            {
                var a_name = std.fmt.bufPrint(&nb, "blk.{d}.ssm_a", .{il}) catch return error.Name;
                var dt_name = std.fmt.bufPrint(&nb, "blk.{d}.ssm_dt.bias", .{il}) catch return error.Name;
                const p = pcBuf(&.{ try tensorF32Off(m, alname), try tensorF32Off(m, betaname), try tensorF32Off(m, a_name), try tensorF32Off(m, dt_name) });
                dispatch(g_pipes.gates, &.{ .{ .b = m.weights, .off = 0 }, .{ .b = c.scratch, .off = SCR_GB } }, &p, 1, 1, 1);
            }
            {
                var cw_name = std.fmt.bufPrint(&nb, "blk.{d}.ssm_conv1d.weight", .{il}) catch return error.Name;
                const p = pcBuf(&.{ QKV_DIM, try tensorF32Off(m, cw_name), 0, 0 });
                dispatch(g_pipes.conv_silu, &.{ .{ .b = c.scratch, .off = SCR_QKV }, .{ .b = m.weights, .off = 0 }, .{ .b = c.convhist, .off = @as(u64, slotOf(il)) * 3 * QKV_DIM * 4 }, .{ .b = c.scratch, .off = SCR_QKV2 } }, &p, QKV_DIM / 256, 1, 1);
            }
            {
                const p = pcBuf(&.{ 32, 0, 0, 0 });
                dispatch(g_pipes.l2norm, &.{.{ .b = c.scratch, .off = SCR_QKV2 }}, &p, 32, 1, 1);
            }
            {
                var nw_name = std.fmt.bufPrint(&nb, "blk.{d}.ssm_norm.weight", .{il}) catch return error.Name;
                const p = pcBuf(&.{ try tensorF32Off(m, nw_name), 0, 0, 0 });
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
            var oname = std.fmt.bufPrint(&nb, "blk.{d}.ssm_out.weight", .{il}) catch return error.Name;
            t = m.find(oname) orelse return error.NoTensor;
            matvecPtq(c, SCR_ATTNF, @intCast(t.offset / 4), INNER, N_EMBD, SCR_Y);
            ewDispatch(c, SCR_X, SCR_Y, SCR_X, N_EMBD, 0);
        } else {
            const fl = il / FULL_ATTN_INTERVAL;
            var qname = std.fmt.bufPrint(&nb, "blk.{d}.attn_q.weight", .{il}) catch return error.Name;
            var t = m.find(qname) orelse return error.NoTensor;
            matvecPtq(c, SCR_HF, @intCast(t.offset / 4), N_EMBD, QKV_DIM, SCR_QKV);
            var kname = std.fmt.bufPrint(&nb, "blk.{d}.attn_k.weight", .{il}) catch return error.Name;
            t = m.find(kname) orelse return error.NoTensor;
            matvecPtq(c, SCR_HF, @intCast(t.offset / 4), N_EMBD, 1024, SCR_KRAW);
            var vname = std.fmt.bufPrint(&nb, "blk.{d}.attn_v.weight", .{il}) catch return error.Name;
            t = m.find(vname) orelse return error.NoTensor;
            matvecPtq(c, SCR_HF, @intCast(t.offset / 4), N_EMBD, 1024, SCR_VRAW);
            var qn = std.fmt.bufPrint(&nb, "blk.{d}.attn_q_norm.weight", .{il}) catch return error.Name;
            var kn = std.fmt.bufPrint(&nb, "blk.{d}.attn_k_norm.weight", .{il}) catch return error.Name;
            rmsDispatch(c, SCR_QKV, SCR_QKV, 256, try tensorF32Off(m, qn), 1, 16);
            rmsDispatch(c, SCR_KRAW, SCR_KRAW, 256, try tensorF32Off(m, kn), 1, 4);
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
            var oname = std.fmt.bufPrint(&nb, "blk.{d}.attn_output.weight", .{il}) catch return error.Name;
            t = m.find(oname) orelse return error.NoTensor;
            matvecPtq(c, SCR_ATTNF, @intCast(t.offset / 4), INNER, N_EMBD, SCR_Y);
            ewDispatch(c, SCR_X, SCR_Y, SCR_X, N_EMBD, 0);
        }
        // FFN
        var pn = std.fmt.bufPrint(&nb, "blk.{d}.post_attention_norm.weight", .{il}) catch return error.Name;
        rmsDispatch(c, SCR_X, SCR_H, N_EMBD, try tensorF32Off(m, pn), 1, 1);
        tfDispatch(c, SCR_H, SCR_HF, N_EMBD, SIGNS_2560, 0, 1);
        var fg = std.fmt.bufPrint(&nb, "blk.{d}.ffn_gate.weight", .{il}) catch return error.Name;
        var fu = std.fmt.bufPrint(&nb, "blk.{d}.ffn_up.weight", .{il}) catch return error.Name;
        var t2 = m.find(fg) orelse return error.NoTensor;
        matvecPtq(c, SCR_HF, @intCast(t2.offset / 4), N_EMBD, N_FF, SCR_FF);
        t2 = m.find(fu) orelse return error.NoTensor;
        matvecPtq(c, SCR_HF, @intCast(t2.offset / 4), N_EMBD, N_FF, SCR_FU);
        ewDispatch(c, SCR_FF, SCR_FU, SCR_FA, N_FF, 1);
        tfDispatch(c, SCR_FA, SCR_FA, N_FF, SIGNS_9216, 0, 1);
        var fd = std.fmt.bufPrint(&nb, "blk.{d}.ffn_down.weight", .{il}) catch return error.Name;
        t2 = m.find(fd) orelse return error.NoTensor;
        matvecPtq(c, SCR_FA, @intCast(t2.offset / 4), N_FF, N_EMBD, SCR_Y);
        ewDispatch(c, SCR_X, SCR_Y, SCR_X, N_EMBD, 0);
    }
    // final norm + logits
    const on = m.find("output_norm.weight") orelse return error.NoTensor;
    rmsDispatch(c, SCR_X, SCR_H, N_EMBD, @intCast(on.offset / 4), 1, 1);
    tfDispatch(c, SCR_H, SCR_HF, N_EMBD, SIGNS_2560, 0, 1);
    if (want_logits) {
        matvecPtq(c, SCR_HF, @intCast(emb.offset / 4), N_EMBD, @intCast(m.n_vocab), SCR_LOGITS);
        vk.copyBufferO(&g_ctx, c.scratch, SCR_LOGITS, c.readback, 0, @as(u64, @intCast(m.n_vocab)) * 4);
    }
    try vk.endSubmitWait(&g_ctx);
    if (want_logits) {
        const dstf: [*]f32 = undefined;
        _ = dstf;
    }
}

export fn bonsai_context_create(
    model: ?*ModelOpaque,
    n_ctx: i32,
    n_batch: i32,
    out_ctx: ?*?*ContextOpaque,
) callconv(.c) i32 {
    _ = n_batch;
    if (out_ctx) |p| p.* = null;
    if (model == null or out_ctx == null) {
        setErr("invalid argument", .{});
        return -1;
    }
    if (n_ctx < 1 or n_ctx > 16384) {
        setErr("invalid n_ctx: {d}", .{n_ctx});
        return -1;
    }
    const m: *Model = @ptrCast(@alignCast(model.?));
    if (!m.have_gpu) {
        setErr("model has no Vulkan weights (GPU init failed at open)", .{});
        return -5;
    }
    const c = alloc.create(Context) catch return -2;
    c.* = .{ .model = m, .n_ctx = n_ctx };
    ctxAlloc(c) catch |e| {
        setErr("context allocation failed: {s}", .{@errorName(e)});
        alloc.destroy(c);
        return -5;
    };
    out_ctx.?.* = @ptrCast(c);
    return 0;
}

fn ctxAlloc(c: *Context) !void {
    const m = c.model;
    const usage: u32 = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const usage_td: u32 = usage | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const dl: u32 = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    const hv: u32 = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    c.scratch = try vk.createBuffer(&g_ctx, SCR_TOTAL, usage | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, dl);
    c.kv_layer_stride = (@as(u64, @intCast(c.n_ctx)) * 1024 * 2 + 255) / 256 * 256;
    c.kv = try vk.createBuffer(&g_ctx, 16 * c.kv_layer_stride, usage, dl);
    c.states = try vk.createBuffer(&g_ctx, 24 * 32 * 128 * 128 * 4, usage_td, dl);
    c.convhist = try vk.createBuffer(&g_ctx, 24 * 3 * QKV_DIM * 4, usage_td, dl);
    c.scores = try vk.createBuffer(&g_ctx, 16 * 16384 * 4, usage, dl);
    c.readback = try vk.createBuffer(&g_ctx, @as(u64, @intCast(m.n_vocab)) * 4, usage_td, hv);
    c.rb_ptr = (try vk.mapBuffer(&g_ctx, c.readback)).ptr;
    try vk.beginCommands(&g_ctx);
    vk.fillBuffer(&g_ctx, c.states, 0, c.states.size, 0);
    vk.fillBuffer(&g_ctx, c.convhist, 0, c.convhist.size, 0);
    try vk.endSubmitWait(&g_ctx);
    c.pos = 0;
}

export fn bonsai_context_close(ctx: ?*ContextOpaque) callconv(.c) void {
    if (ctx == null) return;
    const c: *Context = @ptrCast(@alignCast(ctx.?));
    alloc.destroy(c);
}

export fn bonsai_context_reset(ctx: ?*ContextOpaque) callconv(.c) i32 {
    if (ctx == null) return -1;
    const c: *Context = @ptrCast(@alignCast(ctx.?));
    vk.beginCommands(&g_ctx) catch return -5;
    vk.fillBuffer(&g_ctx, c.states, 0, c.states.size, 0);
    vk.fillBuffer(&g_ctx, c.convhist, 0, c.convhist.size, 0);
    vk.endSubmitWait(&g_ctx) catch return -5;
    c.pos = 0;
    return 0;
}

export fn bonsai_eval(
    ctx: ?*ContextOpaque,
    tokens: ?[*]const i32,
    n_tokens: i32,
    logits: ?[*]f32,
    logits_capacity: i32,
) callconv(.c) i32 {
    if (ctx == null or tokens == null or n_tokens < 1) {
        setErr("invalid argument", .{});
        return -1;
    }
    const c: *Context = @ptrCast(@alignCast(ctx.?));
    const m = c.model;
    if (logits != null and logits_capacity < m.n_vocab) {
        setErr("logits capacity {d} < n_vocab {d}", .{ logits_capacity, m.n_vocab });
        return -1;
    }
    var i: i32 = 0;
    while (i < n_tokens) : (i += 1) {
        const t = tokens.?[@intCast(i)];
        if (t < 0 or t >= m.n_vocab) {
            setErr("token id {d} out of range", .{t});
            return -1;
        }
    }
    if (c.pos + n_tokens > c.n_ctx) {
        setErr("context overflow: pos {d} + {d} > {d}", .{ c.pos, n_tokens, c.n_ctx });
        return -1;
    }
    i = 0;
    while (i < n_tokens) : (i += 1) {
        const last = (i == n_tokens - 1);
        evalOne(c, tokens.?[@intCast(i)], last and logits != null) catch |e| {
            setErr("eval failed: {s}", .{@errorName(e)});
            return -6;
        };
        c.pos += 1;
    }
    if (logits != null) {
        const nv: usize = @intCast(m.n_vocab);
        const src: [*]const f32 = @ptrCast(@alignCast(c.rb_ptr));
        @memcpy(logits.?[0..nv], src[0..nv]);
    }
    return 0;
}

export fn bonsai_generate_greedy(
    ctx: ?*ContextOpaque,
    prompt: ?[*]const i32,
    n_prompt: i32,
    n_predict: i32,
    flags: u32,
    out_ids: ?[*]i32,
) callconv(.c) i32 {
    _ = flags;
    if (ctx == null or out_ids == null or n_predict < 0 or n_prompt < 0) {
        setErr("invalid argument", .{});
        return -1;
    }
    const c: *Context = @ptrCast(@alignCast(ctx.?));
    const m = c.model;
    var nv: usize = @intCast(m.n_vocab);
    const buf = alloc.alloc(f32, nv) catch return -2;
    defer alloc.free(buf);
    if (n_prompt > 0) {
        const rc = bonsai_eval(ctx, prompt, n_prompt, buf.ptr, m.n_vocab);
        if (rc != 0) return rc;
    }
    var gen: i32 = 0;
    var next: i32 = 0;
    var have_next = false;
    while (gen < n_predict) : (gen += 1) {
        if (!have_next) {
            if (n_prompt == 0) break;
            var best: f32 = -std.math.inf(f32);
            var bi: i32 = 0;
            for (buf[0..nv]) |v, idx| {
                if (v > best) {
                    best = v;
                    bi = @intCast(idx);
                }
            }
            next = bi;
            have_next = true;
        }
        out_ids.?[@intCast(gen)] = next;
        const tok_arr: [1]i32 = .{next};
        const rc = bonsai_eval(ctx, &tok_arr, 1, buf.ptr, m.n_vocab);
        if (rc != 0) return rc;
        have_next = false;
        if (next == m.eos_id) {
            gen += 1;
            break;
        }
    }
    _ = &nv;
    return gen;
}

export fn bonsai_tokenize(
    model: ?*const ModelOpaque,
    utf8: ?[*:0]const u8,
    add_bos: i32,
    ids: ?[*]i32,
    capacity: i32,
) callconv(.c) i32 {
    _ = add_bos;
    if (model == null or utf8 == null) {
        setErr("invalid argument", .{});
        return -1;
    }
    const m: *const Model = @ptrCast(@alignCast(model.?));
    if (m.tokens.len == 0) {
        setErr("tokenizer vocabulary missing", .{});
        return -3;
    }
    const text = std.mem.span(utf8.?);
    const needed: i32 = @intCast(text.len);
    if (ids == null) return needed;
    if (capacity < needed) {
        setErr("capacity {d} < required {d}", .{ capacity, needed });
        return -1;
    }
    var i: usize = 0;
    while (i < text.len) : (i += 1) {
        const b: [1]u8 = .{text[i]};
        ids.?[i] = findToken(m, &b);
    }
    return @intCast(i);
}

fn findToken(m: *const Model, b: []const u8) i32 {
    for (m.tokens, 0..) |t, i| {
        const tn = if (t.len > 0 and t[t.len - 1] == 0) t[0 .. t.len - 1] else t;
        if (std.mem.eql(u8, tn, b)) return @intCast(i);
    }
    return 0;
}
