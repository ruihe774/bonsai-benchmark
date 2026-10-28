const std = @import("std");
const vk = @import("vk.zig");
const c = vk.c;
const gg = @import("model.zig");
const tok = @import("tokenizer.zig");
const shaders = @import("shaders");
const alloc = std.heap.c_allocator;
const batch_size = 512;
pub const Params = extern struct { w: u64 = 0, x: u64 = 0, y: u64 = 0, z: u64 = 0, s: u64 = 0, t: u64 = 0, u: u64 = 0, v: u64 = 0, a: u32 = 0, b: u32 = 0, c: u32 = 0, d: u32 = 0, e: u32 = 0, f: u32 = 0, g: u32 = 0, h: u32 = 0 };
const Op = enum { matvec, hadamard, norm, element, embed, conv, gdn, attprep, attscore, softmax, attvalue, argmax, gdn_batch, matvec_fast, dequant, cast, matmul_f16, flash, attreduce, matvec_vocab };
const op_count = 20;
pub const Model = struct {
    refs: usize,
    gguf: gg.Model,
    tokenizer: tok.Tokenizer,
    gpu: vk.Runtime,
    weights: vk.Buffer,
    signs: vk.Buffer,
    pipes: [op_count]vk.Pipeline,
    n_vocab: u32,
    n_ctx_train: u32,
    eos: i32,
    pub fn open(path: []const u8) !*Model {
        const self = try alloc.create(Model);
        self.refs = 1;
        errdefer alloc.destroy(self);
        self.gguf = try gg.Model.open(alloc, path);
        errdefer self.gguf.deinit();
        if (!std.mem.eql(u8, self.gguf.getString("general.architecture") orelse "", "qwen35") or
            self.gguf.getUInt("qwen35.embedding_length") != 2560 or
            self.gguf.getUInt("qwen35.block_count") != 32 or
            self.gguf.getUInt("qwen35.feed_forward_length") != 9216 or
            self.gguf.getUInt("prism.hadamard.block_size") != 512) return error.UnsupportedArchitecture;
        self.tokenizer = try tok.Tokenizer.init(alloc, &self.gguf);
        errdefer self.tokenizer.deinit();
        self.n_vocab = @intCast((self.gguf.getArray("tokenizer.ggml.tokens") orelse return error.MissingVocabulary).len);
        const train_context = self.gguf.getUInt("qwen35.context_length") orelse return error.MissingContextLength;
        const eos_id = self.gguf.getUInt("tokenizer.ggml.eos_token_id") orelse return error.MissingEosToken;
        if (self.n_vocab != 248320 or train_context != 262144 or eos_id >= self.n_vocab) return error.UnsupportedArchitecture;
        self.n_ctx_train = @intCast(train_context);
        self.eos = @intCast(eos_id);
        try self.validateTensors();
        self.gpu = try vk.Runtime.init();
        errdefer self.gpu.deinit();
        self.weights = try self.gpu.buffer(self.gguf.weights.len);
        errdefer self.gpu.freeBuffer(&self.weights);
        @memcpy(self.weights.ptr[0..self.gguf.weights.len], self.gguf.weights);
        const signs = self.gguf.getArray("prism.hadamard.sign_values") orelse return error.MissingSigns;
        if (signs.len != 15872) return error.InvalidSigns;
        self.signs = try self.gpu.buffer(signs.len * 4);
        errdefer self.gpu.freeBuffer(&self.signs);
        const dest: [*]f32 = @ptrCast(@alignCast(self.signs.ptr));
        for (signs, 0..) |s, i| {
            const value = s.asInt() orelse return error.InvalidSigns;
            if (value != 1 and value != -1) return error.InvalidSigns;
            dest[i] = @floatFromInt(value);
        }
        var initialized: usize = 0;
        errdefer for (self.pipes[0..initialized]) |pipeline| c.vkDestroyPipeline(self.gpu.device, pipeline, null);
        inline for (0..op_count) |i| {
            const name = @tagName(@as(Op, @fromBackingInt(@intCast(i))));
            self.pipes[i] = try self.gpu.pipeline(&@field(shaders, name));
            initialized += 1;
        }
        return self;
    }
    pub fn close(self: *Model) void {
        self.refs -= 1;
        if (self.refs != 0) return;
        _ = c.vkDeviceWaitIdle(self.gpu.device);
        for (self.pipes) |p| c.vkDestroyPipeline(self.gpu.device, p, null);
        self.gpu.freeBuffer(&self.signs);
        self.gpu.freeBuffer(&self.weights);
        self.gpu.deinit();
        self.tokenizer.deinit();
        self.gguf.deinit();
        alloc.destroy(self);
    }
    fn expectTensor(self: *const Model, name: []const u8, x: u64, y: u64, typ: u32) !void {
        const t = try self.tensor(name);
        if (t.typ != typ or t.n_dims != (if (y == 0) @as(u32, 1) else 2) or t.dims[0] != x or t.dims[1] != (if (y == 0) @as(u64, 1) else y)) return error.UnsupportedTensorShape;
    }
    fn expectLayer(self: *const Model, l: u32, suffix: []const u8, x: u64, y: u64, typ: u32) !void {
        var buf: [96]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, "blk.{d}.{s}", .{ l, suffix });
        try self.expectTensor(name, x, y, typ);
    }
    fn validateTensors(self: *const Model) !void {
        inline for (.{ .{ "qwen35.attention.head_count", 16 }, .{ "qwen35.attention.head_count_kv", 4 }, .{ "qwen35.attention.key_length", 256 }, .{ "qwen35.attention.value_length", 256 }, .{ "qwen35.ssm.conv_kernel", 4 }, .{ "qwen35.ssm.state_size", 128 }, .{ "qwen35.ssm.group_count", 16 }, .{ "qwen35.ssm.time_step_rank", 32 }, .{ "qwen35.ssm.inner_size", 4096 }, .{ "qwen35.full_attention_interval", 4 }, .{ "qwen35.rope.dimension_count", 64 } }) |item| {
            if (self.gguf.getUInt(item[0]) != item[1]) return error.UnsupportedArchitecture;
        }
        try self.expectTensor("token_embd.weight", 2560, 248320, 143);
        try self.expectTensor("output_norm.weight", 2560, 0, 0);
        for (0..32) |li| {
            const l: u32 = @intCast(li);
            try self.expectLayer(l, "attn_norm.weight", 2560, 0, 0);
            try self.expectLayer(l, "post_attention_norm.weight", 2560, 0, 0);
            try self.expectLayer(l, "ffn_gate.weight", 2560, 9216, 143);
            try self.expectLayer(l, "ffn_up.weight", 2560, 9216, 143);
            try self.expectLayer(l, "ffn_down.weight", 9216, 2560, 143);
            if ((l + 1) % 4 != 0) {
                try self.expectLayer(l, "attn_qkv.weight", 2560, 8192, 143);
                try self.expectLayer(l, "attn_gate.weight", 2560, 4096, 143);
                try self.expectLayer(l, "ssm_out.weight", 4096, 2560, 143);
                try self.expectLayer(l, "ssm_alpha.weight", 2560, 32, 0);
                try self.expectLayer(l, "ssm_beta.weight", 2560, 32, 0);
                try self.expectLayer(l, "ssm_a", 32, 0, 0);
                try self.expectLayer(l, "ssm_dt.bias", 32, 0, 0);
                try self.expectLayer(l, "ssm_norm.weight", 128, 0, 0);
                try self.expectLayer(l, "ssm_conv1d.weight", 4, 8192, 0);
            } else {
                try self.expectLayer(l, "attn_q.weight", 2560, 8192, 143);
                try self.expectLayer(l, "attn_k.weight", 2560, 1024, 143);
                try self.expectLayer(l, "attn_v.weight", 2560, 1024, 143);
                try self.expectLayer(l, "attn_output.weight", 4096, 2560, 143);
                try self.expectLayer(l, "attn_q_norm.weight", 256, 0, 0);
                try self.expectLayer(l, "attn_k_norm.weight", 256, 0, 0);
            }
        }
    }
    fn tensor(self: *const Model, name: []const u8) !*const gg.Tensor {
        return self.gguf.findTensor(name) orelse error.MissingTensor;
    }
    fn addr(self: *const Model, name: []const u8) !u64 {
        const t = try self.tensor(name);
        return self.weights.addr + t.offset;
    }
    fn sign(self: *const Model, width: u32) u64 {
        return self.signs.addr + @as(u64, switch (width) {
            2560 => @as(u32, 0),
            4096 => 2560,
            9216 => 6656,
            else => 0,
        }) * 4;
    }
    fn dispatch(self: *Model, op: Op, p: Params, x: u32, y: u32) void {
        self.gpu.profile_op = @backingInt(op);
        self.gpu.dispatch(self.pipes[@backingInt(op)], p, x, y, 1);
    }
};
pub const Context = struct {
    model: *Model,
    n_ctx: u32,
    kv_stride: u32,
    pos: u32 = 0,
    work: vk.Buffer,
    state: vk.Buffer,
    scratch: vk.Buffer,
    weights16: u64,
    inputs16: u64,
    x: u64,
    norm: u64,
    trans: u64,
    qkv: u64,
    gate: u64,
    alpha: u64,
    beta: u64,
    conv: u64,
    att: u64,
    tmp: u64,
    fg: u64,
    fu: u64,
    mid: u64,
    down: u64,
    k: u64,
    v: u64,
    q: u64,
    qgate: u64,
    scores: u64,
    logits: u64,
    argmax: u64,
    gdn_state: u64,
    conv_state: u64,
    kv: u64,
    pub fn create(m: *Model, n_ctx: u32) !*Context {
        const self = try alloc.create(Context);
        errdefer alloc.destroy(self);
        self.model = m;
        self.n_ctx = n_ctx;
        self.kv_stride = (n_ctx + 63) & ~@as(u32, 63);
        self.pos = 0;
        var off: u64 = 0;
        self.work = try m.gpu.buffer((248320 + 100000 * batch_size + @as(usize, n_ctx) * 16) * 4);
        errdefer m.gpu.freeBuffer(&self.work);
        self.x = self.reserve(&off, 2560);
        self.norm = self.reserve(&off, 2560);
        self.trans = self.reserve(&off, 9216);
        self.qkv = self.reserve(&off, 8192);
        self.gate = self.reserve(&off, 4096);
        self.alpha = self.reserve(&off, 32);
        self.beta = self.reserve(&off, 32);
        self.conv = self.reserve(&off, 8192);
        self.att = self.reserve(&off, 4096);
        self.tmp = self.reserve(&off, 4096);
        self.fg = self.reserve(&off, 9216);
        self.fu = self.reserve(&off, 9216);
        self.mid = self.reserve(&off, 9216);
        self.down = self.reserve(&off, 2560);
        self.k = self.reserve(&off, 1024);
        self.v = self.reserve(&off, 1024);
        self.q = self.reserve(&off, 4096);
        self.qgate = self.reserve(&off, 4096);
        self.scores = self.reserveFixed(&off, @as(u64, n_ctx) * 16);
        self.logits = self.reserveFixed(&off, 248320);
        self.argmax = self.reserveFixed(&off, 512);
        self.scratch = try m.gpu.buffer(2560 * 9216 * 2 + batch_size * 9216 * 2);
        errdefer m.gpu.freeBuffer(&self.scratch);
        self.weights16 = self.scratch.addr;
        self.inputs16 = self.weights16 + 2560 * 9216 * 2;
        const gdn_bytes: u64 = 24 * 32 * 128 * 128 * 4;
        const conv_bytes: u64 = 24 * 8192 * 3 * 4;
        const kv_bytes: u64 = 8 * 2 * @as(u64, self.kv_stride) * 1024 * 2;
        self.state = try m.gpu.buffer(@intCast(gdn_bytes + conv_bytes + kv_bytes));
        errdefer m.gpu.freeBuffer(&self.state);
        self.gdn_state = self.state.addr;
        self.conv_state = self.gdn_state + gdn_bytes;
        self.kv = self.conv_state + conv_bytes;
        try self.reset();
        m.refs += 1;
        return self;
    }
    fn reserve(self: *Context, off: *u64, n: u64) u64 {
        return self.reserveFixed(off, n * batch_size);
    }
    fn reserveFixed(self: *Context, off: *u64, n: u64) u64 {
        const a = self.work.addr + off.*;
        off.* += (n * 4 + 255) & ~@as(u64, 255);
        return a;
    }
    pub fn close(self: *Context) void {
        const model = self.model;
        _ = c.vkDeviceWaitIdle(model.gpu.device);
        model.gpu.freeBuffer(&self.scratch);
        model.gpu.freeBuffer(&self.state);
        model.gpu.freeBuffer(&self.work);
        alloc.destroy(self);
        model.close();
    }
    pub fn reset(self: *Context) !void {
        const gpu = &self.model.gpu;
        try gpu.begin();
        const count: u32 = @intCast(self.state.size / 4);
        const groups = (count + 255) / 256;
        const x = @min(groups, 65535);
        const y = (groups + x - 1) / x;
        self.model.dispatch(.element, .{ .y = self.state.addr, .a = count, .b = 4 }, x, y);
        try gpu.finish();
        self.pos = 0;
    }
    fn had(self: *Context, x: u64, y: u64, width: u32, inverse: bool) void {
        self.model.dispatch(.hadamard, .{ .w = self.model.sign(width), .x = x, .y = y, .a = width, .b = @intFromBool(inverse) }, width / 512, 1);
    }
    fn mat(self: *Context, name: []const u8, x: u64, y: u64) !void {
        const t = try self.model.tensor(name);
        const wide = t.typ == 143 and t.dims[1] > 65536;
        const op: Op = if (wide) .matvec_vocab else if (t.typ == 143) .matvec_fast else .matvec;
        const rows_per_group: u64 = if (wide) 32 else if (t.typ == 143) 16 else 4;
        self.model.dispatch(op, .{ .w = self.model.weights.addr + t.offset, .x = x, .y = y, .a = @intCast(t.dims[0]), .b = @intCast(t.dims[1]), .c = t.typ }, @intCast((t.dims[1] + rows_per_group - 1) / rows_per_group), 1);
    }
    fn normOp(self: *Context, name: []const u8, x: u64, y: u64, width: u32, heads: u32, z: u64) !void {
        self.model.dispatch(.norm, .{ .w = try self.model.addr(name), .x = x, .y = y, .z = z, .a = width, .b = if (z != 0) 2 else 0 }, heads, 1);
    }
    fn elem(self: *Context, x: u64, z: u64, y: u64, n: u32, op: u32) void {
        self.model.dispatch(.element, .{ .x = x, .z = z, .y = y, .a = n, .b = op }, (n + 255) / 256, 1);
    }
    fn layerName(buf: []u8, layer: u32, suffix: []const u8) ![]const u8 {
        return std.fmt.bufPrint(buf, "blk.{d}.{s}", .{ layer, suffix });
    }
    fn layerMat(self: *Context, layer: u32, suffix: []const u8, x: u64, y: u64) !void {
        var buf: [96]u8 = undefined;
        try self.mat(try layerName(&buf, layer, suffix), x, y);
    }
    fn layerAddr(self: *Context, layer: u32, suffix: []const u8) !u64 {
        var buf: [96]u8 = undefined;
        return self.model.addr(try layerName(&buf, layer, suffix));
    }
    fn layerNorm(self: *Context, layer: u32, suffix: []const u8, x: u64, y: u64, width: u32, heads: u32, z: u64) !void {
        var buf: [96]u8 = undefined;
        try self.normOp(try layerName(&buf, layer, suffix), x, y, width, heads, z);
    }
    fn forward(self: *Context, token: i32, with_logits: bool) !void {
        const m = self.model;
        m.dispatch(.embed, .{ .w = try m.addr("token_embd.weight"), .y = self.tmp, .a = 2560, .b = @intCast(token) }, 10, 1);
        self.had(self.tmp, self.x, 2560, true);
        var linear: u64 = 0;
        var full: u64 = 0;
        for (0..32) |li| {
            const l: u32 = @intCast(li);
            try self.layerNorm(l, "attn_norm.weight", self.x, self.norm, 2560, 1, 0);
            self.had(self.norm, self.trans, 2560, false);
            if ((l + 1) % 4 != 0) {
                try self.layerMat(l, "attn_qkv.weight", self.trans, self.qkv);
                try self.layerMat(l, "attn_gate.weight", self.trans, self.gate);
                try self.layerMat(l, "ssm_alpha.weight", self.norm, self.alpha);
                try self.layerMat(l, "ssm_beta.weight", self.norm, self.beta);
                m.dispatch(.conv, .{ .w = try self.layerAddr(l, "ssm_conv1d.weight"), .x = self.qkv, .y = self.conv, .s = self.conv_state + linear * 8192 * 3 * 4 }, 32, 1);
                m.dispatch(.gdn, .{ .w = try self.layerAddr(l, "ssm_dt.bias"), .x = self.conv, .y = self.att, .z = try self.layerAddr(l, "ssm_a"), .s = self.gdn_state + linear * 32 * 128 * 128 * 4, .t = self.alpha, .u = self.beta }, 64, 1);
                try self.layerNorm(l, "ssm_norm.weight", self.att, self.tmp, 128, 32, self.gate);
                self.had(self.tmp, self.trans, 4096, false);
                try self.layerMat(l, "ssm_out.weight", self.trans, self.down);
                linear += 1;
            } else {
                try self.layerMat(l, "attn_q.weight", self.trans, self.qkv);
                try self.layerMat(l, "attn_k.weight", self.trans, self.k);
                try self.layerMat(l, "attn_v.weight", self.trans, self.v);
                const cache = self.kv + full * 2 * @as(u64, self.kv_stride) * 1024 * 2;
                m.dispatch(.attprep, .{ .w = try self.layerAddr(l, "attn_q_norm.weight"), .x = self.qkv, .y = self.q, .z = self.qgate, .a = self.pos, .b = 0 }, 16, 1);
                m.dispatch(.attprep, .{ .w = try self.layerAddr(l, "attn_k_norm.weight"), .x = self.k, .y = cache, .z = cache + @as(u64, self.kv_stride) * 1024 * 2, .s = self.v, .a = self.pos, .b = 1 }, 4, 1);
                m.dispatch(.attscore, .{ .w = cache, .x = self.q, .y = self.scores, .a = self.pos + 1, .b = self.n_ctx }, (self.pos + 4) / 4, 16);
                m.dispatch(.softmax, .{ .x = self.scores, .a = self.pos + 1, .b = self.n_ctx }, 16, 1);
                const splits = @min(128, (self.pos + 64) / 64);
                m.dispatch(.attvalue, .{ .w = cache + @as(u64, self.kv_stride) * 1024 * 2, .x = self.scores, .y = if (splits > 1) self.tmp else self.att, .a = self.pos + 1, .b = self.n_ctx, .c = splits }, 16, splits);
                if (splits > 1) m.dispatch(.attreduce, .{ .x = self.tmp, .y = self.att, .a = splits }, 16, 1);
                self.elem(self.att, self.qgate, self.tmp, 4096, 2);
                self.had(self.tmp, self.trans, 4096, false);
                try self.layerMat(l, "attn_output.weight", self.trans, self.down);
                full += 1;
            }
            self.elem(self.x, self.down, self.x, 2560, 0);
            try self.layerNorm(l, "post_attention_norm.weight", self.x, self.norm, 2560, 1, 0);
            self.had(self.norm, self.trans, 2560, false);
            try self.layerMat(l, "ffn_gate.weight", self.trans, self.fg);
            try self.layerMat(l, "ffn_up.weight", self.trans, self.fu);
            self.elem(self.fg, self.fu, self.mid, 9216, 1);
            self.had(self.mid, self.trans, 9216, false);
            try self.layerMat(l, "ffn_down.weight", self.trans, self.down);
            self.elem(self.x, self.down, self.x, 2560, 0);
        }
        if (with_logits) {
            try self.normOp("output_norm.weight", self.x, self.norm, 2560, 1, 0);
            self.had(self.norm, self.trans, 2560, false);
            try self.mat("token_embd.weight", self.trans, self.logits);
        }
        self.pos += 1;
    }
    fn hadBatch(self: *Context, x: u64, y: u64, width: u32, inverse: bool, n: u32) void {
        self.model.dispatch(.hadamard, .{ .w = self.model.sign(width), .x = x, .y = y, .a = width, .b = @intFromBool(inverse) }, width / 512, n);
    }
    fn matBatch(self: *Context, name: []const u8, x: u64, y: u64, n: u32) !void {
        const t = try self.model.tensor(name);
        const width: u32 = @intCast(t.dims[0]);
        const rows: u32 = @intCast(t.dims[1]);
        const count = width * rows;
        // Expand only this projection, never a dequantized model copy.
        self.model.dispatch(.dequant, .{ .w = self.model.weights.addr + t.offset, .y = self.weights16, .a = count, .c = t.typ }, if (t.typ == 143) (count + 511) / 512 else (count + 127) / 128, 1);
        self.model.dispatch(.cast, .{ .x = x, .y = self.inputs16, .a = width * n }, (width * n + 255) / 256, 1);
        self.model.dispatch(.matmul_f16, .{ .w = self.weights16, .x = self.inputs16, .y = y, .a = width, .b = rows, .d = n }, (rows + 63) / 64, (n + 63) / 64);
    }
    fn layerMatBatch(self: *Context, l: u32, suffix: []const u8, x: u64, y: u64, n: u32) !void {
        var buf: [96]u8 = undefined;
        try self.matBatch(try layerName(&buf, l, suffix), x, y, n);
    }
    fn normBatch(self: *Context, l: u32, suffix: []const u8, x: u64, y: u64, width: u32, heads: u32, z: u64, n: u32) !void {
        self.model.dispatch(.norm, .{ .w = try self.layerAddr(l, suffix), .x = x, .y = y, .z = z, .a = width, .b = if (z != 0) 2 else 0 }, heads * n, 1);
    }
    fn forwardBatch(self: *Context, ids: []const i32, with_logits: bool) !void {
        const m = self.model;
        const n: u32 = @intCast(ids.len);
        const token_ptr: [*]i32 = @ptrCast(@alignCast(self.work.ptr + @as(usize, @intCast(self.argmax - self.work.addr))));
        @memcpy(token_ptr[0..ids.len], ids);
        m.dispatch(.embed, .{ .w = try m.addr("token_embd.weight"), .x = self.argmax, .y = self.tmp, .a = 2560, .c = 1 }, 10, n);
        self.hadBatch(self.tmp, self.x, 2560, true, n);
        var linear: u64 = 0;
        var full: u64 = 0;
        for (0..32) |li| {
            const l: u32 = @intCast(li);
            try self.normBatch(l, "attn_norm.weight", self.x, self.norm, 2560, 1, 0, n);
            self.hadBatch(self.norm, self.trans, 2560, false, n);
            if ((l + 1) % 4 != 0) {
                try self.layerMatBatch(l, "attn_qkv.weight", self.trans, self.qkv, n);
                try self.layerMatBatch(l, "attn_gate.weight", self.trans, self.gate, n);
                try self.layerMatBatch(l, "ssm_alpha.weight", self.norm, self.alpha, n);
                try self.layerMatBatch(l, "ssm_beta.weight", self.norm, self.beta, n);
                m.dispatch(.conv, .{ .w = try self.layerAddr(l, "ssm_conv1d.weight"), .x = self.qkv, .y = self.conv, .s = self.conv_state + linear * 8192 * 3 * 4, .a = n }, 32, 1);
                m.dispatch(.gdn_batch, .{ .w = try self.layerAddr(l, "ssm_dt.bias"), .x = self.conv, .y = self.att, .z = try self.layerAddr(l, "ssm_a"), .s = self.gdn_state + linear * 32 * 128 * 128 * 4, .t = self.alpha, .u = self.beta, .a = n }, 64, 1);
                try self.normBatch(l, "ssm_norm.weight", self.att, self.tmp, 128, 32, self.gate, n);
                self.hadBatch(self.tmp, self.trans, 4096, false, n);
                try self.layerMatBatch(l, "ssm_out.weight", self.trans, self.down, n);
                linear += 1;
            } else {
                try self.layerMatBatch(l, "attn_q.weight", self.trans, self.qkv, n);
                try self.layerMatBatch(l, "attn_k.weight", self.trans, self.k, n);
                try self.layerMatBatch(l, "attn_v.weight", self.trans, self.v, n);
                const cache = self.kv + full * 2 * @as(u64, self.kv_stride) * 1024 * 2;
                m.dispatch(.attprep, .{ .w = try self.layerAddr(l, "attn_q_norm.weight"), .x = self.qkv, .y = self.q, .z = self.qgate, .a = self.pos, .b = 0 }, 16, n);
                m.dispatch(.attprep, .{ .w = try self.layerAddr(l, "attn_k_norm.weight"), .x = self.k, .y = cache, .z = cache + @as(u64, self.kv_stride) * 1024 * 2, .s = self.v, .a = self.pos, .b = 1 }, 4, n);
                m.dispatch(.flash, .{ .w = cache, .x = self.q, .y = self.att, .z = cache + @as(u64, self.kv_stride) * 1024 * 2, .a = self.pos, .d = n }, (n + 15) / 16, 16);
                self.elem(self.att, self.qgate, self.tmp, 4096 * n, 2);
                self.hadBatch(self.tmp, self.trans, 4096, false, n);
                try self.layerMatBatch(l, "attn_output.weight", self.trans, self.down, n);
                full += 1;
            }
            self.elem(self.x, self.down, self.x, 2560 * n, 0);
            try self.normBatch(l, "post_attention_norm.weight", self.x, self.norm, 2560, 1, 0, n);
            self.hadBatch(self.norm, self.trans, 2560, false, n);
            try self.layerMatBatch(l, "ffn_gate.weight", self.trans, self.fg, n);
            try self.layerMatBatch(l, "ffn_up.weight", self.trans, self.fu, n);
            self.elem(self.fg, self.fu, self.mid, 9216 * n, 1);
            self.hadBatch(self.mid, self.trans, 9216, false, n);
            try self.layerMatBatch(l, "ffn_down.weight", self.trans, self.down, n);
            self.elem(self.x, self.down, self.x, 2560 * n, 0);
        }
        if (with_logits) {
            const last = self.x + @as(u64, n - 1) * 2560 * 4;
            try self.normOp("output_norm.weight", last, self.norm, 2560, 1, 0);
            self.had(self.norm, self.trans, 2560, false);
            try self.mat("token_embd.weight", self.trans, self.logits);
        }
        self.pos += n;
    }
    pub fn eval(self: *Context, tokens: []const i32, logits: ?[*]f32) !void {
        if (tokens.len == 0 or tokens.len > self.n_ctx - self.pos) return error.ContextCapacity;
        for (tokens) |id| if (id < 0 or id >= self.model.n_vocab) return error.InvalidToken;
        // A layer-major batch reuses packed weights and matrix cores for prefill.
        var i: usize = 0;
        while (i < tokens.len) {
            const end = @min(i + batch_size, tokens.len);
            try self.model.gpu.begin();
            if (end - i >= 8) {
                try self.forwardBatch(tokens[i..end], end == tokens.len);
            } else {
                for (tokens[i..end], i..) |id, j| try self.forward(id, j + 1 == tokens.len);
            }
            try self.model.gpu.finish();
            i = end;
        }
        if (logits) |out| {
            const off: usize = @intCast(self.logits - self.work.addr);
            const src: [*]const f32 = @ptrCast(@alignCast(self.work.ptr + off));
            @memcpy(out[0..self.model.n_vocab], src[0..self.model.n_vocab]);
        }
    }
    pub fn best(self: *Context) !i32 {
        try self.model.gpu.begin();
        const groups = (self.model.n_vocab + 1023) / 1024;
        self.model.dispatch(.argmax, .{ .x = self.logits, .y = self.argmax, .a = self.model.n_vocab }, groups, 1);
        self.model.dispatch(.argmax, .{ .x = self.argmax, .y = self.argmax, .a = groups, .b = 1 }, 1, 1);
        try self.model.gpu.finish();
        const ptr: *const i32 = @ptrCast(@alignCast(self.work.ptr + @as(usize, @intCast(self.argmax - self.work.addr))));
        return ptr.*;
    }
};
