// bonsai-vk: inference engine for Ternary-Bonsai-2-4B (qwen35) reading GGUF at runtime.
// GGUF parser, tokenizer metadata, exact PTQ1_0 dequantization, and Vulkan compute
// engine scaffolding (shaders embedded under src/spv).
const std = @import("std");
const vk = @import("vk.zig");
const tokmod = @import("tok.zig");

const alloc = std.heap.c_allocator;
const PTQ1_0: u32 = 143;

var g_last_lookup: [128]u8 = undefined;
var g_last_len: usize = 0;

// qwen35 architecture constants
const N_LAYER: u32 = 32;
const N_EMBD: u32 = 2560;
const N_FF: u32 = 9216;
const N_HEAD: u32 = 16;
const N_HEAD_KV: u32 = 4;
const HEAD_DIM: u32 = 256;
const INNER: u32 = 4096;
const N_VH: u32 = 32;
const N_KH: u32 = 16;
const SSM_STATE: u32 = 128;
const CONV_K: u32 = 4;
const QKV_DIM: u32 = 8192;
const FULL_ATTN_INTERVAL: u32 = 4;

threadlocal var err_buf: [512]u8 = undefined;
threadlocal var err_set: bool = false;

fn setErr(comptime fmt: []const u8, args: anytype) void {
    const s = std.fmt.bufPrint(&err_buf, fmt, args) catch "";
    _ = s;
    err_set = true;
}

pub const TensorInfo = struct {
    name: []u8,
    ttype: u32,
    ndims: u32,
    dims: [4]u64,
    offset: u64,
    size: u64,
    nelem: u64,
};

extern "c" fn open(path: [*:0]const u8, flags: c_int, mode: c_int) c_int;
extern "c" fn pread(fd: c_int, buf: [*]u8, count: usize, offset: i64) isize;
extern "c" fn close(fd: c_int) c_int;

const Rd = struct {
    fd: c_int,
    off: u64 = 0,
    buf: [131072]u8 = undefined,
    bpos: usize = 0,
    blen: usize = 0,

    fn fill(self: *Rd) !void {
        const n = pread(self.fd, &self.buf, self.buf.len, @intCast(self.off));
        if (n < 0) return error.ReadFailed;
        self.blen = @intCast(n);
        self.off += self.blen;
        self.bpos = 0;
        if (self.blen == 0) return error.Eof;
    }
    fn read(self: *Rd, dst: []u8) !void {
        var done: usize = 0;
        while (done < dst.len) {
            if (self.bpos >= self.blen) try self.fill();
            const n = @min(dst.len - done, self.blen - self.bpos);
            @memcpy(dst[done .. done + n], self.buf[self.bpos .. self.bpos + n]);
            self.bpos += n;
            done += n;
        }
    }
    fn u8v(self: *Rd) !u8 {
        var b: [1]u8 = undefined;
        try self.read(&b);
        return b[0];
    }
    fn u32v(self: *Rd) !u32 {
        var b: [4]u8 = undefined;
        try self.read(&b);
        return std.mem.readInt(u32, &b, .little);
    }
    fn u64v(self: *Rd) !u64 {
        var b: [8]u8 = undefined;
        try self.read(&b);
        return std.mem.readInt(u64, &b, .little);
    }
    fn str(self: *Rd) ![]u8 {
        const n = try self.u64v();
        if (n > 1 << 30) return error.BadString;
        const s = alloc.alloc(u8, @intCast(n)) catch return error.NoMem;
        try self.read(s);
        return s;
    }
    fn skip(self: *Rd, n: u64) !void {
        var buf: [4096]u8 = undefined;
        var left = n;
        while (left > 0) {
            const k: usize = @intCast(@min(left, buf.len));
            try self.read(buf[0..k]);
            left -= k;
        }
    }
};

const Model = struct {
    tensors: []TensorInfo = &[_]TensorInfo{},
    n_vocab: i32 = 0,
    n_ctx_train: i32 = 2048,
    eos_id: i32 = 2,
    bos_id: i32 = 1,
    add_bos: bool = false,
    data_start: u64 = 0,
    data_size: u64 = 0,
    fd: c_int = -1,
    signs: []f32 = &[_]f32{},
    tokens: [][]u8 = &[_][]u8{},
    tok_types: []i32 = &[_]i32{},
    merges: [][]u8 = &[_][]u8{},
    tok: ?*tokmod.Tokenizer = null,
    weights: vk.Buffer = .{},
    signs_buf: vk.Buffer = .{},
    have_gpu: bool = false,

    fn find(self: *const Model, name: []const u8) ?*const TensorInfo {
        const nn = @min(name.len, 127);
        @memcpy(g_last_lookup[0..nn], name[0..nn]);
        g_last_len = nn;
        for (self.tensors) |*t| {
            const tn = if (t.name.len > 0 and t.name[t.name.len - 1] == 0) t.name[0 .. t.name.len - 1] else t.name;
            if (std.mem.eql(u8, tn, name)) return t;
        }
        return null;
    }
};

fn tensorSize(ttype: u32, nelem: u64) !u64 {
    return switch (ttype) {
        PTQ1_0 => blk: {
            if (nelem % 128 != 0) return error.BadQuant;
            break :blk nelem / 128 * 28;
        },
        0 => nelem * 4,
        1 => nelem * 2,
        else => error.UnsupportedType,
    };
}

fn parseValue(rd: *Rd, m: *Model, key: []const u8, vtype: u32, alignment: *u64) !void {
    switch (vtype) {
        0, 1 => _ = try rd.u8v(),
        2, 3 => try rd.skip(2),
        4, 5, 6 => {
            const v = try rd.u32v();
            if (std.mem.endsWith(u8, key, ".context_length")) m.n_ctx_train = @intCast(v);
            if (std.mem.eql(u8, key, "tokenizer.ggml.bos_token_id")) m.bos_id = @intCast(v);
            if (std.mem.eql(u8, key, "tokenizer.ggml.eos_token_id")) m.eos_id = @intCast(v);
            if (std.mem.eql(u8, key, "general.alignment")) alignment.* = v;
        },
        7 => {
            const v = try rd.u8v();
            if (std.mem.eql(u8, key, "tokenizer.ggml.add_bos_token")) m.add_bos = v != 0;
        },
        8 => {
            const s = try rd.str();
            alloc.free(s);
        },
        9 => {
            const atype = try rd.u32v();
            const count = try rd.u64v();
            if (count > 4_000_000) return error.BadArray;
            if (atype == 8) {
                var arr = alloc.alloc([]u8, @intCast(count)) catch return error.NoMem;
                var k: usize = 0;
                while (k < arr.len) : (k += 1) arr[k] = try rd.str();
                if (std.mem.eql(u8, key, "tokenizer.ggml.tokens")) {
                    m.tokens = arr;
                    m.n_vocab = @intCast(count);
                } else if (std.mem.eql(u8, key, "tokenizer.ggml.merges")) {
                    m.merges = arr;
                } else {
                    for (arr) |s2| alloc.free(s2);
                    alloc.free(arr);
                }
            } else if (atype == 5) {
                var arr = alloc.alloc(i32, @intCast(count)) catch return error.NoMem;
                var k: usize = 0;
                while (k < arr.len) : (k += 1) arr[k] = @bitCast(try rd.u32v());
                if (std.mem.eql(u8, key, "prism.hadamard.sign_values")) {
                    const sf = alloc.alloc(f32, arr.len) catch return error.NoMem;
                    for (arr, 0..) |v, j| sf[j] = @floatFromInt(v);
                    m.signs = sf;
                    alloc.free(arr);
                } else if (std.mem.eql(u8, key, "tokenizer.ggml.token_type")) {
                    m.tok_types = arr;
                } else alloc.free(arr);
            } else {
                var k: u64 = 0;
                while (k < count) : (k += 1) try parseValue(rd, m, "", atype, alignment);
            }
        },
        10, 11, 12 => try rd.skip(8),
        else => return error.BadValueType,
    }
}

fn parseGGUF(path: []const u8, m: *Model) !void {
    const pz = alloc.allocSentinel(u8, path.len, 0) catch return error.NoMem;
    @memcpy(pz, path);
    defer alloc.free(pz);
    const fd = open(pz.ptr, 0, 0);
    if (fd < 0) return error.OpenFailed;
    m.fd = fd;
    var rd = Rd{ .fd = fd };
    var magic: [4]u8 = undefined;
    try rd.read(&magic);
    if (std.mem.readInt(u32, &magic, .little) != 0x46554747) return error.BadMagic;
    const version = try rd.u32v();
    if (version != 3) return error.BadVersion;
    const n_tensors = try rd.u64v();
    const n_kv = try rd.u64v();
    if (n_tensors > 8192 or n_kv > 8192) return error.BadHeader;
    var alignment: u64 = 32;
    var i: u64 = 0;
    while (i < n_kv) : (i += 1) {
        const key = try rd.str();
        defer alloc.free(key);
        const vtype = try rd.u32v();
        try parseValue(&rd, m, key, vtype, &alignment);
    }
    m.tensors = alloc.alloc(TensorInfo, @intCast(n_tensors)) catch return error.NoMem;
    var ti: usize = 0;
    var maxend: u64 = 0;
    while (ti < n_tensors) : (ti += 1) {
        const raw_nm = try rd.str();
        const nm = alloc.alloc(u8, raw_nm.len + 1) catch return error.NoMem;
        @memcpy(nm[0..raw_nm.len], raw_nm);
        nm[raw_nm.len] = 0;
        alloc.free(raw_nm);
        const nd = try rd.u32v();
        if (nd > 4) return error.BadTensor;
        var dims = [4]u64{ 1, 1, 1, 1 };
        var d: u32 = 0;
        while (d < nd) : (d += 1) dims[d] = try rd.u64v();
        const tt = try rd.u32v();
        const off = try rd.u64v();
        var nelem: u64 = 1;
        for (dims) |dv| nelem *= dv;
        const sz = tensorSize(tt, nelem) catch return error.UnsupportedType;
        m.tensors[ti] = .{ .name = nm, .ttype = tt, .ndims = nd, .dims = dims, .offset = off, .size = sz, .nelem = nelem };
        if (off + sz > maxend) maxend = off + sz;
    }
    m.data_size = maxend;
    const consumed = rd.off - @as(u64, rd.blen) + @as(u64, rd.bpos);
    m.data_start = (consumed + alignment - 1) / alignment * alignment;
    if (m.n_vocab == 0) return error.NoVocab;
}

fn dequantPTQ1(raw: []const u8, out: []f32) void {
    const nb = out.len / 128;
    var b: usize = 0;
    while (b < nb) : (b += 1) {
        const blk = raw[b * 28 .. b * 28 + 28];
        const scale: f16 = @bitCast(std.mem.readInt(u16, blk[26..28], .little));
        const s: f32 = @floatCast(scale);
        var idx: usize = 0;
        while (idx < 128) : (idx += 1) {
            const tid = idx & 127;
            var byte_idx: usize = undefined;
            var p: u32 = undefined;
            var is4: bool = false;
            if (tid < 80) {
                byte_idx = tid % 16;
                p = @intCast(tid / 16);
            } else if (tid < 120) {
                const j = tid - 80;
                byte_idx = 16 + j % 8;
                p = @intCast(j / 8);
            } else {
                const j = tid - 120;
                byte_idx = 24 + j % 2;
                p = @intCast(j / 2);
                is4 = true;
            }
            var v: u32 = undefined;
            if (is4) {
                v = (@as(u32, blk[byte_idx]) * 81) >> 8;
                const div = pow3(3 - p);
                out[b * 128 + idx] = s * (@as(f32, @floatFromInt((v / div) % 3)) - 1.0);
            } else {
                v = (@as(u32, blk[byte_idx]) * 243) >> 8;
                const div = pow3(4 - p);
                out[b * 128 + idx] = s * (@as(f32, @floatFromInt((v / div) % 3)) - 1.0);
            }
        }
    }
}

fn pow3(k: u32) u32 {
    return switch (k) {
        0 => 1,
        1 => 3,
        2 => 9,
        3 => 27,
        else => 81,
    };
}

// ---------------- ABI ----------------
const ModelOpaque = opaque {};
const ContextOpaque = opaque {};

export fn bonsai_abi_version() callconv(.c) u32 {
    return 1;
}

export fn bonsai_last_error() callconv(.c) ?[*:0]const u8 {
    if (!err_set) return null;
    err_buf[err_buf.len - 1] = 0;
    var n: usize = 0;
    while (n < err_buf.len and err_buf[n] != 0) n += 1;
    if (n == err_buf.len) err_buf[err_buf.len - 1] = 0;
    return @ptrCast(&err_buf);
}

export fn bonsai_model_open(gguf_path: ?[*:0]const u8, out_model: ?*?*ModelOpaque) callconv(.c) i32 {
    if (out_model) |p| p.* = null;
    if (gguf_path == null or out_model == null) {
        setErr("invalid argument", .{});
        return -1;
    }
    const path = std.mem.span(gguf_path.?);
    const m = alloc.create(Model) catch {
        setErr("out of memory", .{});
        return -2;
    };
    m.* = .{};
    parseGGUF(path, m) catch |e| {
        setErr("failed to parse GGUF: {s}", .{@errorName(e)});
        if (m.fd >= 0) _ = close(m.fd);
        alloc.destroy(m);
        return -3;
    };
    if (m.tokens.len > 0 and m.merges.len > 0) {
        const t = alloc.create(tokmod.Tokenizer) catch null;
        if (t) |tp| {
            if (tokmod.Tokenizer.build(alloc, m.tokens, m.merges, m.tok_types)) |tb| {
                tp.* = tb;
                m.tok = tp;
            } else |_| {
                setErr("tokenizer build failed", .{});
                alloc.destroy(tp);
            }
        }
    }
    gpuSetupModel(m) catch |e| {
        setErr("GPU setup failed: {s}", .{@errorName(e)});
    };
    out_model.?.* = @ptrCast(m);
    return 0;
}

export fn bonsai_model_close(model: ?*ModelOpaque) callconv(.c) void {
    if (model == null) return;
    const m: *Model = @ptrCast(@alignCast(model.?));
    if (m.tok) |t| {
        t.deinit();
        alloc.destroy(t);
    }
    if (m.fd >= 0) _ = close(m.fd);
    for (m.tensors) |t| alloc.free(t.name);
    if (m.tensors.len > 0) alloc.free(m.tensors);
    for (m.tokens) |s| alloc.free(s);
    if (m.tokens.len > 0) alloc.free(m.tokens);
    for (m.merges) |s| alloc.free(s);
    if (m.merges.len > 0) alloc.free(m.merges);
    if (m.tok_types.len > 0) alloc.free(m.tok_types);
    if (m.signs.len > 0) alloc.free(m.signs);
    alloc.destroy(m);
}

export fn bonsai_n_vocab(model: ?*const ModelOpaque) callconv(.c) i32 {
    if (model == null) return -1;
    const m: *const Model = @ptrCast(@alignCast(model.?));
    return m.n_vocab;
}

export fn bonsai_n_ctx_train(model: ?*const ModelOpaque) callconv(.c) i32 {
    if (model == null) return -1;
    const m: *const Model = @ptrCast(@alignCast(model.?));
    return m.n_ctx_train;
}

export fn bonsai_tensor_count(model: ?*const ModelOpaque) callconv(.c) i32 {
    if (model == null) return -1;
    const m: *const Model = @ptrCast(@alignCast(model.?));
    return @intCast(m.tensors.len);
}

export fn bonsai_tensor_info(
    model: ?*const ModelOpaque,
    index: i32,
    name: ?*?[*:0]const u8,
    ggml_type: ?*i32,
    rank: ?*i32,
    ne: ?[*]i64,
) callconv(.c) i32 {
    if (model == null or index < 0) return -1;
    const m: *const Model = @ptrCast(@alignCast(model.?));
    if (index >= m.tensors.len) return -1;
    const t = m.tensors[@intCast(index)];
    if (name) |p| p.* = @ptrCast(t.name.ptr);
    if (ggml_type) |p| p.* = @intCast(t.ttype);
    if (rank) |p| p.* = @intCast(t.ndims);
    if (ne) |p| {
        var i: usize = 0;
        while (i < 4) : (i += 1) p[i] = @intCast(t.dims[i]);
    }
    return 0;
}

export fn bonsai_tensor_dequant(
    model: ?*const ModelOpaque,
    name: ?[*:0]const u8,
    out: ?[*]f32,
    capacity: i64,
) callconv(.c) i64 {
    if (model == null or name == null) return -1;
    const m: *const Model = @ptrCast(@alignCast(model.?));
    const nm = std.mem.span(name.?);
    const t = m.find(nm) orelse {
        setErr("tensor not found: {s}", .{nm});
        return -1;
    };
    if (t.nelem > @as(u64, @intCast(1 << 40))) return -1;
    if (capacity < @as(i64, @intCast(t.nelem))) return -1;
    if (out == null) return -1;
    const dst = out.?[0..@intCast(t.nelem)];
    if (t.ttype == 0) {
        if (m.fd < 0) return -1;
        const tmp = alloc.alloc(u8, @intCast(t.size)) catch return -2;
        defer alloc.free(tmp);
        const nread = pread(m.fd, tmp.ptr, tmp.len, @intCast(m.data_start + t.offset));
        if (nread != tmp.len) return -3;
        const src: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, tmp));
        @memcpy(dst, src);
        return @intCast(t.nelem);
    }
    if (t.ttype != PTQ1_0) {
        setErr("unsupported tensor type {d}", .{t.ttype});
        return -1;
    }
    if (m.fd < 0) return -1;
    const raw = alloc.alloc(u8, @intCast(t.size)) catch return -2;
    defer alloc.free(raw);
    const nread = pread(m.fd, raw.ptr, raw.len, @intCast(m.data_start + t.offset));
    if (nread != raw.len) return -3;
    dequantPTQ1(raw, dst);
    return @intCast(t.nelem);
}

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
var g_stop_layer: i32 = -1;
var g_dbg_model: ?*Model = null;
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
    var p = std.mem.zeroes([16]u32);
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
    rmsDispatchS(c, src, dst, n, woff, has_w, rows, 0);
}

fn rmsDispatchS(c: *Context, src: u64, dst: u64, n: u32, woff: u32, has_w: u32, rows: u32, stride: u32) void {
    const p = pcBuf(&.{ n, fbits(1e-6), woff, has_w, stride });
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
        if (g_stop_layer == -2) break;
        const full = (il % FULL_ATTN_INTERVAL) == (FULL_ATTN_INTERVAL - 1);
        rmsDispatch(c, SCR_X, SCR_H, N_EMBD, try f32OffBlk(m, "blk.{d}.attn_norm.weight", il), 1, 1);
        if (g_stop_layer == -3 and il == 0) break;
        tfDispatch(c, SCR_H, SCR_HF, N_EMBD, SIGNS_2560, 0, 1);
        if (g_stop_layer == -4 and il == 0) break;
        if (!full) {
            const tqkv = try findBlk(m, "blk.{d}.attn_qkv.weight", il);
            matvecPtq(c, SCR_HF, @intCast(tqkv.offset / 4), N_EMBD, QKV_DIM, SCR_QKV);
            if (g_stop_layer == -5 and il == 0) break;
            const tgate = try findBlk(m, "blk.{d}.attn_gate.weight", il);
            matvecPtq(c, SCR_HF, @intCast(tgate.offset / 4), N_EMBD, INNER, SCR_Z);
            const al_off = try f32OffBlk(m, "blk.{d}.ssm_alpha.weight", il);
            const beta_off = try f32OffBlk(m, "blk.{d}.ssm_beta.weight", il);
            matvecF32(c, SCR_H, al_off, N_EMBD, 32, SCR_GB);
            matvecF32(c, SCR_H, beta_off, N_EMBD, 32, SCR_GB + 128);
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
            if (g_stop_layer == -9 and il == 0) break;
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
            if (g_stop_layer == -10 and il == 0) break;
            tfDispatch(c, SCR_ATTN, SCR_ATTNF, INNER, SIGNS_4096, 0, 1);
            const tout = try findBlk(m, "blk.{d}.ssm_out.weight", il);
            matvecPtq(c, SCR_ATTNF, @intCast(tout.offset / 4), INNER, N_EMBD, SCR_Y);
            if (g_stop_layer == -12) break;
            ewDispatch(c, SCR_X, SCR_Y, SCR_X, N_EMBD, 0);
        } else {
            const fl = il / FULL_ATTN_INTERVAL;
            const tq = try findBlk(m, "blk.{d}.attn_q.weight", il);
            matvecPtq(c, SCR_HF, @intCast(tq.offset / 4), N_EMBD, QKV_DIM, SCR_QKV);
            const tk = try findBlk(m, "blk.{d}.attn_k.weight", il);
            matvecPtq(c, SCR_HF, @intCast(tk.offset / 4), N_EMBD, 1024, SCR_KRAW);
            const tv = try findBlk(m, "blk.{d}.attn_v.weight", il);
            matvecPtq(c, SCR_HF, @intCast(tv.offset / 4), N_EMBD, 1024, SCR_VRAW);
            rmsDispatchS(c, SCR_QKV, SCR_QKV, 256, try f32OffBlk(m, "blk.{d}.attn_q_norm.weight", il), 1, 16, 512);
            rmsDispatch(c, SCR_KRAW, SCR_KRAW, 256, try f32OffBlk(m, "blk.{d}.attn_k_norm.weight", il), 1, 4);
            {
                const p = pcBuf(&.{ @intCast(c.pos), 16, 512, 0 });
                dispatch(g_pipes.rope, &.{ .{ .b = c.scratch, .off = SCR_QKV }, .{ .b = c.scratch, .off = SCR_QROT } }, &p, 16, 1, 1);
            }
            {
                const p = pcBuf(&.{ @intCast(c.pos), 4, 256, 0 });
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
            ewDispatch(c, SCR_ATTN, SCR_QKV, SCR_ATTN, INNER, 2);
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
        if (g_stop_layer >= 0 and @as(i32, @intCast(il)) >= g_stop_layer) break;
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
    g_dbg_model = m;
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
            setErr("eval failed: {s} (last tensor {s})", .{ @errorName(e), g_last_lookup[0..g_last_len] });
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
    if (ctx == null or out_ids == null or n_predict < 0 or n_prompt < 0) {
        setErr("invalid argument", .{});
        return -1;
    }
    const c: *Context = @ptrCast(@alignCast(ctx.?));
    const m = c.model;
    const nv: usize = @intCast(m.n_vocab);
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
        for (buf[0..nv], 0..) |v, idx| {
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
    if (ids == null) {
        // caller queries required capacity: worst case one token per byte
        return @intCast(text.len + 8);
    }
    if (m.tok) |t| {
        // upper bound on output ids: one per mapped symbol, at most ~3 per byte
        const cap: usize = @intCast(capacity);
        var tmp_buf: [4096]i32 = undefined;
        var total: usize = 0;
        var buf_view: []i32 = undefined;
        var owned: ?[]i32 = null;
        defer if (owned) |o| alloc.free(o);
        if (text.len + 8 <= tmp_buf.len) {
            buf_view = tmp_buf[0..];
        } else {
            const need = text.len * 4 + 16;
            const mem = alloc.alloc(i32, need) catch return -2;
            owned = mem;
            buf_view = mem;
        }
        const n = t.tokenize(text, buf_view) catch |e| {
            setErr("tokenize failed: {s}", .{@errorName(e)});
            return -3;
        };
        total = n;
        if (total > cap) {
            setErr("capacity {d} < required {d}", .{ capacity, total });
            return -1;
        }
        @memcpy(ids.?[0..total], buf_view[0..total]);
        return @intCast(total);
    }
    setErr("tokenizer not initialized", .{});
    return -3;
}


export fn bonsai_dbg_stop(layer: i32) callconv(.c) void {
    g_stop_layer = layer;
}

export fn bonsai_dbg_x(ctx: ?*ContextOpaque, off: u32, out: ?[*]f32, n: i32) callconv(.c) i32 {
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
