const std = @import("std");

const libc = struct {
    extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
};
extern "c" fn close(fd: c_int) c_int;
extern "c" fn lseek(fd: c_int, offset: i64, whence: c_int) i64;
extern "c" fn mmap(address: ?*anyopaque, length: usize, prot: c_int, flags: c_int, fd: c_int, offset: i64) ?*anyopaque;
extern "c" fn munmap(address: *const anyopaque, length: usize) c_int;

pub const Value = union(enum) {
    unsigned: u64,
    signed: i64,
    real: f64,
    boolean: bool,
    string: []const u8,
    array: []const Value,

    pub fn asUInt(self: Value) ?u64 {
        return switch (self) {
            .unsigned => |v| v,
            .signed => |v| if (v >= 0) @intCast(v) else null,
            else => null,
        };
    }
    pub fn asInt(self: Value) ?i64 {
        return switch (self) {
            .signed => |v| v,
            .unsigned => |v| if (v <= std.math.maxInt(i64)) @intCast(v) else null,
            else => null,
        };
    }
};

pub const Metadata = struct { key: []const u8, value: Value };

pub const Tensor = struct {
    name: [:0]const u8,
    typ: u32,
    n_dims: u32,
    dims: [4]u64,
    offset: u64,
    data: []const u8,

    pub fn elementCount(self: *const Tensor) usize {
        var n: usize = 1;
        for (self.dims) |dim| n *= @intCast(dim);
        return n;
    }

    /// CPU decoding is exclusively for ABI tensor introspection, never inference.
    pub fn dequant(self: *const Tensor, out: []f32) !void {
        const n = self.elementCount();
        if (out.len < n) return error.Capacity;
        switch (self.typ) {
            0 => @memcpy(std.mem.sliceAsBytes(out[0..n]), self.data),
            1 => for (out[0..n], 0..) |*v, i| {
                v.* = half(self.data[i * 2 ..][0..2]);
            },
            143 => {
                var block: usize = 0;
                while (block < n / 128) : (block += 1) {
                    const src = self.data[block * 28 ..][0..28];
                    const d = half(src[26..28]);
                    var i: usize = 0;
                    while (i < 128) : (i += 1) out[block * 128 + i] = ptqValue(src, i, d);
                }
            },
            else => return error.UnsupportedTensorType,
        }
    }
};

fn dupeZ(allocator: std.mem.Allocator, bytes: []const u8) ![:0]u8 {
    const result = try allocator.alloc(u8, bytes.len + 1);
    @memcpy(result[0..bytes.len], bytes);
    result[bytes.len] = 0;
    return result[0..bytes.len :0];
}

fn half(bytes: *const [2]u8) f32 {
    return @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes, .little))));
}

/// PTQ1_0: 128 trits, 26 packed bytes, then one little-endian fp16 scale.
/// The first 80 values have stride16, next40 stride8, final8 stride2.
/// Recovered directly from GGUF and the safetensors development weights.
pub fn ptqValue(src: []const u8, index: usize, scale: f32) f32 {
    const j: usize = if (index < 80) index % 16 else if (index < 120) 16 + (index - 80) % 8 else 24 + (index - 120) % 2;
    const k: usize = if (index < 80) index / 16 else if (index < 120) (index - 80) / 8 else (index - 120) / 2;
    const powers = [_]u32{ 1, 3, 9, 27, 81 };
    const trit: i32 = @as(i32, @intCast((((@as(u32, src[j]) * powers[k]) & 255) * 3) >> 8)) - 1;
    return scale * @as(f32, @floatFromInt(trit));
}

pub const Model = struct {
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    mapped: []const u8,
    weights: []const u8,
    data_offset: usize,
    tensors: []Tensor,
    metadata: []Metadata,

    pub fn open(allocator: std.mem.Allocator, path: []const u8) !Model {
        if (std.mem.indexOfScalar(u8, path, 0) != null) return error.InvalidPath;
        const path_z = try dupeZ(allocator, path);
        defer allocator.free(path_z);
        const fd = libc.open(path_z.ptr, @as(c_int, 0));
        if (fd < 0) return error.OpenFailed;
        defer _ = close(fd);
        const length = lseek(fd, 0, 2);
        if (length < 24) return error.InvalidGGUF;
        const size: usize = @intCast(length);
        const address = mmap(null, size, 1, 2, fd, 0) orelse return error.MapFailed;
        if (@intFromPtr(address) == std.math.maxInt(usize)) return error.MapFailed;
        const mapped = @as([*]const u8, @ptrCast(address))[0..size];
        errdefer _ = munmap(address, size);
        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();
        const a = arena.allocator();
        var p = Parser{ .bytes = mapped, .allocator = a };
        if (try p.integer(u32) != 0x46554747) return error.InvalidGGUF;
        const version = try p.integer(u32);
        if (version < 2 or version > 3) return error.UnsupportedGGUFVersion;
        const tensor_count = try p.count();
        const metadata_count = try p.count();
        if (tensor_count > size / 24 or metadata_count > size / 12) return error.InvalidGGUF;
        const metadata = try a.alloc(Metadata, metadata_count);
        for (metadata) |*entry| {
            entry.key = try p.string();
            const typ = try p.integer(u32);
            entry.value = try p.value(typ, 0);
        }
        const tensors = try a.alloc(Tensor, tensor_count);
        for (tensors) |*tensor| {
            const name = try p.string();
            if (std.mem.indexOfScalar(u8, name, 0) != null) return error.InvalidTensorName;
            tensor.name = try dupeZ(a, name);
            tensor.n_dims = try p.integer(u32);
            if (tensor.n_dims == 0 or tensor.n_dims > 4) return error.InvalidTensorShape;
            tensor.dims = .{ 1, 1, 1, 1 };
            var n: usize = 1;
            for (tensor.dims[0..tensor.n_dims]) |*dim| {
                dim.* = try p.integer(u64);
                if (dim.* == 0 or dim.* > std.math.maxInt(usize) / n) return error.InvalidTensorShape;
                n *= @intCast(dim.*);
            }
            tensor.typ = try p.integer(u32);
            if (tensor.typ == 143 and tensor.dims[0] % 128 != 0) return error.InvalidTensorShape;
            tensor.offset = try p.integer(u64);
            const byte_count = try tensorBytes(tensor.typ, n);
            if (byte_count > size) return error.TruncatedTensor;
            tensor.data = mapped[0..byte_count]; // replaced after data section is known
        }
        var alignment: usize = 32;
        for (metadata) |entry| {
            if (std.mem.eql(u8, entry.key, "general.alignment")) {
                const al = entry.value.asUInt() orelse return error.InvalidAlignment;
                if (al == 0 or al > 1048576 or al & (al - 1) != 0) return error.InvalidAlignment;
                alignment = @intCast(al);
            }
        }
        if (p.pos > size -| (alignment - 1)) return error.InvalidGGUF;
        const data_offset = (p.pos + alignment - 1) & ~(alignment - 1);
        if (data_offset > size) return error.InvalidGGUF;
        const weights = mapped[data_offset..];
        for (tensors) |*tensor| {
            if (tensor.offset & (alignment - 1) != 0) return error.InvalidAlignment;
            if (tensor.offset > weights.len or tensor.data.len > weights.len - @as(usize, @intCast(tensor.offset))) return error.TruncatedTensor;
            const off: usize = @intCast(tensor.offset);
            tensor.data = weights[off..][0..tensor.data.len];
        }
        return .{ .allocator = allocator, .arena = arena, .mapped = mapped, .weights = weights, .data_offset = data_offset, .tensors = tensors, .metadata = metadata };
    }

    pub fn deinit(self: *Model) void {
        _ = munmap(self.mapped.ptr, self.mapped.len);
        self.arena.deinit();
        self.* = undefined;
    }
    pub fn findTensor(self: *const Model, name: []const u8) ?*const Tensor {
        for (self.tensors) |*tensor| if (std.mem.eql(u8, tensor.name, name)) return tensor;
        return null;
    }
    pub fn get(self: *const Model, key: []const u8) ?Value {
        for (self.metadata) |entry| if (std.mem.eql(u8, entry.key, key)) return entry.value;
        return null;
    }
    pub fn getUInt(self: *const Model, key: []const u8) ?u64 {
        return (self.get(key) orelse return null).asUInt();
    }
    pub fn getInt(self: *const Model, key: []const u8) ?i64 {
        return (self.get(key) orelse return null).asInt();
    }
    pub fn getFloat(self: *const Model, key: []const u8) ?f64 {
        return switch (self.get(key) orelse return null) {
            .real => |v| v,
            else => null,
        };
    }
    pub fn getBool(self: *const Model, key: []const u8) ?bool {
        return switch (self.get(key) orelse return null) {
            .boolean => |v| v,
            else => null,
        };
    }
    pub fn getString(self: *const Model, key: []const u8) ?[]const u8 {
        return switch (self.get(key) orelse return null) {
            .string => |v| v,
            else => null,
        };
    }
    pub fn getArray(self: *const Model, key: []const u8) ?[]const Value {
        return switch (self.get(key) orelse return null) {
            .array => |v| v,
            else => null,
        };
    }
};

fn tensorBytes(typ: u32, count: usize) !usize {
    return switch (typ) {
        0 => std.math.mul(usize, count, 4) catch return error.InvalidTensorShape,
        1 => std.math.mul(usize, count, 2) catch return error.InvalidTensorShape,
        143 => if (count % 128 == 0) std.math.mul(usize, count / 128, 28) catch return error.InvalidTensorShape else error.InvalidTensorShape,
        else => error.UnsupportedTensorType,
    };
}

const Parser = struct {
    bytes: []const u8,
    pos: usize = 0,
    allocator: std.mem.Allocator,

    fn take(self: *Parser, n: usize) ![]const u8 {
        if (n > self.bytes.len - self.pos) return error.TruncatedGGUF;
        const out = self.bytes[self.pos..][0..n];
        self.pos += n;
        return out;
    }
    fn integer(self: *Parser, comptime T: type) !T {
        const bytes = try self.take(@sizeOf(T));
        return std.mem.readInt(T, bytes[0..@sizeOf(T)], .little);
    }
    fn count(self: *Parser) !usize {
        const n = try self.integer(u64);
        if (n > self.bytes.len) return error.InvalidGGUF;
        return @intCast(n);
    }
    fn string(self: *Parser) ![]const u8 {
        return self.take(try self.count());
    }
    fn value(self: *Parser, typ: u32, depth: usize) anyerror!Value {
        if (depth > 16) return error.InvalidGGUF;
        return switch (typ) {
            0 => .{ .unsigned = try self.integer(u8) },
            1 => .{ .signed = try self.integer(i8) },
            2 => .{ .unsigned = try self.integer(u16) },
            3 => .{ .signed = try self.integer(i16) },
            4 => .{ .unsigned = try self.integer(u32) },
            5 => .{ .signed = try self.integer(i32) },
            6 => .{ .real = @as(f32, @bitCast(try self.integer(u32))) },
            7 => blk: {
                const v = try self.integer(u8);
                if (v > 1) return error.InvalidGGUF;
                break :blk .{ .boolean = v != 0 };
            },
            8 => .{ .string = try self.string() },
            9 => blk: {
                const element_type = try self.integer(u32);
                const n = try self.count();
                if (n > self.bytes.len - self.pos) return error.TruncatedGGUF;
                const values = try self.allocator.alloc(Value, n);
                for (values) |*v| v.* = try self.value(element_type, depth + 1);
                break :blk .{ .array = values };
            },
            10 => .{ .unsigned = try self.integer(u64) },
            11 => .{ .signed = try self.integer(i64) },
            12 => .{ .real = @as(f64, @bitCast(try self.integer(u64))) },
            else => error.InvalidMetadataType,
        };
    }
};

test "PTQ packed layout and signed zero" {
    var src: [28]u8 = @splat(0);
    var output: [128]f32 = undefined;
    src[26] = 0;
    src[27] = 0x3c;
    const tensor = Tensor{ .name = "test", .typ = 143, .n_dims = 1, .dims = .{ 128, 1, 1, 1 }, .offset = 0, .data = &src };
    try tensor.dequant(&output);
    for (output) |v| try std.testing.expectEqual(@as(f32, -1), v);
    @memset(src[0..26], 128);
    try tensor.dequant(&output);
    try std.testing.expectEqual(@as(f32, 0), output[0]);
    try std.testing.expectEqual(@as(f32, 0), output[16]);
}
