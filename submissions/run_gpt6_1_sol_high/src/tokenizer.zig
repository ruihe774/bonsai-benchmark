const std = @import("std");
const model_mod = @import("model.zig");
pub const Model = model_mod.Model;

const Special = struct { text: []const u8, id: u32 };
const Merge = struct { rank: u32, id: u32 };
const none = std.math.maxInt(usize);

pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    tokens: []const model_mod.Value,
    token_types: []const model_mod.Value,
    token_map: std.StringHashMap(u32),
    merges: std.AutoHashMap(u64, Merge),
    special: []Special = &.{},
    byte_ids: [256]u32 = undefined,
    bos_id: ?u32 = null,
    add_bos: bool = false,

    pub fn init(allocator: std.mem.Allocator, model: *const model_mod.Model) !Tokenizer {
        if (!std.mem.eql(u8, model.getString("tokenizer.ggml.model") orelse return error.MissingTokenizer, "gpt2")) return error.UnsupportedTokenizer;
        if (!std.mem.eql(u8, model.getString("tokenizer.ggml.pre") orelse return error.MissingTokenizer, "qwen35")) return error.UnsupportedTokenizer;
        const tokens = model.getArray("tokenizer.ggml.tokens") orelse return error.MissingTokenizer;
        const types = model.getArray("tokenizer.ggml.token_type") orelse return error.MissingTokenizer;
        const rules = model.getArray("tokenizer.ggml.merges") orelse return error.MissingTokenizer;
        if (tokens.len != types.len or tokens.len > std.math.maxInt(i32)) return error.InvalidTokenizer;
        const bos = model.getUInt("tokenizer.ggml.bos_token_id");
        if (bos) |id| if (id >= tokens.len) return error.InvalidTokenizer;
        var self = Tokenizer{
            .allocator = allocator,
            .tokens = tokens,
            .token_types = types,
            .token_map = std.StringHashMap(u32).init(allocator),
            .merges = std.AutoHashMap(u64, Merge).init(allocator),
            .add_bos = model.getBool("tokenizer.ggml.add_bos_token") orelse false,
            .bos_id = if (bos) |id| @intCast(id) else null,
        };
        errdefer self.deinit();
        try self.token_map.ensureTotalCapacity(@intCast(tokens.len));
        var specials: std.ArrayList(Special) = .empty;
        defer specials.deinit(allocator);
        for (tokens, types, 0..) |token, typ, i| {
            const text = switch (token) {
                .string => |s| s,
                else => return error.InvalidTokenizer,
            };
            try self.token_map.put(text, @intCast(i));
            const kind = typ.asUInt() orelse return error.InvalidTokenizer;
            if ((kind == 3 or kind == 4) and text.len > 0) try specials.append(allocator, .{ .text = text, .id = @intCast(i) });
        }
        self.special = try specials.toOwnedSlice(allocator);
        var cp_next: u21 = 256;
        for (0..256) |byte| {
            const cp: u21 = if (directByte(@intCast(byte))) @intCast(byte) else blk: {
                const value = cp_next;
                cp_next += 1;
                break :blk value;
            };
            var encoded: [4]u8 = undefined;
            const len = encodeUtf8(cp, &encoded);
            self.byte_ids[byte] = self.token_map.get(encoded[0..len]) orelse return error.MissingByteToken;
        }
        try self.merges.ensureTotalCapacity(@intCast(rules.len));
        var scratch: std.ArrayList(u8) = .empty;
        defer scratch.deinit(allocator);
        for (rules, 0..) |rule, rank| {
            const text = switch (rule) {
                .string => |s| s,
                else => return error.InvalidTokenizer,
            };
            const space = std.mem.indexOfScalar(u8, text, ' ') orelse return error.InvalidMerge;
            const left = self.token_map.get(text[0..space]) orelse return error.InvalidMerge;
            const right = self.token_map.get(text[space + 1 ..]) orelse return error.InvalidMerge;
            try scratch.resize(allocator, text.len - 1);
            @memcpy(scratch.items[0..space], text[0..space]);
            @memcpy(scratch.items[space..], text[space + 1 ..]);
            const merged = self.token_map.get(scratch.items) orelse return error.InvalidMerge;
            const result = try self.merges.getOrPut(pairKey(left, right));
            if (!result.found_existing) result.value_ptr.* = .{ .rank = @intCast(rank), .id = merged };
        }
        return self;
    }

    pub fn deinit(self: *Tokenizer) void {
        self.token_map.deinit();
        self.merges.deinit();
        self.allocator.free(self.special);
        self.* = undefined;
    }

    /// The returned ids are owned by the caller and use this tokenizer's allocator.
    pub fn tokenize(self: *const Tokenizer, text: []const u8, add_bos: bool) ![]i32 {
        var result: std.ArrayList(i32) = .empty;
        errdefer result.deinit(self.allocator);
        if (add_bos and self.add_bos) {
            if (self.bos_id) |id| try result.append(self.allocator, @intCast(id));
        }
        var start: usize = 0;
        var position: usize = 0;
        while (position < text.len) {
            var matched: ?Special = null;
            for (self.special) |special| {
                if (std.mem.startsWith(u8, text[position..], special.text)) {
                    if (matched == null or special.text.len > matched.?.text.len) matched = special;
                }
            }
            if (matched) |special| {
                try self.ordinary(text[start..position], &result);
                try result.append(self.allocator, @intCast(special.id));
                position += special.text.len;
                start = position;
            } else position += 1;
        }
        try self.ordinary(text[start..], &result);
        return result.toOwnedSlice(self.allocator);
    }

    pub fn encode(self: *const Tokenizer, text: []const u8) ![]i32 {
        return self.tokenize(text, false);
    }

    /// Decode GPT-2's byte alphabet; special tokens retain their literal spelling.
    pub fn decode(self: *const Tokenizer, ids: []const i32) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(self.allocator);
        for (ids) |id| {
            if (id < 0 or @as(usize, @intCast(id)) >= self.tokens.len) return error.InvalidTokenId;
            const i: usize = @intCast(id);
            const token = self.tokens[i].string;
            const typ = self.token_types[i].asUInt() orelse return error.InvalidTokenizer;
            if (typ != 1 and typ != 6) {
                try out.appendSlice(self.allocator, token);
                continue;
            }
            var pos: usize = 0;
            while (pos < token.len) {
                const rune = try readRune(token, pos);
                const byte = unicodeByte(rune.cp) orelse return error.InvalidByteToken;
                try out.append(self.allocator, byte);
                pos += rune.len;
            }
        }
        return out.toOwnedSlice(self.allocator);
    }

    fn ordinary(self: *const Tokenizer, text: []const u8, result: *std.ArrayList(i32)) !void {
        if (text.len == 0) return;
        var runes: std.ArrayList(Rune) = .empty;
        defer runes.deinit(self.allocator);
        var pos: usize = 0;
        while (pos < text.len) {
            var rune = try readRune(text, pos);
            rune.start = pos;
            try runes.append(self.allocator, rune);
            pos += rune.len;
        }
        var start: usize = 0;
        while (start < runes.items.len) {
            const end = pieceEnd(runes.items, start);
            const a = runes.items[start].start;
            const b = if (end == runes.items.len) text.len else runes.items[end].start;
            try self.bpe(text[a..b], result);
            start = end;
        }
    }

    fn bpe(self: *const Tokenizer, piece: []const u8, result: *std.ArrayList(i32)) !void {
        if (piece.len == 0) return;
        const nodes = try self.allocator.alloc(Node, piece.len);
        defer self.allocator.free(nodes);
        for (piece, 0..) |byte, i| nodes[i] = .{ .id = self.byte_ids[byte], .prev = if (i == 0) none else i - 1, .next = if (i + 1 == piece.len) none else i + 1 };
        var heap: std.ArrayList(Candidate) = .empty;
        defer heap.deinit(self.allocator);
        for (0..piece.len - 1) |i| try self.queuePair(nodes, i, &heap);
        while (heap.items.len > 0) {
            const candidate = heapPop(&heap);
            const left = &nodes[candidate.left];
            const right = &nodes[candidate.right];
            if (!left.active or !right.active or left.next != candidate.right or left.id != candidate.left_id or right.id != candidate.right_id) continue;
            left.id = candidate.merged_id;
            left.next = right.next;
            right.active = false;
            if (left.next != none) nodes[left.next].prev = candidate.left;
            if (left.prev != none) try self.queuePair(nodes, left.prev, &heap);
            try self.queuePair(nodes, candidate.left, &heap);
        }
        var i: usize = 0;
        while (i != none) {
            try result.append(self.allocator, @intCast(nodes[i].id));
            i = nodes[i].next;
        }
    }

    fn queuePair(self: *const Tokenizer, nodes: []const Node, left: usize, heap: *std.ArrayList(Candidate)) !void {
        const right = nodes[left].next;
        if (right == none) return;
        const merge = self.merges.get(pairKey(nodes[left].id, nodes[right].id)) orelse return;
        try heapPush(self.allocator, heap, .{ .rank = merge.rank, .left = left, .right = right, .left_id = nodes[left].id, .right_id = nodes[right].id, .merged_id = merge.id });
    }
};

fn pairKey(left: u32, right: u32) u64 {
    return (@as(u64, left) << 32) | right;
}
const Node = struct { id: u32, prev: usize, next: usize, active: bool = true };
const Candidate = struct {
    rank: u32,
    left: usize,
    right: usize,
    left_id: u32,
    right_id: u32,
    merged_id: u32,
};
fn earlier(a: Candidate, b: Candidate) bool {
    return a.rank < b.rank or (a.rank == b.rank and a.left < b.left);
}
fn heapPush(a: std.mem.Allocator, heap: *std.ArrayList(Candidate), item: Candidate) !void {
    try heap.append(a, item);
    var i = heap.items.len - 1;
    while (i > 0) {
        const parent = (i - 1) / 2;
        if (!earlier(heap.items[i], heap.items[parent])) break;
        std.mem.swap(Candidate, &heap.items[i], &heap.items[parent]);
        i = parent;
    }
}
fn heapPop(heap: *std.ArrayList(Candidate)) Candidate {
    const item = heap.items[0];
    heap.items[0] = heap.items[heap.items.len - 1];
    heap.items.len -= 1;
    var i: usize = 0;
    while (i * 2 + 1 < heap.items.len) {
        var child = i * 2 + 1;
        if (child + 1 < heap.items.len and earlier(heap.items[child + 1], heap.items[child])) child += 1;
        if (!earlier(heap.items[child], heap.items[i])) break;
        std.mem.swap(Candidate, &heap.items[child], &heap.items[i]);
        i = child;
    }
    return item;
}

const letter: u8 = 1;
const mark: u8 = 2;
const number: u8 = 4;
const whitespace: u8 = 8;
const Rune = struct { cp: u21, len: usize, start: usize = 0, category: u8 };

fn readRune(text: []const u8, pos: usize) !Rune {
    const b = text[pos];
    const len: usize = if (b < 0x80) 1 else if (b >= 0xc2 and b <= 0xdf) 2 else if (b >= 0xe0 and b <= 0xef) 3 else if (b >= 0xf0 and b <= 0xf4) 4 else return error.InvalidUtf8;
    if (len > text.len - pos) return error.InvalidUtf8;
    var cp: u21 = b & @as(u8, if (len == 1) 0x7f else if (len == 2) 0x1f else if (len == 3) 0x0f else 0x07);
    for (text[pos + 1 .. pos + len]) |continuation| {
        if (continuation & 0xc0 != 0x80) return error.InvalidUtf8;
        cp = (cp << 6) | (continuation & 0x3f);
    }
    if ((len == 2 and cp < 0x80) or (len == 3 and cp < 0x800) or (len == 4 and cp < 0x10000) or cp > 0x10ffff or (cp >= 0xd800 and cp <= 0xdfff)) return error.InvalidUtf8;
    return .{ .cp = cp, .len = len, .category = classify(cp) };
}

fn encodeUtf8(cp: u21, out: *[4]u8) usize {
    if (cp < 128) {
        out[0] = @intCast(cp);
        return 1;
    }
    if (cp < 2048) {
        out[0] = 0xc0 | @as(u8, @intCast(cp >> 6));
        out[1] = 0x80 | @as(u8, @intCast(cp & 63));
        return 2;
    }
    if (cp < 65536) {
        out[0] = 0xe0 | @as(u8, @intCast(cp >> 12));
        out[1] = 0x80 | @as(u8, @intCast((cp >> 6) & 63));
        out[2] = 0x80 | @as(u8, @intCast(cp & 63));
        return 3;
    }
    out[0] = 0xf0 | @as(u8, @intCast(cp >> 18));
    out[1] = 0x80 | @as(u8, @intCast((cp >> 12) & 63));
    out[2] = 0x80 | @as(u8, @intCast((cp >> 6) & 63));
    out[3] = 0x80 | @as(u8, @intCast(cp & 63));
    return 4;
}
fn directByte(byte: u8) bool {
    return (byte >= 33 and byte <= 126) or (byte >= 161 and byte <= 172) or byte >= 174;
}
fn unicodeByte(cp: u21) ?u8 {
    if (cp <= 255 and directByte(@intCast(cp))) return @intCast(cp);
    var next: u21 = 256;
    for (0..256) |byte| {
        if (!directByte(@intCast(byte))) {
            if (next == cp) return @intCast(byte);
            next += 1;
        }
    }
    return null;
}

/// Ordered, greedy implementation of the qwen35 regex in TOKENIZER.md.
fn pieceEnd(runes: []const Rune, start: usize) usize {
    const first = runes[start];
    if (first.cp == '\'' and start + 1 < runes.len) {
        const second = asciiLower(runes[start + 1].cp);
        if (second == 's' or second == 't' or second == 'm' or second == 'd') return start + 2;
        if (start + 2 < runes.len) {
            const third = asciiLower(runes[start + 2].cp);
            if ((second == 'r' and third == 'e') or (second == 'v' and third == 'e') or (second == 'l' and third == 'l')) return start + 3;
        }
    }
    var i = start;
    if (first.category & (letter | mark) != 0) {
        while (i < runes.len and runes[i].category & (letter | mark) != 0) : (i += 1) {}
        return i;
    }
    if (!newline(first.cp) and first.category & (letter | number) == 0 and start + 1 < runes.len and runes[start + 1].category & (letter | mark) != 0) {
        i = start + 1;
        while (i < runes.len and runes[i].category & (letter | mark) != 0) : (i += 1) {}
        return i;
    }
    if (first.category & number != 0) return start + 1;
    i = start;
    if (first.cp == ' ') i += 1;
    const punctuation_start = i;
    while (i < runes.len and runes[i].category & (whitespace | letter | mark | number) == 0) : (i += 1) {}
    if (i > punctuation_start) {
        while (i < runes.len and newline(runes[i].cp)) : (i += 1) {}
        return i;
    }
    if (first.category & whitespace != 0) {
        var last_newline: usize = start;
        i = start;
        while (i < runes.len and runes[i].category & whitespace != 0) : (i += 1) {
            if (newline(runes[i].cp)) last_newline = i + 1;
        }
        if (last_newline > start) return last_newline;
        if (i == runes.len) return i;
        // \s+(?!\S) backtracks one whitespace before a non-whitespace.
        if (i > start + 1) return i - 1;
        return i;
    }
    return start + 1;
}
fn asciiLower(cp: u21) u21 {
    return if (cp >= 'A' and cp <= 'Z') cp + 32 else cp;
}
fn newline(cp: u21) bool {
    return cp == '\r' or cp == '\n';
}

fn classify(cp: u21) u8 {
    var lo: usize = 0;
    var hi: usize = unicode_ranges.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        const range = unicode_ranges[mid];
        if (cp < range.first) hi = mid else if (cp > range.last) lo = mid + 1 else return range.category;
    }
    return 0;
}
const UnicodeRange = struct { first: u21, last: u21, category: u8 };
// Unicode 16.0.0 general-category L/M/N and Unicode White_Space ranges.
const unicode_ranges = [_]UnicodeRange{
    .{ .first = 0x9, .last = 0xd, .category = 8 },
    .{ .first = 0x20, .last = 0x20, .category = 8 },
    .{ .first = 0x30, .last = 0x39, .category = 4 },
    .{ .first = 0x41, .last = 0x5a, .category = 1 },
    .{ .first = 0x61, .last = 0x7a, .category = 1 },
    .{ .first = 0x85, .last = 0x85, .category = 8 },
    .{ .first = 0xa0, .last = 0xa0, .category = 8 },
    .{ .first = 0xaa, .last = 0xaa, .category = 1 },
    .{ .first = 0xb2, .last = 0xb3, .category = 4 },
    .{ .first = 0xb5, .last = 0xb5, .category = 1 },
    .{ .first = 0xb9, .last = 0xb9, .category = 4 },
    .{ .first = 0xba, .last = 0xba, .category = 1 },
    .{ .first = 0xbc, .last = 0xbe, .category = 4 },
    .{ .first = 0xc0, .last = 0xd6, .category = 1 },
    .{ .first = 0xd8, .last = 0xf6, .category = 1 },
    .{ .first = 0xf8, .last = 0x2c1, .category = 1 },
    .{ .first = 0x2c6, .last = 0x2d1, .category = 1 },
    .{ .first = 0x2e0, .last = 0x2e4, .category = 1 },
    .{ .first = 0x2ec, .last = 0x2ec, .category = 1 },
    .{ .first = 0x2ee, .last = 0x2ee, .category = 1 },
    .{ .first = 0x300, .last = 0x36f, .category = 2 },
    .{ .first = 0x370, .last = 0x374, .category = 1 },
    .{ .first = 0x376, .last = 0x377, .category = 1 },
    .{ .first = 0x37a, .last = 0x37d, .category = 1 },
    .{ .first = 0x37f, .last = 0x37f, .category = 1 },
    .{ .first = 0x386, .last = 0x386, .category = 1 },
    .{ .first = 0x388, .last = 0x38a, .category = 1 },
    .{ .first = 0x38c, .last = 0x38c, .category = 1 },
    .{ .first = 0x38e, .last = 0x3a1, .category = 1 },
    .{ .first = 0x3a3, .last = 0x3f5, .category = 1 },
    .{ .first = 0x3f7, .last = 0x481, .category = 1 },
    .{ .first = 0x483, .last = 0x489, .category = 2 },
    .{ .first = 0x48a, .last = 0x52f, .category = 1 },
    .{ .first = 0x531, .last = 0x556, .category = 1 },
    .{ .first = 0x559, .last = 0x559, .category = 1 },
    .{ .first = 0x560, .last = 0x588, .category = 1 },
    .{ .first = 0x591, .last = 0x5bd, .category = 2 },
    .{ .first = 0x5bf, .last = 0x5bf, .category = 2 },
    .{ .first = 0x5c1, .last = 0x5c2, .category = 2 },
    .{ .first = 0x5c4, .last = 0x5c5, .category = 2 },
    .{ .first = 0x5c7, .last = 0x5c7, .category = 2 },
    .{ .first = 0x5d0, .last = 0x5ea, .category = 1 },
    .{ .first = 0x5ef, .last = 0x5f2, .category = 1 },
    .{ .first = 0x610, .last = 0x61a, .category = 2 },
    .{ .first = 0x620, .last = 0x64a, .category = 1 },
    .{ .first = 0x64b, .last = 0x65f, .category = 2 },
    .{ .first = 0x660, .last = 0x669, .category = 4 },
    .{ .first = 0x66e, .last = 0x66f, .category = 1 },
    .{ .first = 0x670, .last = 0x670, .category = 2 },
    .{ .first = 0x671, .last = 0x6d3, .category = 1 },
    .{ .first = 0x6d5, .last = 0x6d5, .category = 1 },
    .{ .first = 0x6d6, .last = 0x6dc, .category = 2 },
    .{ .first = 0x6df, .last = 0x6e4, .category = 2 },
    .{ .first = 0x6e5, .last = 0x6e6, .category = 1 },
    .{ .first = 0x6e7, .last = 0x6e8, .category = 2 },
    .{ .first = 0x6ea, .last = 0x6ed, .category = 2 },
    .{ .first = 0x6ee, .last = 0x6ef, .category = 1 },
    .{ .first = 0x6f0, .last = 0x6f9, .category = 4 },
    .{ .first = 0x6fa, .last = 0x6fc, .category = 1 },
    .{ .first = 0x6ff, .last = 0x6ff, .category = 1 },
    .{ .first = 0x710, .last = 0x710, .category = 1 },
    .{ .first = 0x711, .last = 0x711, .category = 2 },
    .{ .first = 0x712, .last = 0x72f, .category = 1 },
    .{ .first = 0x730, .last = 0x74a, .category = 2 },
    .{ .first = 0x74d, .last = 0x7a5, .category = 1 },
    .{ .first = 0x7a6, .last = 0x7b0, .category = 2 },
    .{ .first = 0x7b1, .last = 0x7b1, .category = 1 },
    .{ .first = 0x7c0, .last = 0x7c9, .category = 4 },
    .{ .first = 0x7ca, .last = 0x7ea, .category = 1 },
    .{ .first = 0x7eb, .last = 0x7f3, .category = 2 },
    .{ .first = 0x7f4, .last = 0x7f5, .category = 1 },
    .{ .first = 0x7fa, .last = 0x7fa, .category = 1 },
    .{ .first = 0x7fd, .last = 0x7fd, .category = 2 },
    .{ .first = 0x800, .last = 0x815, .category = 1 },
    .{ .first = 0x816, .last = 0x819, .category = 2 },
    .{ .first = 0x81a, .last = 0x81a, .category = 1 },
    .{ .first = 0x81b, .last = 0x823, .category = 2 },
    .{ .first = 0x824, .last = 0x824, .category = 1 },
    .{ .first = 0x825, .last = 0x827, .category = 2 },
    .{ .first = 0x828, .last = 0x828, .category = 1 },
    .{ .first = 0x829, .last = 0x82d, .category = 2 },
    .{ .first = 0x840, .last = 0x858, .category = 1 },
    .{ .first = 0x859, .last = 0x85b, .category = 2 },
    .{ .first = 0x860, .last = 0x86a, .category = 1 },
    .{ .first = 0x870, .last = 0x887, .category = 1 },
    .{ .first = 0x889, .last = 0x88e, .category = 1 },
    .{ .first = 0x897, .last = 0x89f, .category = 2 },
    .{ .first = 0x8a0, .last = 0x8c9, .category = 1 },
    .{ .first = 0x8ca, .last = 0x8e1, .category = 2 },
    .{ .first = 0x8e3, .last = 0x903, .category = 2 },
    .{ .first = 0x904, .last = 0x939, .category = 1 },
    .{ .first = 0x93a, .last = 0x93c, .category = 2 },
    .{ .first = 0x93d, .last = 0x93d, .category = 1 },
    .{ .first = 0x93e, .last = 0x94f, .category = 2 },
    .{ .first = 0x950, .last = 0x950, .category = 1 },
    .{ .first = 0x951, .last = 0x957, .category = 2 },
    .{ .first = 0x958, .last = 0x961, .category = 1 },
    .{ .first = 0x962, .last = 0x963, .category = 2 },
    .{ .first = 0x966, .last = 0x96f, .category = 4 },
    .{ .first = 0x971, .last = 0x980, .category = 1 },
    .{ .first = 0x981, .last = 0x983, .category = 2 },
    .{ .first = 0x985, .last = 0x98c, .category = 1 },
    .{ .first = 0x98f, .last = 0x990, .category = 1 },
    .{ .first = 0x993, .last = 0x9a8, .category = 1 },
    .{ .first = 0x9aa, .last = 0x9b0, .category = 1 },
    .{ .first = 0x9b2, .last = 0x9b2, .category = 1 },
    .{ .first = 0x9b6, .last = 0x9b9, .category = 1 },
    .{ .first = 0x9bc, .last = 0x9bc, .category = 2 },
    .{ .first = 0x9bd, .last = 0x9bd, .category = 1 },
    .{ .first = 0x9be, .last = 0x9c4, .category = 2 },
    .{ .first = 0x9c7, .last = 0x9c8, .category = 2 },
    .{ .first = 0x9cb, .last = 0x9cd, .category = 2 },
    .{ .first = 0x9ce, .last = 0x9ce, .category = 1 },
    .{ .first = 0x9d7, .last = 0x9d7, .category = 2 },
    .{ .first = 0x9dc, .last = 0x9dd, .category = 1 },
    .{ .first = 0x9df, .last = 0x9e1, .category = 1 },
    .{ .first = 0x9e2, .last = 0x9e3, .category = 2 },
    .{ .first = 0x9e6, .last = 0x9ef, .category = 4 },
    .{ .first = 0x9f0, .last = 0x9f1, .category = 1 },
    .{ .first = 0x9f4, .last = 0x9f9, .category = 4 },
    .{ .first = 0x9fc, .last = 0x9fc, .category = 1 },
    .{ .first = 0x9fe, .last = 0x9fe, .category = 2 },
    .{ .first = 0xa01, .last = 0xa03, .category = 2 },
    .{ .first = 0xa05, .last = 0xa0a, .category = 1 },
    .{ .first = 0xa0f, .last = 0xa10, .category = 1 },
    .{ .first = 0xa13, .last = 0xa28, .category = 1 },
    .{ .first = 0xa2a, .last = 0xa30, .category = 1 },
    .{ .first = 0xa32, .last = 0xa33, .category = 1 },
    .{ .first = 0xa35, .last = 0xa36, .category = 1 },
    .{ .first = 0xa38, .last = 0xa39, .category = 1 },
    .{ .first = 0xa3c, .last = 0xa3c, .category = 2 },
    .{ .first = 0xa3e, .last = 0xa42, .category = 2 },
    .{ .first = 0xa47, .last = 0xa48, .category = 2 },
    .{ .first = 0xa4b, .last = 0xa4d, .category = 2 },
    .{ .first = 0xa51, .last = 0xa51, .category = 2 },
    .{ .first = 0xa59, .last = 0xa5c, .category = 1 },
    .{ .first = 0xa5e, .last = 0xa5e, .category = 1 },
    .{ .first = 0xa66, .last = 0xa6f, .category = 4 },
    .{ .first = 0xa70, .last = 0xa71, .category = 2 },
    .{ .first = 0xa72, .last = 0xa74, .category = 1 },
    .{ .first = 0xa75, .last = 0xa75, .category = 2 },
    .{ .first = 0xa81, .last = 0xa83, .category = 2 },
    .{ .first = 0xa85, .last = 0xa8d, .category = 1 },
    .{ .first = 0xa8f, .last = 0xa91, .category = 1 },
    .{ .first = 0xa93, .last = 0xaa8, .category = 1 },
    .{ .first = 0xaaa, .last = 0xab0, .category = 1 },
    .{ .first = 0xab2, .last = 0xab3, .category = 1 },
    .{ .first = 0xab5, .last = 0xab9, .category = 1 },
    .{ .first = 0xabc, .last = 0xabc, .category = 2 },
    .{ .first = 0xabd, .last = 0xabd, .category = 1 },
    .{ .first = 0xabe, .last = 0xac5, .category = 2 },
    .{ .first = 0xac7, .last = 0xac9, .category = 2 },
    .{ .first = 0xacb, .last = 0xacd, .category = 2 },
    .{ .first = 0xad0, .last = 0xad0, .category = 1 },
    .{ .first = 0xae0, .last = 0xae1, .category = 1 },
    .{ .first = 0xae2, .last = 0xae3, .category = 2 },
    .{ .first = 0xae6, .last = 0xaef, .category = 4 },
    .{ .first = 0xaf9, .last = 0xaf9, .category = 1 },
    .{ .first = 0xafa, .last = 0xaff, .category = 2 },
    .{ .first = 0xb01, .last = 0xb03, .category = 2 },
    .{ .first = 0xb05, .last = 0xb0c, .category = 1 },
    .{ .first = 0xb0f, .last = 0xb10, .category = 1 },
    .{ .first = 0xb13, .last = 0xb28, .category = 1 },
    .{ .first = 0xb2a, .last = 0xb30, .category = 1 },
    .{ .first = 0xb32, .last = 0xb33, .category = 1 },
    .{ .first = 0xb35, .last = 0xb39, .category = 1 },
    .{ .first = 0xb3c, .last = 0xb3c, .category = 2 },
    .{ .first = 0xb3d, .last = 0xb3d, .category = 1 },
    .{ .first = 0xb3e, .last = 0xb44, .category = 2 },
    .{ .first = 0xb47, .last = 0xb48, .category = 2 },
    .{ .first = 0xb4b, .last = 0xb4d, .category = 2 },
    .{ .first = 0xb55, .last = 0xb57, .category = 2 },
    .{ .first = 0xb5c, .last = 0xb5d, .category = 1 },
    .{ .first = 0xb5f, .last = 0xb61, .category = 1 },
    .{ .first = 0xb62, .last = 0xb63, .category = 2 },
    .{ .first = 0xb66, .last = 0xb6f, .category = 4 },
    .{ .first = 0xb71, .last = 0xb71, .category = 1 },
    .{ .first = 0xb72, .last = 0xb77, .category = 4 },
    .{ .first = 0xb82, .last = 0xb82, .category = 2 },
    .{ .first = 0xb83, .last = 0xb83, .category = 1 },
    .{ .first = 0xb85, .last = 0xb8a, .category = 1 },
    .{ .first = 0xb8e, .last = 0xb90, .category = 1 },
    .{ .first = 0xb92, .last = 0xb95, .category = 1 },
    .{ .first = 0xb99, .last = 0xb9a, .category = 1 },
    .{ .first = 0xb9c, .last = 0xb9c, .category = 1 },
    .{ .first = 0xb9e, .last = 0xb9f, .category = 1 },
    .{ .first = 0xba3, .last = 0xba4, .category = 1 },
    .{ .first = 0xba8, .last = 0xbaa, .category = 1 },
    .{ .first = 0xbae, .last = 0xbb9, .category = 1 },
    .{ .first = 0xbbe, .last = 0xbc2, .category = 2 },
    .{ .first = 0xbc6, .last = 0xbc8, .category = 2 },
    .{ .first = 0xbca, .last = 0xbcd, .category = 2 },
    .{ .first = 0xbd0, .last = 0xbd0, .category = 1 },
    .{ .first = 0xbd7, .last = 0xbd7, .category = 2 },
    .{ .first = 0xbe6, .last = 0xbf2, .category = 4 },
    .{ .first = 0xc00, .last = 0xc04, .category = 2 },
    .{ .first = 0xc05, .last = 0xc0c, .category = 1 },
    .{ .first = 0xc0e, .last = 0xc10, .category = 1 },
    .{ .first = 0xc12, .last = 0xc28, .category = 1 },
    .{ .first = 0xc2a, .last = 0xc39, .category = 1 },
    .{ .first = 0xc3c, .last = 0xc3c, .category = 2 },
    .{ .first = 0xc3d, .last = 0xc3d, .category = 1 },
    .{ .first = 0xc3e, .last = 0xc44, .category = 2 },
    .{ .first = 0xc46, .last = 0xc48, .category = 2 },
    .{ .first = 0xc4a, .last = 0xc4d, .category = 2 },
    .{ .first = 0xc55, .last = 0xc56, .category = 2 },
    .{ .first = 0xc58, .last = 0xc5a, .category = 1 },
    .{ .first = 0xc5d, .last = 0xc5d, .category = 1 },
    .{ .first = 0xc60, .last = 0xc61, .category = 1 },
    .{ .first = 0xc62, .last = 0xc63, .category = 2 },
    .{ .first = 0xc66, .last = 0xc6f, .category = 4 },
    .{ .first = 0xc78, .last = 0xc7e, .category = 4 },
    .{ .first = 0xc80, .last = 0xc80, .category = 1 },
    .{ .first = 0xc81, .last = 0xc83, .category = 2 },
    .{ .first = 0xc85, .last = 0xc8c, .category = 1 },
    .{ .first = 0xc8e, .last = 0xc90, .category = 1 },
    .{ .first = 0xc92, .last = 0xca8, .category = 1 },
    .{ .first = 0xcaa, .last = 0xcb3, .category = 1 },
    .{ .first = 0xcb5, .last = 0xcb9, .category = 1 },
    .{ .first = 0xcbc, .last = 0xcbc, .category = 2 },
    .{ .first = 0xcbd, .last = 0xcbd, .category = 1 },
    .{ .first = 0xcbe, .last = 0xcc4, .category = 2 },
    .{ .first = 0xcc6, .last = 0xcc8, .category = 2 },
    .{ .first = 0xcca, .last = 0xccd, .category = 2 },
    .{ .first = 0xcd5, .last = 0xcd6, .category = 2 },
    .{ .first = 0xcdd, .last = 0xcde, .category = 1 },
    .{ .first = 0xce0, .last = 0xce1, .category = 1 },
    .{ .first = 0xce2, .last = 0xce3, .category = 2 },
    .{ .first = 0xce6, .last = 0xcef, .category = 4 },
    .{ .first = 0xcf1, .last = 0xcf2, .category = 1 },
    .{ .first = 0xcf3, .last = 0xcf3, .category = 2 },
    .{ .first = 0xd00, .last = 0xd03, .category = 2 },
    .{ .first = 0xd04, .last = 0xd0c, .category = 1 },
    .{ .first = 0xd0e, .last = 0xd10, .category = 1 },
    .{ .first = 0xd12, .last = 0xd3a, .category = 1 },
    .{ .first = 0xd3b, .last = 0xd3c, .category = 2 },
    .{ .first = 0xd3d, .last = 0xd3d, .category = 1 },
    .{ .first = 0xd3e, .last = 0xd44, .category = 2 },
    .{ .first = 0xd46, .last = 0xd48, .category = 2 },
    .{ .first = 0xd4a, .last = 0xd4d, .category = 2 },
    .{ .first = 0xd4e, .last = 0xd4e, .category = 1 },
    .{ .first = 0xd54, .last = 0xd56, .category = 1 },
    .{ .first = 0xd57, .last = 0xd57, .category = 2 },
    .{ .first = 0xd58, .last = 0xd5e, .category = 4 },
    .{ .first = 0xd5f, .last = 0xd61, .category = 1 },
    .{ .first = 0xd62, .last = 0xd63, .category = 2 },
    .{ .first = 0xd66, .last = 0xd78, .category = 4 },
    .{ .first = 0xd7a, .last = 0xd7f, .category = 1 },
    .{ .first = 0xd81, .last = 0xd83, .category = 2 },
    .{ .first = 0xd85, .last = 0xd96, .category = 1 },
    .{ .first = 0xd9a, .last = 0xdb1, .category = 1 },
    .{ .first = 0xdb3, .last = 0xdbb, .category = 1 },
    .{ .first = 0xdbd, .last = 0xdbd, .category = 1 },
    .{ .first = 0xdc0, .last = 0xdc6, .category = 1 },
    .{ .first = 0xdca, .last = 0xdca, .category = 2 },
    .{ .first = 0xdcf, .last = 0xdd4, .category = 2 },
    .{ .first = 0xdd6, .last = 0xdd6, .category = 2 },
    .{ .first = 0xdd8, .last = 0xddf, .category = 2 },
    .{ .first = 0xde6, .last = 0xdef, .category = 4 },
    .{ .first = 0xdf2, .last = 0xdf3, .category = 2 },
    .{ .first = 0xe01, .last = 0xe30, .category = 1 },
    .{ .first = 0xe31, .last = 0xe31, .category = 2 },
    .{ .first = 0xe32, .last = 0xe33, .category = 1 },
    .{ .first = 0xe34, .last = 0xe3a, .category = 2 },
    .{ .first = 0xe40, .last = 0xe46, .category = 1 },
    .{ .first = 0xe47, .last = 0xe4e, .category = 2 },
    .{ .first = 0xe50, .last = 0xe59, .category = 4 },
    .{ .first = 0xe81, .last = 0xe82, .category = 1 },
    .{ .first = 0xe84, .last = 0xe84, .category = 1 },
    .{ .first = 0xe86, .last = 0xe8a, .category = 1 },
    .{ .first = 0xe8c, .last = 0xea3, .category = 1 },
    .{ .first = 0xea5, .last = 0xea5, .category = 1 },
    .{ .first = 0xea7, .last = 0xeb0, .category = 1 },
    .{ .first = 0xeb1, .last = 0xeb1, .category = 2 },
    .{ .first = 0xeb2, .last = 0xeb3, .category = 1 },
    .{ .first = 0xeb4, .last = 0xebc, .category = 2 },
    .{ .first = 0xebd, .last = 0xebd, .category = 1 },
    .{ .first = 0xec0, .last = 0xec4, .category = 1 },
    .{ .first = 0xec6, .last = 0xec6, .category = 1 },
    .{ .first = 0xec8, .last = 0xece, .category = 2 },
    .{ .first = 0xed0, .last = 0xed9, .category = 4 },
    .{ .first = 0xedc, .last = 0xedf, .category = 1 },
    .{ .first = 0xf00, .last = 0xf00, .category = 1 },
    .{ .first = 0xf18, .last = 0xf19, .category = 2 },
    .{ .first = 0xf20, .last = 0xf33, .category = 4 },
    .{ .first = 0xf35, .last = 0xf35, .category = 2 },
    .{ .first = 0xf37, .last = 0xf37, .category = 2 },
    .{ .first = 0xf39, .last = 0xf39, .category = 2 },
    .{ .first = 0xf3e, .last = 0xf3f, .category = 2 },
    .{ .first = 0xf40, .last = 0xf47, .category = 1 },
    .{ .first = 0xf49, .last = 0xf6c, .category = 1 },
    .{ .first = 0xf71, .last = 0xf84, .category = 2 },
    .{ .first = 0xf86, .last = 0xf87, .category = 2 },
    .{ .first = 0xf88, .last = 0xf8c, .category = 1 },
    .{ .first = 0xf8d, .last = 0xf97, .category = 2 },
    .{ .first = 0xf99, .last = 0xfbc, .category = 2 },
    .{ .first = 0xfc6, .last = 0xfc6, .category = 2 },
    .{ .first = 0x1000, .last = 0x102a, .category = 1 },
    .{ .first = 0x102b, .last = 0x103e, .category = 2 },
    .{ .first = 0x103f, .last = 0x103f, .category = 1 },
    .{ .first = 0x1040, .last = 0x1049, .category = 4 },
    .{ .first = 0x1050, .last = 0x1055, .category = 1 },
    .{ .first = 0x1056, .last = 0x1059, .category = 2 },
    .{ .first = 0x105a, .last = 0x105d, .category = 1 },
    .{ .first = 0x105e, .last = 0x1060, .category = 2 },
    .{ .first = 0x1061, .last = 0x1061, .category = 1 },
    .{ .first = 0x1062, .last = 0x1064, .category = 2 },
    .{ .first = 0x1065, .last = 0x1066, .category = 1 },
    .{ .first = 0x1067, .last = 0x106d, .category = 2 },
    .{ .first = 0x106e, .last = 0x1070, .category = 1 },
    .{ .first = 0x1071, .last = 0x1074, .category = 2 },
    .{ .first = 0x1075, .last = 0x1081, .category = 1 },
    .{ .first = 0x1082, .last = 0x108d, .category = 2 },
    .{ .first = 0x108e, .last = 0x108e, .category = 1 },
    .{ .first = 0x108f, .last = 0x108f, .category = 2 },
    .{ .first = 0x1090, .last = 0x1099, .category = 4 },
    .{ .first = 0x109a, .last = 0x109d, .category = 2 },
    .{ .first = 0x10a0, .last = 0x10c5, .category = 1 },
    .{ .first = 0x10c7, .last = 0x10c7, .category = 1 },
    .{ .first = 0x10cd, .last = 0x10cd, .category = 1 },
    .{ .first = 0x10d0, .last = 0x10fa, .category = 1 },
    .{ .first = 0x10fc, .last = 0x1248, .category = 1 },
    .{ .first = 0x124a, .last = 0x124d, .category = 1 },
    .{ .first = 0x1250, .last = 0x1256, .category = 1 },
    .{ .first = 0x1258, .last = 0x1258, .category = 1 },
    .{ .first = 0x125a, .last = 0x125d, .category = 1 },
    .{ .first = 0x1260, .last = 0x1288, .category = 1 },
    .{ .first = 0x128a, .last = 0x128d, .category = 1 },
    .{ .first = 0x1290, .last = 0x12b0, .category = 1 },
    .{ .first = 0x12b2, .last = 0x12b5, .category = 1 },
    .{ .first = 0x12b8, .last = 0x12be, .category = 1 },
    .{ .first = 0x12c0, .last = 0x12c0, .category = 1 },
    .{ .first = 0x12c2, .last = 0x12c5, .category = 1 },
    .{ .first = 0x12c8, .last = 0x12d6, .category = 1 },
    .{ .first = 0x12d8, .last = 0x1310, .category = 1 },
    .{ .first = 0x1312, .last = 0x1315, .category = 1 },
    .{ .first = 0x1318, .last = 0x135a, .category = 1 },
    .{ .first = 0x135d, .last = 0x135f, .category = 2 },
    .{ .first = 0x1369, .last = 0x137c, .category = 4 },
    .{ .first = 0x1380, .last = 0x138f, .category = 1 },
    .{ .first = 0x13a0, .last = 0x13f5, .category = 1 },
    .{ .first = 0x13f8, .last = 0x13fd, .category = 1 },
    .{ .first = 0x1401, .last = 0x166c, .category = 1 },
    .{ .first = 0x166f, .last = 0x167f, .category = 1 },
    .{ .first = 0x1680, .last = 0x1680, .category = 8 },
    .{ .first = 0x1681, .last = 0x169a, .category = 1 },
    .{ .first = 0x16a0, .last = 0x16ea, .category = 1 },
    .{ .first = 0x16ee, .last = 0x16f0, .category = 4 },
    .{ .first = 0x16f1, .last = 0x16f8, .category = 1 },
    .{ .first = 0x1700, .last = 0x1711, .category = 1 },
    .{ .first = 0x1712, .last = 0x1715, .category = 2 },
    .{ .first = 0x171f, .last = 0x1731, .category = 1 },
    .{ .first = 0x1732, .last = 0x1734, .category = 2 },
    .{ .first = 0x1740, .last = 0x1751, .category = 1 },
    .{ .first = 0x1752, .last = 0x1753, .category = 2 },
    .{ .first = 0x1760, .last = 0x176c, .category = 1 },
    .{ .first = 0x176e, .last = 0x1770, .category = 1 },
    .{ .first = 0x1772, .last = 0x1773, .category = 2 },
    .{ .first = 0x1780, .last = 0x17b3, .category = 1 },
    .{ .first = 0x17b4, .last = 0x17d3, .category = 2 },
    .{ .first = 0x17d7, .last = 0x17d7, .category = 1 },
    .{ .first = 0x17dc, .last = 0x17dc, .category = 1 },
    .{ .first = 0x17dd, .last = 0x17dd, .category = 2 },
    .{ .first = 0x17e0, .last = 0x17e9, .category = 4 },
    .{ .first = 0x17f0, .last = 0x17f9, .category = 4 },
    .{ .first = 0x180b, .last = 0x180d, .category = 2 },
    .{ .first = 0x180f, .last = 0x180f, .category = 2 },
    .{ .first = 0x1810, .last = 0x1819, .category = 4 },
    .{ .first = 0x1820, .last = 0x1878, .category = 1 },
    .{ .first = 0x1880, .last = 0x1884, .category = 1 },
    .{ .first = 0x1885, .last = 0x1886, .category = 2 },
    .{ .first = 0x1887, .last = 0x18a8, .category = 1 },
    .{ .first = 0x18a9, .last = 0x18a9, .category = 2 },
    .{ .first = 0x18aa, .last = 0x18aa, .category = 1 },
    .{ .first = 0x18b0, .last = 0x18f5, .category = 1 },
    .{ .first = 0x1900, .last = 0x191e, .category = 1 },
    .{ .first = 0x1920, .last = 0x192b, .category = 2 },
    .{ .first = 0x1930, .last = 0x193b, .category = 2 },
    .{ .first = 0x1946, .last = 0x194f, .category = 4 },
    .{ .first = 0x1950, .last = 0x196d, .category = 1 },
    .{ .first = 0x1970, .last = 0x1974, .category = 1 },
    .{ .first = 0x1980, .last = 0x19ab, .category = 1 },
    .{ .first = 0x19b0, .last = 0x19c9, .category = 1 },
    .{ .first = 0x19d0, .last = 0x19da, .category = 4 },
    .{ .first = 0x1a00, .last = 0x1a16, .category = 1 },
    .{ .first = 0x1a17, .last = 0x1a1b, .category = 2 },
    .{ .first = 0x1a20, .last = 0x1a54, .category = 1 },
    .{ .first = 0x1a55, .last = 0x1a5e, .category = 2 },
    .{ .first = 0x1a60, .last = 0x1a7c, .category = 2 },
    .{ .first = 0x1a7f, .last = 0x1a7f, .category = 2 },
    .{ .first = 0x1a80, .last = 0x1a89, .category = 4 },
    .{ .first = 0x1a90, .last = 0x1a99, .category = 4 },
    .{ .first = 0x1aa7, .last = 0x1aa7, .category = 1 },
    .{ .first = 0x1ab0, .last = 0x1ace, .category = 2 },
    .{ .first = 0x1b00, .last = 0x1b04, .category = 2 },
    .{ .first = 0x1b05, .last = 0x1b33, .category = 1 },
    .{ .first = 0x1b34, .last = 0x1b44, .category = 2 },
    .{ .first = 0x1b45, .last = 0x1b4c, .category = 1 },
    .{ .first = 0x1b50, .last = 0x1b59, .category = 4 },
    .{ .first = 0x1b6b, .last = 0x1b73, .category = 2 },
    .{ .first = 0x1b80, .last = 0x1b82, .category = 2 },
    .{ .first = 0x1b83, .last = 0x1ba0, .category = 1 },
    .{ .first = 0x1ba1, .last = 0x1bad, .category = 2 },
    .{ .first = 0x1bae, .last = 0x1baf, .category = 1 },
    .{ .first = 0x1bb0, .last = 0x1bb9, .category = 4 },
    .{ .first = 0x1bba, .last = 0x1be5, .category = 1 },
    .{ .first = 0x1be6, .last = 0x1bf3, .category = 2 },
    .{ .first = 0x1c00, .last = 0x1c23, .category = 1 },
    .{ .first = 0x1c24, .last = 0x1c37, .category = 2 },
    .{ .first = 0x1c40, .last = 0x1c49, .category = 4 },
    .{ .first = 0x1c4d, .last = 0x1c4f, .category = 1 },
    .{ .first = 0x1c50, .last = 0x1c59, .category = 4 },
    .{ .first = 0x1c5a, .last = 0x1c7d, .category = 1 },
    .{ .first = 0x1c80, .last = 0x1c8a, .category = 1 },
    .{ .first = 0x1c90, .last = 0x1cba, .category = 1 },
    .{ .first = 0x1cbd, .last = 0x1cbf, .category = 1 },
    .{ .first = 0x1cd0, .last = 0x1cd2, .category = 2 },
    .{ .first = 0x1cd4, .last = 0x1ce8, .category = 2 },
    .{ .first = 0x1ce9, .last = 0x1cec, .category = 1 },
    .{ .first = 0x1ced, .last = 0x1ced, .category = 2 },
    .{ .first = 0x1cee, .last = 0x1cf3, .category = 1 },
    .{ .first = 0x1cf4, .last = 0x1cf4, .category = 2 },
    .{ .first = 0x1cf5, .last = 0x1cf6, .category = 1 },
    .{ .first = 0x1cf7, .last = 0x1cf9, .category = 2 },
    .{ .first = 0x1cfa, .last = 0x1cfa, .category = 1 },
    .{ .first = 0x1d00, .last = 0x1dbf, .category = 1 },
    .{ .first = 0x1dc0, .last = 0x1dff, .category = 2 },
    .{ .first = 0x1e00, .last = 0x1f15, .category = 1 },
    .{ .first = 0x1f18, .last = 0x1f1d, .category = 1 },
    .{ .first = 0x1f20, .last = 0x1f45, .category = 1 },
    .{ .first = 0x1f48, .last = 0x1f4d, .category = 1 },
    .{ .first = 0x1f50, .last = 0x1f57, .category = 1 },
    .{ .first = 0x1f59, .last = 0x1f59, .category = 1 },
    .{ .first = 0x1f5b, .last = 0x1f5b, .category = 1 },
    .{ .first = 0x1f5d, .last = 0x1f5d, .category = 1 },
    .{ .first = 0x1f5f, .last = 0x1f7d, .category = 1 },
    .{ .first = 0x1f80, .last = 0x1fb4, .category = 1 },
    .{ .first = 0x1fb6, .last = 0x1fbc, .category = 1 },
    .{ .first = 0x1fbe, .last = 0x1fbe, .category = 1 },
    .{ .first = 0x1fc2, .last = 0x1fc4, .category = 1 },
    .{ .first = 0x1fc6, .last = 0x1fcc, .category = 1 },
    .{ .first = 0x1fd0, .last = 0x1fd3, .category = 1 },
    .{ .first = 0x1fd6, .last = 0x1fdb, .category = 1 },
    .{ .first = 0x1fe0, .last = 0x1fec, .category = 1 },
    .{ .first = 0x1ff2, .last = 0x1ff4, .category = 1 },
    .{ .first = 0x1ff6, .last = 0x1ffc, .category = 1 },
    .{ .first = 0x2000, .last = 0x200a, .category = 8 },
    .{ .first = 0x2028, .last = 0x2029, .category = 8 },
    .{ .first = 0x202f, .last = 0x202f, .category = 8 },
    .{ .first = 0x205f, .last = 0x205f, .category = 8 },
    .{ .first = 0x2070, .last = 0x2070, .category = 4 },
    .{ .first = 0x2071, .last = 0x2071, .category = 1 },
    .{ .first = 0x2074, .last = 0x2079, .category = 4 },
    .{ .first = 0x207f, .last = 0x207f, .category = 1 },
    .{ .first = 0x2080, .last = 0x2089, .category = 4 },
    .{ .first = 0x2090, .last = 0x209c, .category = 1 },
    .{ .first = 0x20d0, .last = 0x20f0, .category = 2 },
    .{ .first = 0x2102, .last = 0x2102, .category = 1 },
    .{ .first = 0x2107, .last = 0x2107, .category = 1 },
    .{ .first = 0x210a, .last = 0x2113, .category = 1 },
    .{ .first = 0x2115, .last = 0x2115, .category = 1 },
    .{ .first = 0x2119, .last = 0x211d, .category = 1 },
    .{ .first = 0x2124, .last = 0x2124, .category = 1 },
    .{ .first = 0x2126, .last = 0x2126, .category = 1 },
    .{ .first = 0x2128, .last = 0x2128, .category = 1 },
    .{ .first = 0x212a, .last = 0x212d, .category = 1 },
    .{ .first = 0x212f, .last = 0x2139, .category = 1 },
    .{ .first = 0x213c, .last = 0x213f, .category = 1 },
    .{ .first = 0x2145, .last = 0x2149, .category = 1 },
    .{ .first = 0x214e, .last = 0x214e, .category = 1 },
    .{ .first = 0x2150, .last = 0x2182, .category = 4 },
    .{ .first = 0x2183, .last = 0x2184, .category = 1 },
    .{ .first = 0x2185, .last = 0x2189, .category = 4 },
    .{ .first = 0x2460, .last = 0x249b, .category = 4 },
    .{ .first = 0x24ea, .last = 0x24ff, .category = 4 },
    .{ .first = 0x2776, .last = 0x2793, .category = 4 },
    .{ .first = 0x2c00, .last = 0x2ce4, .category = 1 },
    .{ .first = 0x2ceb, .last = 0x2cee, .category = 1 },
    .{ .first = 0x2cef, .last = 0x2cf1, .category = 2 },
    .{ .first = 0x2cf2, .last = 0x2cf3, .category = 1 },
    .{ .first = 0x2cfd, .last = 0x2cfd, .category = 4 },
    .{ .first = 0x2d00, .last = 0x2d25, .category = 1 },
    .{ .first = 0x2d27, .last = 0x2d27, .category = 1 },
    .{ .first = 0x2d2d, .last = 0x2d2d, .category = 1 },
    .{ .first = 0x2d30, .last = 0x2d67, .category = 1 },
    .{ .first = 0x2d6f, .last = 0x2d6f, .category = 1 },
    .{ .first = 0x2d7f, .last = 0x2d7f, .category = 2 },
    .{ .first = 0x2d80, .last = 0x2d96, .category = 1 },
    .{ .first = 0x2da0, .last = 0x2da6, .category = 1 },
    .{ .first = 0x2da8, .last = 0x2dae, .category = 1 },
    .{ .first = 0x2db0, .last = 0x2db6, .category = 1 },
    .{ .first = 0x2db8, .last = 0x2dbe, .category = 1 },
    .{ .first = 0x2dc0, .last = 0x2dc6, .category = 1 },
    .{ .first = 0x2dc8, .last = 0x2dce, .category = 1 },
    .{ .first = 0x2dd0, .last = 0x2dd6, .category = 1 },
    .{ .first = 0x2dd8, .last = 0x2dde, .category = 1 },
    .{ .first = 0x2de0, .last = 0x2dff, .category = 2 },
    .{ .first = 0x2e2f, .last = 0x2e2f, .category = 1 },
    .{ .first = 0x3000, .last = 0x3000, .category = 8 },
    .{ .first = 0x3005, .last = 0x3006, .category = 1 },
    .{ .first = 0x3007, .last = 0x3007, .category = 4 },
    .{ .first = 0x3021, .last = 0x3029, .category = 4 },
    .{ .first = 0x302a, .last = 0x302f, .category = 2 },
    .{ .first = 0x3031, .last = 0x3035, .category = 1 },
    .{ .first = 0x3038, .last = 0x303a, .category = 4 },
    .{ .first = 0x303b, .last = 0x303c, .category = 1 },
    .{ .first = 0x3041, .last = 0x3096, .category = 1 },
    .{ .first = 0x3099, .last = 0x309a, .category = 2 },
    .{ .first = 0x309d, .last = 0x309f, .category = 1 },
    .{ .first = 0x30a1, .last = 0x30fa, .category = 1 },
    .{ .first = 0x30fc, .last = 0x30ff, .category = 1 },
    .{ .first = 0x3105, .last = 0x312f, .category = 1 },
    .{ .first = 0x3131, .last = 0x318e, .category = 1 },
    .{ .first = 0x3192, .last = 0x3195, .category = 4 },
    .{ .first = 0x31a0, .last = 0x31bf, .category = 1 },
    .{ .first = 0x31f0, .last = 0x31ff, .category = 1 },
    .{ .first = 0x3220, .last = 0x3229, .category = 4 },
    .{ .first = 0x3248, .last = 0x324f, .category = 4 },
    .{ .first = 0x3251, .last = 0x325f, .category = 4 },
    .{ .first = 0x3280, .last = 0x3289, .category = 4 },
    .{ .first = 0x32b1, .last = 0x32bf, .category = 4 },
    .{ .first = 0x3400, .last = 0x4dbf, .category = 1 },
    .{ .first = 0x4e00, .last = 0xa48c, .category = 1 },
    .{ .first = 0xa4d0, .last = 0xa4fd, .category = 1 },
    .{ .first = 0xa500, .last = 0xa60c, .category = 1 },
    .{ .first = 0xa610, .last = 0xa61f, .category = 1 },
    .{ .first = 0xa620, .last = 0xa629, .category = 4 },
    .{ .first = 0xa62a, .last = 0xa62b, .category = 1 },
    .{ .first = 0xa640, .last = 0xa66e, .category = 1 },
    .{ .first = 0xa66f, .last = 0xa672, .category = 2 },
    .{ .first = 0xa674, .last = 0xa67d, .category = 2 },
    .{ .first = 0xa67f, .last = 0xa69d, .category = 1 },
    .{ .first = 0xa69e, .last = 0xa69f, .category = 2 },
    .{ .first = 0xa6a0, .last = 0xa6e5, .category = 1 },
    .{ .first = 0xa6e6, .last = 0xa6ef, .category = 4 },
    .{ .first = 0xa6f0, .last = 0xa6f1, .category = 2 },
    .{ .first = 0xa717, .last = 0xa71f, .category = 1 },
    .{ .first = 0xa722, .last = 0xa788, .category = 1 },
    .{ .first = 0xa78b, .last = 0xa7cd, .category = 1 },
    .{ .first = 0xa7d0, .last = 0xa7d1, .category = 1 },
    .{ .first = 0xa7d3, .last = 0xa7d3, .category = 1 },
    .{ .first = 0xa7d5, .last = 0xa7dc, .category = 1 },
    .{ .first = 0xa7f2, .last = 0xa801, .category = 1 },
    .{ .first = 0xa802, .last = 0xa802, .category = 2 },
    .{ .first = 0xa803, .last = 0xa805, .category = 1 },
    .{ .first = 0xa806, .last = 0xa806, .category = 2 },
    .{ .first = 0xa807, .last = 0xa80a, .category = 1 },
    .{ .first = 0xa80b, .last = 0xa80b, .category = 2 },
    .{ .first = 0xa80c, .last = 0xa822, .category = 1 },
    .{ .first = 0xa823, .last = 0xa827, .category = 2 },
    .{ .first = 0xa82c, .last = 0xa82c, .category = 2 },
    .{ .first = 0xa830, .last = 0xa835, .category = 4 },
    .{ .first = 0xa840, .last = 0xa873, .category = 1 },
    .{ .first = 0xa880, .last = 0xa881, .category = 2 },
    .{ .first = 0xa882, .last = 0xa8b3, .category = 1 },
    .{ .first = 0xa8b4, .last = 0xa8c5, .category = 2 },
    .{ .first = 0xa8d0, .last = 0xa8d9, .category = 4 },
    .{ .first = 0xa8e0, .last = 0xa8f1, .category = 2 },
    .{ .first = 0xa8f2, .last = 0xa8f7, .category = 1 },
    .{ .first = 0xa8fb, .last = 0xa8fb, .category = 1 },
    .{ .first = 0xa8fd, .last = 0xa8fe, .category = 1 },
    .{ .first = 0xa8ff, .last = 0xa8ff, .category = 2 },
    .{ .first = 0xa900, .last = 0xa909, .category = 4 },
    .{ .first = 0xa90a, .last = 0xa925, .category = 1 },
    .{ .first = 0xa926, .last = 0xa92d, .category = 2 },
    .{ .first = 0xa930, .last = 0xa946, .category = 1 },
    .{ .first = 0xa947, .last = 0xa953, .category = 2 },
    .{ .first = 0xa960, .last = 0xa97c, .category = 1 },
    .{ .first = 0xa980, .last = 0xa983, .category = 2 },
    .{ .first = 0xa984, .last = 0xa9b2, .category = 1 },
    .{ .first = 0xa9b3, .last = 0xa9c0, .category = 2 },
    .{ .first = 0xa9cf, .last = 0xa9cf, .category = 1 },
    .{ .first = 0xa9d0, .last = 0xa9d9, .category = 4 },
    .{ .first = 0xa9e0, .last = 0xa9e4, .category = 1 },
    .{ .first = 0xa9e5, .last = 0xa9e5, .category = 2 },
    .{ .first = 0xa9e6, .last = 0xa9ef, .category = 1 },
    .{ .first = 0xa9f0, .last = 0xa9f9, .category = 4 },
    .{ .first = 0xa9fa, .last = 0xa9fe, .category = 1 },
    .{ .first = 0xaa00, .last = 0xaa28, .category = 1 },
    .{ .first = 0xaa29, .last = 0xaa36, .category = 2 },
    .{ .first = 0xaa40, .last = 0xaa42, .category = 1 },
    .{ .first = 0xaa43, .last = 0xaa43, .category = 2 },
    .{ .first = 0xaa44, .last = 0xaa4b, .category = 1 },
    .{ .first = 0xaa4c, .last = 0xaa4d, .category = 2 },
    .{ .first = 0xaa50, .last = 0xaa59, .category = 4 },
    .{ .first = 0xaa60, .last = 0xaa76, .category = 1 },
    .{ .first = 0xaa7a, .last = 0xaa7a, .category = 1 },
    .{ .first = 0xaa7b, .last = 0xaa7d, .category = 2 },
    .{ .first = 0xaa7e, .last = 0xaaaf, .category = 1 },
    .{ .first = 0xaab0, .last = 0xaab0, .category = 2 },
    .{ .first = 0xaab1, .last = 0xaab1, .category = 1 },
    .{ .first = 0xaab2, .last = 0xaab4, .category = 2 },
    .{ .first = 0xaab5, .last = 0xaab6, .category = 1 },
    .{ .first = 0xaab7, .last = 0xaab8, .category = 2 },
    .{ .first = 0xaab9, .last = 0xaabd, .category = 1 },
    .{ .first = 0xaabe, .last = 0xaabf, .category = 2 },
    .{ .first = 0xaac0, .last = 0xaac0, .category = 1 },
    .{ .first = 0xaac1, .last = 0xaac1, .category = 2 },
    .{ .first = 0xaac2, .last = 0xaac2, .category = 1 },
    .{ .first = 0xaadb, .last = 0xaadd, .category = 1 },
    .{ .first = 0xaae0, .last = 0xaaea, .category = 1 },
    .{ .first = 0xaaeb, .last = 0xaaef, .category = 2 },
    .{ .first = 0xaaf2, .last = 0xaaf4, .category = 1 },
    .{ .first = 0xaaf5, .last = 0xaaf6, .category = 2 },
    .{ .first = 0xab01, .last = 0xab06, .category = 1 },
    .{ .first = 0xab09, .last = 0xab0e, .category = 1 },
    .{ .first = 0xab11, .last = 0xab16, .category = 1 },
    .{ .first = 0xab20, .last = 0xab26, .category = 1 },
    .{ .first = 0xab28, .last = 0xab2e, .category = 1 },
    .{ .first = 0xab30, .last = 0xab5a, .category = 1 },
    .{ .first = 0xab5c, .last = 0xab69, .category = 1 },
    .{ .first = 0xab70, .last = 0xabe2, .category = 1 },
    .{ .first = 0xabe3, .last = 0xabea, .category = 2 },
    .{ .first = 0xabec, .last = 0xabed, .category = 2 },
    .{ .first = 0xabf0, .last = 0xabf9, .category = 4 },
    .{ .first = 0xac00, .last = 0xd7a3, .category = 1 },
    .{ .first = 0xd7b0, .last = 0xd7c6, .category = 1 },
    .{ .first = 0xd7cb, .last = 0xd7fb, .category = 1 },
    .{ .first = 0xf900, .last = 0xfa6d, .category = 1 },
    .{ .first = 0xfa70, .last = 0xfad9, .category = 1 },
    .{ .first = 0xfb00, .last = 0xfb06, .category = 1 },
    .{ .first = 0xfb13, .last = 0xfb17, .category = 1 },
    .{ .first = 0xfb1d, .last = 0xfb1d, .category = 1 },
    .{ .first = 0xfb1e, .last = 0xfb1e, .category = 2 },
    .{ .first = 0xfb1f, .last = 0xfb28, .category = 1 },
    .{ .first = 0xfb2a, .last = 0xfb36, .category = 1 },
    .{ .first = 0xfb38, .last = 0xfb3c, .category = 1 },
    .{ .first = 0xfb3e, .last = 0xfb3e, .category = 1 },
    .{ .first = 0xfb40, .last = 0xfb41, .category = 1 },
    .{ .first = 0xfb43, .last = 0xfb44, .category = 1 },
    .{ .first = 0xfb46, .last = 0xfbb1, .category = 1 },
    .{ .first = 0xfbd3, .last = 0xfd3d, .category = 1 },
    .{ .first = 0xfd50, .last = 0xfd8f, .category = 1 },
    .{ .first = 0xfd92, .last = 0xfdc7, .category = 1 },
    .{ .first = 0xfdf0, .last = 0xfdfb, .category = 1 },
    .{ .first = 0xfe00, .last = 0xfe0f, .category = 2 },
    .{ .first = 0xfe20, .last = 0xfe2f, .category = 2 },
    .{ .first = 0xfe70, .last = 0xfe74, .category = 1 },
    .{ .first = 0xfe76, .last = 0xfefc, .category = 1 },
    .{ .first = 0xff10, .last = 0xff19, .category = 4 },
    .{ .first = 0xff21, .last = 0xff3a, .category = 1 },
    .{ .first = 0xff41, .last = 0xff5a, .category = 1 },
    .{ .first = 0xff66, .last = 0xffbe, .category = 1 },
    .{ .first = 0xffc2, .last = 0xffc7, .category = 1 },
    .{ .first = 0xffca, .last = 0xffcf, .category = 1 },
    .{ .first = 0xffd2, .last = 0xffd7, .category = 1 },
    .{ .first = 0xffda, .last = 0xffdc, .category = 1 },
    .{ .first = 0x10000, .last = 0x1000b, .category = 1 },
    .{ .first = 0x1000d, .last = 0x10026, .category = 1 },
    .{ .first = 0x10028, .last = 0x1003a, .category = 1 },
    .{ .first = 0x1003c, .last = 0x1003d, .category = 1 },
    .{ .first = 0x1003f, .last = 0x1004d, .category = 1 },
    .{ .first = 0x10050, .last = 0x1005d, .category = 1 },
    .{ .first = 0x10080, .last = 0x100fa, .category = 1 },
    .{ .first = 0x10107, .last = 0x10133, .category = 4 },
    .{ .first = 0x10140, .last = 0x10178, .category = 4 },
    .{ .first = 0x1018a, .last = 0x1018b, .category = 4 },
    .{ .first = 0x101fd, .last = 0x101fd, .category = 2 },
    .{ .first = 0x10280, .last = 0x1029c, .category = 1 },
    .{ .first = 0x102a0, .last = 0x102d0, .category = 1 },
    .{ .first = 0x102e0, .last = 0x102e0, .category = 2 },
    .{ .first = 0x102e1, .last = 0x102fb, .category = 4 },
    .{ .first = 0x10300, .last = 0x1031f, .category = 1 },
    .{ .first = 0x10320, .last = 0x10323, .category = 4 },
    .{ .first = 0x1032d, .last = 0x10340, .category = 1 },
    .{ .first = 0x10341, .last = 0x10341, .category = 4 },
    .{ .first = 0x10342, .last = 0x10349, .category = 1 },
    .{ .first = 0x1034a, .last = 0x1034a, .category = 4 },
    .{ .first = 0x10350, .last = 0x10375, .category = 1 },
    .{ .first = 0x10376, .last = 0x1037a, .category = 2 },
    .{ .first = 0x10380, .last = 0x1039d, .category = 1 },
    .{ .first = 0x103a0, .last = 0x103c3, .category = 1 },
    .{ .first = 0x103c8, .last = 0x103cf, .category = 1 },
    .{ .first = 0x103d1, .last = 0x103d5, .category = 4 },
    .{ .first = 0x10400, .last = 0x1049d, .category = 1 },
    .{ .first = 0x104a0, .last = 0x104a9, .category = 4 },
    .{ .first = 0x104b0, .last = 0x104d3, .category = 1 },
    .{ .first = 0x104d8, .last = 0x104fb, .category = 1 },
    .{ .first = 0x10500, .last = 0x10527, .category = 1 },
    .{ .first = 0x10530, .last = 0x10563, .category = 1 },
    .{ .first = 0x10570, .last = 0x1057a, .category = 1 },
    .{ .first = 0x1057c, .last = 0x1058a, .category = 1 },
    .{ .first = 0x1058c, .last = 0x10592, .category = 1 },
    .{ .first = 0x10594, .last = 0x10595, .category = 1 },
    .{ .first = 0x10597, .last = 0x105a1, .category = 1 },
    .{ .first = 0x105a3, .last = 0x105b1, .category = 1 },
    .{ .first = 0x105b3, .last = 0x105b9, .category = 1 },
    .{ .first = 0x105bb, .last = 0x105bc, .category = 1 },
    .{ .first = 0x105c0, .last = 0x105f3, .category = 1 },
    .{ .first = 0x10600, .last = 0x10736, .category = 1 },
    .{ .first = 0x10740, .last = 0x10755, .category = 1 },
    .{ .first = 0x10760, .last = 0x10767, .category = 1 },
    .{ .first = 0x10780, .last = 0x10785, .category = 1 },
    .{ .first = 0x10787, .last = 0x107b0, .category = 1 },
    .{ .first = 0x107b2, .last = 0x107ba, .category = 1 },
    .{ .first = 0x10800, .last = 0x10805, .category = 1 },
    .{ .first = 0x10808, .last = 0x10808, .category = 1 },
    .{ .first = 0x1080a, .last = 0x10835, .category = 1 },
    .{ .first = 0x10837, .last = 0x10838, .category = 1 },
    .{ .first = 0x1083c, .last = 0x1083c, .category = 1 },
    .{ .first = 0x1083f, .last = 0x10855, .category = 1 },
    .{ .first = 0x10858, .last = 0x1085f, .category = 4 },
    .{ .first = 0x10860, .last = 0x10876, .category = 1 },
    .{ .first = 0x10879, .last = 0x1087f, .category = 4 },
    .{ .first = 0x10880, .last = 0x1089e, .category = 1 },
    .{ .first = 0x108a7, .last = 0x108af, .category = 4 },
    .{ .first = 0x108e0, .last = 0x108f2, .category = 1 },
    .{ .first = 0x108f4, .last = 0x108f5, .category = 1 },
    .{ .first = 0x108fb, .last = 0x108ff, .category = 4 },
    .{ .first = 0x10900, .last = 0x10915, .category = 1 },
    .{ .first = 0x10916, .last = 0x1091b, .category = 4 },
    .{ .first = 0x10920, .last = 0x10939, .category = 1 },
    .{ .first = 0x10980, .last = 0x109b7, .category = 1 },
    .{ .first = 0x109bc, .last = 0x109bd, .category = 4 },
    .{ .first = 0x109be, .last = 0x109bf, .category = 1 },
    .{ .first = 0x109c0, .last = 0x109cf, .category = 4 },
    .{ .first = 0x109d2, .last = 0x109ff, .category = 4 },
    .{ .first = 0x10a00, .last = 0x10a00, .category = 1 },
    .{ .first = 0x10a01, .last = 0x10a03, .category = 2 },
    .{ .first = 0x10a05, .last = 0x10a06, .category = 2 },
    .{ .first = 0x10a0c, .last = 0x10a0f, .category = 2 },
    .{ .first = 0x10a10, .last = 0x10a13, .category = 1 },
    .{ .first = 0x10a15, .last = 0x10a17, .category = 1 },
    .{ .first = 0x10a19, .last = 0x10a35, .category = 1 },
    .{ .first = 0x10a38, .last = 0x10a3a, .category = 2 },
    .{ .first = 0x10a3f, .last = 0x10a3f, .category = 2 },
    .{ .first = 0x10a40, .last = 0x10a48, .category = 4 },
    .{ .first = 0x10a60, .last = 0x10a7c, .category = 1 },
    .{ .first = 0x10a7d, .last = 0x10a7e, .category = 4 },
    .{ .first = 0x10a80, .last = 0x10a9c, .category = 1 },
    .{ .first = 0x10a9d, .last = 0x10a9f, .category = 4 },
    .{ .first = 0x10ac0, .last = 0x10ac7, .category = 1 },
    .{ .first = 0x10ac9, .last = 0x10ae4, .category = 1 },
    .{ .first = 0x10ae5, .last = 0x10ae6, .category = 2 },
    .{ .first = 0x10aeb, .last = 0x10aef, .category = 4 },
    .{ .first = 0x10b00, .last = 0x10b35, .category = 1 },
    .{ .first = 0x10b40, .last = 0x10b55, .category = 1 },
    .{ .first = 0x10b58, .last = 0x10b5f, .category = 4 },
    .{ .first = 0x10b60, .last = 0x10b72, .category = 1 },
    .{ .first = 0x10b78, .last = 0x10b7f, .category = 4 },
    .{ .first = 0x10b80, .last = 0x10b91, .category = 1 },
    .{ .first = 0x10ba9, .last = 0x10baf, .category = 4 },
    .{ .first = 0x10c00, .last = 0x10c48, .category = 1 },
    .{ .first = 0x10c80, .last = 0x10cb2, .category = 1 },
    .{ .first = 0x10cc0, .last = 0x10cf2, .category = 1 },
    .{ .first = 0x10cfa, .last = 0x10cff, .category = 4 },
    .{ .first = 0x10d00, .last = 0x10d23, .category = 1 },
    .{ .first = 0x10d24, .last = 0x10d27, .category = 2 },
    .{ .first = 0x10d30, .last = 0x10d39, .category = 4 },
    .{ .first = 0x10d40, .last = 0x10d49, .category = 4 },
    .{ .first = 0x10d4a, .last = 0x10d65, .category = 1 },
    .{ .first = 0x10d69, .last = 0x10d6d, .category = 2 },
    .{ .first = 0x10d6f, .last = 0x10d85, .category = 1 },
    .{ .first = 0x10e60, .last = 0x10e7e, .category = 4 },
    .{ .first = 0x10e80, .last = 0x10ea9, .category = 1 },
    .{ .first = 0x10eab, .last = 0x10eac, .category = 2 },
    .{ .first = 0x10eb0, .last = 0x10eb1, .category = 1 },
    .{ .first = 0x10ec2, .last = 0x10ec4, .category = 1 },
    .{ .first = 0x10efc, .last = 0x10eff, .category = 2 },
    .{ .first = 0x10f00, .last = 0x10f1c, .category = 1 },
    .{ .first = 0x10f1d, .last = 0x10f26, .category = 4 },
    .{ .first = 0x10f27, .last = 0x10f27, .category = 1 },
    .{ .first = 0x10f30, .last = 0x10f45, .category = 1 },
    .{ .first = 0x10f46, .last = 0x10f50, .category = 2 },
    .{ .first = 0x10f51, .last = 0x10f54, .category = 4 },
    .{ .first = 0x10f70, .last = 0x10f81, .category = 1 },
    .{ .first = 0x10f82, .last = 0x10f85, .category = 2 },
    .{ .first = 0x10fb0, .last = 0x10fc4, .category = 1 },
    .{ .first = 0x10fc5, .last = 0x10fcb, .category = 4 },
    .{ .first = 0x10fe0, .last = 0x10ff6, .category = 1 },
    .{ .first = 0x11000, .last = 0x11002, .category = 2 },
    .{ .first = 0x11003, .last = 0x11037, .category = 1 },
    .{ .first = 0x11038, .last = 0x11046, .category = 2 },
    .{ .first = 0x11052, .last = 0x1106f, .category = 4 },
    .{ .first = 0x11070, .last = 0x11070, .category = 2 },
    .{ .first = 0x11071, .last = 0x11072, .category = 1 },
    .{ .first = 0x11073, .last = 0x11074, .category = 2 },
    .{ .first = 0x11075, .last = 0x11075, .category = 1 },
    .{ .first = 0x1107f, .last = 0x11082, .category = 2 },
    .{ .first = 0x11083, .last = 0x110af, .category = 1 },
    .{ .first = 0x110b0, .last = 0x110ba, .category = 2 },
    .{ .first = 0x110c2, .last = 0x110c2, .category = 2 },
    .{ .first = 0x110d0, .last = 0x110e8, .category = 1 },
    .{ .first = 0x110f0, .last = 0x110f9, .category = 4 },
    .{ .first = 0x11100, .last = 0x11102, .category = 2 },
    .{ .first = 0x11103, .last = 0x11126, .category = 1 },
    .{ .first = 0x11127, .last = 0x11134, .category = 2 },
    .{ .first = 0x11136, .last = 0x1113f, .category = 4 },
    .{ .first = 0x11144, .last = 0x11144, .category = 1 },
    .{ .first = 0x11145, .last = 0x11146, .category = 2 },
    .{ .first = 0x11147, .last = 0x11147, .category = 1 },
    .{ .first = 0x11150, .last = 0x11172, .category = 1 },
    .{ .first = 0x11173, .last = 0x11173, .category = 2 },
    .{ .first = 0x11176, .last = 0x11176, .category = 1 },
    .{ .first = 0x11180, .last = 0x11182, .category = 2 },
    .{ .first = 0x11183, .last = 0x111b2, .category = 1 },
    .{ .first = 0x111b3, .last = 0x111c0, .category = 2 },
    .{ .first = 0x111c1, .last = 0x111c4, .category = 1 },
    .{ .first = 0x111c9, .last = 0x111cc, .category = 2 },
    .{ .first = 0x111ce, .last = 0x111cf, .category = 2 },
    .{ .first = 0x111d0, .last = 0x111d9, .category = 4 },
    .{ .first = 0x111da, .last = 0x111da, .category = 1 },
    .{ .first = 0x111dc, .last = 0x111dc, .category = 1 },
    .{ .first = 0x111e1, .last = 0x111f4, .category = 4 },
    .{ .first = 0x11200, .last = 0x11211, .category = 1 },
    .{ .first = 0x11213, .last = 0x1122b, .category = 1 },
    .{ .first = 0x1122c, .last = 0x11237, .category = 2 },
    .{ .first = 0x1123e, .last = 0x1123e, .category = 2 },
    .{ .first = 0x1123f, .last = 0x11240, .category = 1 },
    .{ .first = 0x11241, .last = 0x11241, .category = 2 },
    .{ .first = 0x11280, .last = 0x11286, .category = 1 },
    .{ .first = 0x11288, .last = 0x11288, .category = 1 },
    .{ .first = 0x1128a, .last = 0x1128d, .category = 1 },
    .{ .first = 0x1128f, .last = 0x1129d, .category = 1 },
    .{ .first = 0x1129f, .last = 0x112a8, .category = 1 },
    .{ .first = 0x112b0, .last = 0x112de, .category = 1 },
    .{ .first = 0x112df, .last = 0x112ea, .category = 2 },
    .{ .first = 0x112f0, .last = 0x112f9, .category = 4 },
    .{ .first = 0x11300, .last = 0x11303, .category = 2 },
    .{ .first = 0x11305, .last = 0x1130c, .category = 1 },
    .{ .first = 0x1130f, .last = 0x11310, .category = 1 },
    .{ .first = 0x11313, .last = 0x11328, .category = 1 },
    .{ .first = 0x1132a, .last = 0x11330, .category = 1 },
    .{ .first = 0x11332, .last = 0x11333, .category = 1 },
    .{ .first = 0x11335, .last = 0x11339, .category = 1 },
    .{ .first = 0x1133b, .last = 0x1133c, .category = 2 },
    .{ .first = 0x1133d, .last = 0x1133d, .category = 1 },
    .{ .first = 0x1133e, .last = 0x11344, .category = 2 },
    .{ .first = 0x11347, .last = 0x11348, .category = 2 },
    .{ .first = 0x1134b, .last = 0x1134d, .category = 2 },
    .{ .first = 0x11350, .last = 0x11350, .category = 1 },
    .{ .first = 0x11357, .last = 0x11357, .category = 2 },
    .{ .first = 0x1135d, .last = 0x11361, .category = 1 },
    .{ .first = 0x11362, .last = 0x11363, .category = 2 },
    .{ .first = 0x11366, .last = 0x1136c, .category = 2 },
    .{ .first = 0x11370, .last = 0x11374, .category = 2 },
    .{ .first = 0x11380, .last = 0x11389, .category = 1 },
    .{ .first = 0x1138b, .last = 0x1138b, .category = 1 },
    .{ .first = 0x1138e, .last = 0x1138e, .category = 1 },
    .{ .first = 0x11390, .last = 0x113b5, .category = 1 },
    .{ .first = 0x113b7, .last = 0x113b7, .category = 1 },
    .{ .first = 0x113b8, .last = 0x113c0, .category = 2 },
    .{ .first = 0x113c2, .last = 0x113c2, .category = 2 },
    .{ .first = 0x113c5, .last = 0x113c5, .category = 2 },
    .{ .first = 0x113c7, .last = 0x113ca, .category = 2 },
    .{ .first = 0x113cc, .last = 0x113d0, .category = 2 },
    .{ .first = 0x113d1, .last = 0x113d1, .category = 1 },
    .{ .first = 0x113d2, .last = 0x113d2, .category = 2 },
    .{ .first = 0x113d3, .last = 0x113d3, .category = 1 },
    .{ .first = 0x113e1, .last = 0x113e2, .category = 2 },
    .{ .first = 0x11400, .last = 0x11434, .category = 1 },
    .{ .first = 0x11435, .last = 0x11446, .category = 2 },
    .{ .first = 0x11447, .last = 0x1144a, .category = 1 },
    .{ .first = 0x11450, .last = 0x11459, .category = 4 },
    .{ .first = 0x1145e, .last = 0x1145e, .category = 2 },
    .{ .first = 0x1145f, .last = 0x11461, .category = 1 },
    .{ .first = 0x11480, .last = 0x114af, .category = 1 },
    .{ .first = 0x114b0, .last = 0x114c3, .category = 2 },
    .{ .first = 0x114c4, .last = 0x114c5, .category = 1 },
    .{ .first = 0x114c7, .last = 0x114c7, .category = 1 },
    .{ .first = 0x114d0, .last = 0x114d9, .category = 4 },
    .{ .first = 0x11580, .last = 0x115ae, .category = 1 },
    .{ .first = 0x115af, .last = 0x115b5, .category = 2 },
    .{ .first = 0x115b8, .last = 0x115c0, .category = 2 },
    .{ .first = 0x115d8, .last = 0x115db, .category = 1 },
    .{ .first = 0x115dc, .last = 0x115dd, .category = 2 },
    .{ .first = 0x11600, .last = 0x1162f, .category = 1 },
    .{ .first = 0x11630, .last = 0x11640, .category = 2 },
    .{ .first = 0x11644, .last = 0x11644, .category = 1 },
    .{ .first = 0x11650, .last = 0x11659, .category = 4 },
    .{ .first = 0x11680, .last = 0x116aa, .category = 1 },
    .{ .first = 0x116ab, .last = 0x116b7, .category = 2 },
    .{ .first = 0x116b8, .last = 0x116b8, .category = 1 },
    .{ .first = 0x116c0, .last = 0x116c9, .category = 4 },
    .{ .first = 0x116d0, .last = 0x116e3, .category = 4 },
    .{ .first = 0x11700, .last = 0x1171a, .category = 1 },
    .{ .first = 0x1171d, .last = 0x1172b, .category = 2 },
    .{ .first = 0x11730, .last = 0x1173b, .category = 4 },
    .{ .first = 0x11740, .last = 0x11746, .category = 1 },
    .{ .first = 0x11800, .last = 0x1182b, .category = 1 },
    .{ .first = 0x1182c, .last = 0x1183a, .category = 2 },
    .{ .first = 0x118a0, .last = 0x118df, .category = 1 },
    .{ .first = 0x118e0, .last = 0x118f2, .category = 4 },
    .{ .first = 0x118ff, .last = 0x11906, .category = 1 },
    .{ .first = 0x11909, .last = 0x11909, .category = 1 },
    .{ .first = 0x1190c, .last = 0x11913, .category = 1 },
    .{ .first = 0x11915, .last = 0x11916, .category = 1 },
    .{ .first = 0x11918, .last = 0x1192f, .category = 1 },
    .{ .first = 0x11930, .last = 0x11935, .category = 2 },
    .{ .first = 0x11937, .last = 0x11938, .category = 2 },
    .{ .first = 0x1193b, .last = 0x1193e, .category = 2 },
    .{ .first = 0x1193f, .last = 0x1193f, .category = 1 },
    .{ .first = 0x11940, .last = 0x11940, .category = 2 },
    .{ .first = 0x11941, .last = 0x11941, .category = 1 },
    .{ .first = 0x11942, .last = 0x11943, .category = 2 },
    .{ .first = 0x11950, .last = 0x11959, .category = 4 },
    .{ .first = 0x119a0, .last = 0x119a7, .category = 1 },
    .{ .first = 0x119aa, .last = 0x119d0, .category = 1 },
    .{ .first = 0x119d1, .last = 0x119d7, .category = 2 },
    .{ .first = 0x119da, .last = 0x119e0, .category = 2 },
    .{ .first = 0x119e1, .last = 0x119e1, .category = 1 },
    .{ .first = 0x119e3, .last = 0x119e3, .category = 1 },
    .{ .first = 0x119e4, .last = 0x119e4, .category = 2 },
    .{ .first = 0x11a00, .last = 0x11a00, .category = 1 },
    .{ .first = 0x11a01, .last = 0x11a0a, .category = 2 },
    .{ .first = 0x11a0b, .last = 0x11a32, .category = 1 },
    .{ .first = 0x11a33, .last = 0x11a39, .category = 2 },
    .{ .first = 0x11a3a, .last = 0x11a3a, .category = 1 },
    .{ .first = 0x11a3b, .last = 0x11a3e, .category = 2 },
    .{ .first = 0x11a47, .last = 0x11a47, .category = 2 },
    .{ .first = 0x11a50, .last = 0x11a50, .category = 1 },
    .{ .first = 0x11a51, .last = 0x11a5b, .category = 2 },
    .{ .first = 0x11a5c, .last = 0x11a89, .category = 1 },
    .{ .first = 0x11a8a, .last = 0x11a99, .category = 2 },
    .{ .first = 0x11a9d, .last = 0x11a9d, .category = 1 },
    .{ .first = 0x11ab0, .last = 0x11af8, .category = 1 },
    .{ .first = 0x11bc0, .last = 0x11be0, .category = 1 },
    .{ .first = 0x11bf0, .last = 0x11bf9, .category = 4 },
    .{ .first = 0x11c00, .last = 0x11c08, .category = 1 },
    .{ .first = 0x11c0a, .last = 0x11c2e, .category = 1 },
    .{ .first = 0x11c2f, .last = 0x11c36, .category = 2 },
    .{ .first = 0x11c38, .last = 0x11c3f, .category = 2 },
    .{ .first = 0x11c40, .last = 0x11c40, .category = 1 },
    .{ .first = 0x11c50, .last = 0x11c6c, .category = 4 },
    .{ .first = 0x11c72, .last = 0x11c8f, .category = 1 },
    .{ .first = 0x11c92, .last = 0x11ca7, .category = 2 },
    .{ .first = 0x11ca9, .last = 0x11cb6, .category = 2 },
    .{ .first = 0x11d00, .last = 0x11d06, .category = 1 },
    .{ .first = 0x11d08, .last = 0x11d09, .category = 1 },
    .{ .first = 0x11d0b, .last = 0x11d30, .category = 1 },
    .{ .first = 0x11d31, .last = 0x11d36, .category = 2 },
    .{ .first = 0x11d3a, .last = 0x11d3a, .category = 2 },
    .{ .first = 0x11d3c, .last = 0x11d3d, .category = 2 },
    .{ .first = 0x11d3f, .last = 0x11d45, .category = 2 },
    .{ .first = 0x11d46, .last = 0x11d46, .category = 1 },
    .{ .first = 0x11d47, .last = 0x11d47, .category = 2 },
    .{ .first = 0x11d50, .last = 0x11d59, .category = 4 },
    .{ .first = 0x11d60, .last = 0x11d65, .category = 1 },
    .{ .first = 0x11d67, .last = 0x11d68, .category = 1 },
    .{ .first = 0x11d6a, .last = 0x11d89, .category = 1 },
    .{ .first = 0x11d8a, .last = 0x11d8e, .category = 2 },
    .{ .first = 0x11d90, .last = 0x11d91, .category = 2 },
    .{ .first = 0x11d93, .last = 0x11d97, .category = 2 },
    .{ .first = 0x11d98, .last = 0x11d98, .category = 1 },
    .{ .first = 0x11da0, .last = 0x11da9, .category = 4 },
    .{ .first = 0x11ee0, .last = 0x11ef2, .category = 1 },
    .{ .first = 0x11ef3, .last = 0x11ef6, .category = 2 },
    .{ .first = 0x11f00, .last = 0x11f01, .category = 2 },
    .{ .first = 0x11f02, .last = 0x11f02, .category = 1 },
    .{ .first = 0x11f03, .last = 0x11f03, .category = 2 },
    .{ .first = 0x11f04, .last = 0x11f10, .category = 1 },
    .{ .first = 0x11f12, .last = 0x11f33, .category = 1 },
    .{ .first = 0x11f34, .last = 0x11f3a, .category = 2 },
    .{ .first = 0x11f3e, .last = 0x11f42, .category = 2 },
    .{ .first = 0x11f50, .last = 0x11f59, .category = 4 },
    .{ .first = 0x11f5a, .last = 0x11f5a, .category = 2 },
    .{ .first = 0x11fb0, .last = 0x11fb0, .category = 1 },
    .{ .first = 0x11fc0, .last = 0x11fd4, .category = 4 },
    .{ .first = 0x12000, .last = 0x12399, .category = 1 },
    .{ .first = 0x12400, .last = 0x1246e, .category = 4 },
    .{ .first = 0x12480, .last = 0x12543, .category = 1 },
    .{ .first = 0x12f90, .last = 0x12ff0, .category = 1 },
    .{ .first = 0x13000, .last = 0x1342f, .category = 1 },
    .{ .first = 0x13440, .last = 0x13440, .category = 2 },
    .{ .first = 0x13441, .last = 0x13446, .category = 1 },
    .{ .first = 0x13447, .last = 0x13455, .category = 2 },
    .{ .first = 0x13460, .last = 0x143fa, .category = 1 },
    .{ .first = 0x14400, .last = 0x14646, .category = 1 },
    .{ .first = 0x16100, .last = 0x1611d, .category = 1 },
    .{ .first = 0x1611e, .last = 0x1612f, .category = 2 },
    .{ .first = 0x16130, .last = 0x16139, .category = 4 },
    .{ .first = 0x16800, .last = 0x16a38, .category = 1 },
    .{ .first = 0x16a40, .last = 0x16a5e, .category = 1 },
    .{ .first = 0x16a60, .last = 0x16a69, .category = 4 },
    .{ .first = 0x16a70, .last = 0x16abe, .category = 1 },
    .{ .first = 0x16ac0, .last = 0x16ac9, .category = 4 },
    .{ .first = 0x16ad0, .last = 0x16aed, .category = 1 },
    .{ .first = 0x16af0, .last = 0x16af4, .category = 2 },
    .{ .first = 0x16b00, .last = 0x16b2f, .category = 1 },
    .{ .first = 0x16b30, .last = 0x16b36, .category = 2 },
    .{ .first = 0x16b40, .last = 0x16b43, .category = 1 },
    .{ .first = 0x16b50, .last = 0x16b59, .category = 4 },
    .{ .first = 0x16b5b, .last = 0x16b61, .category = 4 },
    .{ .first = 0x16b63, .last = 0x16b77, .category = 1 },
    .{ .first = 0x16b7d, .last = 0x16b8f, .category = 1 },
    .{ .first = 0x16d40, .last = 0x16d6c, .category = 1 },
    .{ .first = 0x16d70, .last = 0x16d79, .category = 4 },
    .{ .first = 0x16e40, .last = 0x16e7f, .category = 1 },
    .{ .first = 0x16e80, .last = 0x16e96, .category = 4 },
    .{ .first = 0x16f00, .last = 0x16f4a, .category = 1 },
    .{ .first = 0x16f4f, .last = 0x16f4f, .category = 2 },
    .{ .first = 0x16f50, .last = 0x16f50, .category = 1 },
    .{ .first = 0x16f51, .last = 0x16f87, .category = 2 },
    .{ .first = 0x16f8f, .last = 0x16f92, .category = 2 },
    .{ .first = 0x16f93, .last = 0x16f9f, .category = 1 },
    .{ .first = 0x16fe0, .last = 0x16fe1, .category = 1 },
    .{ .first = 0x16fe3, .last = 0x16fe3, .category = 1 },
    .{ .first = 0x16fe4, .last = 0x16fe4, .category = 2 },
    .{ .first = 0x16ff0, .last = 0x16ff1, .category = 2 },
    .{ .first = 0x17000, .last = 0x187f7, .category = 1 },
    .{ .first = 0x18800, .last = 0x18cd5, .category = 1 },
    .{ .first = 0x18cff, .last = 0x18d08, .category = 1 },
    .{ .first = 0x1aff0, .last = 0x1aff3, .category = 1 },
    .{ .first = 0x1aff5, .last = 0x1affb, .category = 1 },
    .{ .first = 0x1affd, .last = 0x1affe, .category = 1 },
    .{ .first = 0x1b000, .last = 0x1b122, .category = 1 },
    .{ .first = 0x1b132, .last = 0x1b132, .category = 1 },
    .{ .first = 0x1b150, .last = 0x1b152, .category = 1 },
    .{ .first = 0x1b155, .last = 0x1b155, .category = 1 },
    .{ .first = 0x1b164, .last = 0x1b167, .category = 1 },
    .{ .first = 0x1b170, .last = 0x1b2fb, .category = 1 },
    .{ .first = 0x1bc00, .last = 0x1bc6a, .category = 1 },
    .{ .first = 0x1bc70, .last = 0x1bc7c, .category = 1 },
    .{ .first = 0x1bc80, .last = 0x1bc88, .category = 1 },
    .{ .first = 0x1bc90, .last = 0x1bc99, .category = 1 },
    .{ .first = 0x1bc9d, .last = 0x1bc9e, .category = 2 },
    .{ .first = 0x1ccf0, .last = 0x1ccf9, .category = 4 },
    .{ .first = 0x1cf00, .last = 0x1cf2d, .category = 2 },
    .{ .first = 0x1cf30, .last = 0x1cf46, .category = 2 },
    .{ .first = 0x1d165, .last = 0x1d169, .category = 2 },
    .{ .first = 0x1d16d, .last = 0x1d172, .category = 2 },
    .{ .first = 0x1d17b, .last = 0x1d182, .category = 2 },
    .{ .first = 0x1d185, .last = 0x1d18b, .category = 2 },
    .{ .first = 0x1d1aa, .last = 0x1d1ad, .category = 2 },
    .{ .first = 0x1d242, .last = 0x1d244, .category = 2 },
    .{ .first = 0x1d2c0, .last = 0x1d2d3, .category = 4 },
    .{ .first = 0x1d2e0, .last = 0x1d2f3, .category = 4 },
    .{ .first = 0x1d360, .last = 0x1d378, .category = 4 },
    .{ .first = 0x1d400, .last = 0x1d454, .category = 1 },
    .{ .first = 0x1d456, .last = 0x1d49c, .category = 1 },
    .{ .first = 0x1d49e, .last = 0x1d49f, .category = 1 },
    .{ .first = 0x1d4a2, .last = 0x1d4a2, .category = 1 },
    .{ .first = 0x1d4a5, .last = 0x1d4a6, .category = 1 },
    .{ .first = 0x1d4a9, .last = 0x1d4ac, .category = 1 },
    .{ .first = 0x1d4ae, .last = 0x1d4b9, .category = 1 },
    .{ .first = 0x1d4bb, .last = 0x1d4bb, .category = 1 },
    .{ .first = 0x1d4bd, .last = 0x1d4c3, .category = 1 },
    .{ .first = 0x1d4c5, .last = 0x1d505, .category = 1 },
    .{ .first = 0x1d507, .last = 0x1d50a, .category = 1 },
    .{ .first = 0x1d50d, .last = 0x1d514, .category = 1 },
    .{ .first = 0x1d516, .last = 0x1d51c, .category = 1 },
    .{ .first = 0x1d51e, .last = 0x1d539, .category = 1 },
    .{ .first = 0x1d53b, .last = 0x1d53e, .category = 1 },
    .{ .first = 0x1d540, .last = 0x1d544, .category = 1 },
    .{ .first = 0x1d546, .last = 0x1d546, .category = 1 },
    .{ .first = 0x1d54a, .last = 0x1d550, .category = 1 },
    .{ .first = 0x1d552, .last = 0x1d6a5, .category = 1 },
    .{ .first = 0x1d6a8, .last = 0x1d6c0, .category = 1 },
    .{ .first = 0x1d6c2, .last = 0x1d6da, .category = 1 },
    .{ .first = 0x1d6dc, .last = 0x1d6fa, .category = 1 },
    .{ .first = 0x1d6fc, .last = 0x1d714, .category = 1 },
    .{ .first = 0x1d716, .last = 0x1d734, .category = 1 },
    .{ .first = 0x1d736, .last = 0x1d74e, .category = 1 },
    .{ .first = 0x1d750, .last = 0x1d76e, .category = 1 },
    .{ .first = 0x1d770, .last = 0x1d788, .category = 1 },
    .{ .first = 0x1d78a, .last = 0x1d7a8, .category = 1 },
    .{ .first = 0x1d7aa, .last = 0x1d7c2, .category = 1 },
    .{ .first = 0x1d7c4, .last = 0x1d7cb, .category = 1 },
    .{ .first = 0x1d7ce, .last = 0x1d7ff, .category = 4 },
    .{ .first = 0x1da00, .last = 0x1da36, .category = 2 },
    .{ .first = 0x1da3b, .last = 0x1da6c, .category = 2 },
    .{ .first = 0x1da75, .last = 0x1da75, .category = 2 },
    .{ .first = 0x1da84, .last = 0x1da84, .category = 2 },
    .{ .first = 0x1da9b, .last = 0x1da9f, .category = 2 },
    .{ .first = 0x1daa1, .last = 0x1daaf, .category = 2 },
    .{ .first = 0x1df00, .last = 0x1df1e, .category = 1 },
    .{ .first = 0x1df25, .last = 0x1df2a, .category = 1 },
    .{ .first = 0x1e000, .last = 0x1e006, .category = 2 },
    .{ .first = 0x1e008, .last = 0x1e018, .category = 2 },
    .{ .first = 0x1e01b, .last = 0x1e021, .category = 2 },
    .{ .first = 0x1e023, .last = 0x1e024, .category = 2 },
    .{ .first = 0x1e026, .last = 0x1e02a, .category = 2 },
    .{ .first = 0x1e030, .last = 0x1e06d, .category = 1 },
    .{ .first = 0x1e08f, .last = 0x1e08f, .category = 2 },
    .{ .first = 0x1e100, .last = 0x1e12c, .category = 1 },
    .{ .first = 0x1e130, .last = 0x1e136, .category = 2 },
    .{ .first = 0x1e137, .last = 0x1e13d, .category = 1 },
    .{ .first = 0x1e140, .last = 0x1e149, .category = 4 },
    .{ .first = 0x1e14e, .last = 0x1e14e, .category = 1 },
    .{ .first = 0x1e290, .last = 0x1e2ad, .category = 1 },
    .{ .first = 0x1e2ae, .last = 0x1e2ae, .category = 2 },
    .{ .first = 0x1e2c0, .last = 0x1e2eb, .category = 1 },
    .{ .first = 0x1e2ec, .last = 0x1e2ef, .category = 2 },
    .{ .first = 0x1e2f0, .last = 0x1e2f9, .category = 4 },
    .{ .first = 0x1e4d0, .last = 0x1e4eb, .category = 1 },
    .{ .first = 0x1e4ec, .last = 0x1e4ef, .category = 2 },
    .{ .first = 0x1e4f0, .last = 0x1e4f9, .category = 4 },
    .{ .first = 0x1e5d0, .last = 0x1e5ed, .category = 1 },
    .{ .first = 0x1e5ee, .last = 0x1e5ef, .category = 2 },
    .{ .first = 0x1e5f0, .last = 0x1e5f0, .category = 1 },
    .{ .first = 0x1e5f1, .last = 0x1e5fa, .category = 4 },
    .{ .first = 0x1e7e0, .last = 0x1e7e6, .category = 1 },
    .{ .first = 0x1e7e8, .last = 0x1e7eb, .category = 1 },
    .{ .first = 0x1e7ed, .last = 0x1e7ee, .category = 1 },
    .{ .first = 0x1e7f0, .last = 0x1e7fe, .category = 1 },
    .{ .first = 0x1e800, .last = 0x1e8c4, .category = 1 },
    .{ .first = 0x1e8c7, .last = 0x1e8cf, .category = 4 },
    .{ .first = 0x1e8d0, .last = 0x1e8d6, .category = 2 },
    .{ .first = 0x1e900, .last = 0x1e943, .category = 1 },
    .{ .first = 0x1e944, .last = 0x1e94a, .category = 2 },
    .{ .first = 0x1e94b, .last = 0x1e94b, .category = 1 },
    .{ .first = 0x1e950, .last = 0x1e959, .category = 4 },
    .{ .first = 0x1ec71, .last = 0x1ecab, .category = 4 },
    .{ .first = 0x1ecad, .last = 0x1ecaf, .category = 4 },
    .{ .first = 0x1ecb1, .last = 0x1ecb4, .category = 4 },
    .{ .first = 0x1ed01, .last = 0x1ed2d, .category = 4 },
    .{ .first = 0x1ed2f, .last = 0x1ed3d, .category = 4 },
    .{ .first = 0x1ee00, .last = 0x1ee03, .category = 1 },
    .{ .first = 0x1ee05, .last = 0x1ee1f, .category = 1 },
    .{ .first = 0x1ee21, .last = 0x1ee22, .category = 1 },
    .{ .first = 0x1ee24, .last = 0x1ee24, .category = 1 },
    .{ .first = 0x1ee27, .last = 0x1ee27, .category = 1 },
    .{ .first = 0x1ee29, .last = 0x1ee32, .category = 1 },
    .{ .first = 0x1ee34, .last = 0x1ee37, .category = 1 },
    .{ .first = 0x1ee39, .last = 0x1ee39, .category = 1 },
    .{ .first = 0x1ee3b, .last = 0x1ee3b, .category = 1 },
    .{ .first = 0x1ee42, .last = 0x1ee42, .category = 1 },
    .{ .first = 0x1ee47, .last = 0x1ee47, .category = 1 },
    .{ .first = 0x1ee49, .last = 0x1ee49, .category = 1 },
    .{ .first = 0x1ee4b, .last = 0x1ee4b, .category = 1 },
    .{ .first = 0x1ee4d, .last = 0x1ee4f, .category = 1 },
    .{ .first = 0x1ee51, .last = 0x1ee52, .category = 1 },
    .{ .first = 0x1ee54, .last = 0x1ee54, .category = 1 },
    .{ .first = 0x1ee57, .last = 0x1ee57, .category = 1 },
    .{ .first = 0x1ee59, .last = 0x1ee59, .category = 1 },
    .{ .first = 0x1ee5b, .last = 0x1ee5b, .category = 1 },
    .{ .first = 0x1ee5d, .last = 0x1ee5d, .category = 1 },
    .{ .first = 0x1ee5f, .last = 0x1ee5f, .category = 1 },
    .{ .first = 0x1ee61, .last = 0x1ee62, .category = 1 },
    .{ .first = 0x1ee64, .last = 0x1ee64, .category = 1 },
    .{ .first = 0x1ee67, .last = 0x1ee6a, .category = 1 },
    .{ .first = 0x1ee6c, .last = 0x1ee72, .category = 1 },
    .{ .first = 0x1ee74, .last = 0x1ee77, .category = 1 },
    .{ .first = 0x1ee79, .last = 0x1ee7c, .category = 1 },
    .{ .first = 0x1ee7e, .last = 0x1ee7e, .category = 1 },
    .{ .first = 0x1ee80, .last = 0x1ee89, .category = 1 },
    .{ .first = 0x1ee8b, .last = 0x1ee9b, .category = 1 },
    .{ .first = 0x1eea1, .last = 0x1eea3, .category = 1 },
    .{ .first = 0x1eea5, .last = 0x1eea9, .category = 1 },
    .{ .first = 0x1eeab, .last = 0x1eebb, .category = 1 },
    .{ .first = 0x1f100, .last = 0x1f10c, .category = 4 },
    .{ .first = 0x1fbf0, .last = 0x1fbf9, .category = 4 },
    .{ .first = 0x20000, .last = 0x2a6df, .category = 1 },
    .{ .first = 0x2a700, .last = 0x2b739, .category = 1 },
    .{ .first = 0x2b740, .last = 0x2b81d, .category = 1 },
    .{ .first = 0x2b820, .last = 0x2cea1, .category = 1 },
    .{ .first = 0x2ceb0, .last = 0x2ebe0, .category = 1 },
    .{ .first = 0x2ebf0, .last = 0x2ee5d, .category = 1 },
    .{ .first = 0x2f800, .last = 0x2fa1d, .category = 1 },
    .{ .first = 0x30000, .last = 0x3134a, .category = 1 },
    .{ .first = 0x31350, .last = 0x323af, .category = 1 },
    .{ .first = 0xe0100, .last = 0xe01ef, .category = 2 },
};
