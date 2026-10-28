const std = @import("std");
const uni = @import("uni_tables.zig");

pub const Special = struct { text: []const u8, id: u32 };

pub const Tokenizer = struct {
    alloc: std.mem.Allocator,
    tok_map: std.StringHashMap(u32),
    merge_map: std.StringHashMap(u32),
    specials: std.ArrayListUnmanaged(Special) = .empty,
    b2u: [256]u21 = undefined,
    b2id: [256]u32 = undefined,
    built: bool = false,

    pub fn build(
        alloc: std.mem.Allocator,
        tokens: [][]u8,
        merges: [][]u8,
        types: []i32,
    ) !Tokenizer {
        var self = Tokenizer{
            .alloc = alloc,
            .tok_map = std.StringHashMap(u32).init(alloc),
            .merge_map = std.StringHashMap(u32).init(alloc),
        };
        // gpt2 byte -> unicode codepoint
        var bs = std.mem.zeroes([256]bool);
        var b: usize = 0x21;
        while (b <= 0x7E) : (b += 1) bs[b] = true;
        b = 0xA1;
        while (b <= 0xAC) : (b += 1) bs[b] = true;
        b = 0xAE;
        while (b <= 0xFF) : (b += 1) bs[b] = true;
        var n: u21 = 0;
        b = 0;
        while (b < 256) : (b += 1) {
            if (bs[b]) {
                self.b2u[b] = @intCast(b);
            } else {
                self.b2u[b] = 256 + n;
                n += 1;
            }
        }
        for (tokens, 0..) |t, i| {
            try self.tok_map.put(t, @intCast(i));
        }
        for (merges, 0..) |mg, i| {
            try self.merge_map.put(mg, @intCast(i));
        }
        self.b2id = undefined;
        b = 0;
        while (b < 256) : (b += 1) {
            var buf: [4]u8 = undefined;
            const len = std.unicode.utf8Encode(self.b2u[b], &buf) catch 0;
            self.b2id[b] = self.tok_map.get(buf[0..len]) orelse 0;
        }
        if (types.len == tokens.len) {
            for (types, 0..) |t, i| {
                if (t == 3 or t == 4) {
                    try self.specials.append(alloc, .{ .text = tokens[i], .id = @intCast(i) });
                }
            }
        }
        self.built = true;
        return self;
    }

    pub fn deinit(self: *Tokenizer) void {
        self.tok_map.deinit();
        self.merge_map.deinit();
        self.specials.deinit(self.alloc);
    }

    fn isL(cp: u21) bool {
        return uni.catOf(cp) == 1;
    }
    fn isM(cp: u21) bool {
        return uni.catOf(cp) == 2;
    }
    fn isN(cp: u21) bool {
        return uni.catOf(cp) == 3;
    }
    fn isS(cp: u21) bool {
        return uni.isSpace(cp);
    }

    fn byteLenOfUtf8(c: u8) usize {
        if (c < 0x80) return 1;
        if (c & 0xE0 == 0xC0) return 2;
        if (c & 0xF0 == 0xE0) return 3;
        if (c & 0xF8 == 0xF0) return 4;
        return 1;
    }

    /// Match a special token at position i (longest match); returns .{len, id} or null.
    fn matchSpecial(self: *const Tokenizer, text: []const u8, i: usize) ?Special {
        var best: ?Special = null;
        for (self.specials.items) |sp| {
            if (sp.text.len == 0 or i + sp.text.len > text.len) continue;
            if (std.mem.eql(u8, text[i .. i + sp.text.len], sp.text)) {
                if (best == null or sp.text.len > best.?.text.len) best = sp;
            }
        }
        return best;
    }

    /// Tokenize utf8 text into out; returns the number of ids written.
    pub fn tokenize(self: *const Tokenizer, text: []const u8, out: []i32) !usize {
        const alloc = self.alloc;
        var n_out: usize = 0;
        var i: usize = 0;
        // mapped buffer for the current pre-token
        var mapped = std.ArrayListUnmanaged(u8).empty;
        defer mapped.deinit(alloc);
        while (i < text.len) {
            // special token atomic match
            if (self.matchSpecial(text, i)) |sp| {
                if (n_out >= out.len) return error.BufferTooSmall;
                out[n_out] = @intCast(sp.id);
                n_out += 1;
                i += sp.text.len;
                continue;
            }
            // determine pre-token end (raw byte offset)
            const start = i;
            var j = i;
            var matched = false;
            // alt1: contractions
            if (text[j] == '\'' and j + 1 < text.len) {
                const c1 = std.ascii.toLower(text[j + 1]);
                if (c1 == 's' or c1 == 't' or c1 == 'm' or c1 == 'd') {
                    j += 2;
                    matched = true;
                } else if (j + 2 < text.len) {
                    const c2 = std.ascii.toLower(text[j + 2]);
                    if ((c1 == 'r' and c2 == 'e') or (c1 == 'v' and c2 == 'e') or (c1 == 'l' and c2 == 'l')) {
                        j += 3;
                        matched = true;
                    }
                }
            }
            if (!matched) {
                // decode helper: cp at byte pos
                const Dec = struct {
                    fn cp(t: []const u8, p: usize) u21 {
                        const l = byteLenOfUtf8(t[p]);
                        if (p + l > t.len) return t[p];
                        return std.unicode.utf8Decode(t[p .. p + l]) catch t[p];
                    }
                    fn len(t: []const u8, p: usize) usize {
                        const l = byteLenOfUtf8(t[p]);
                        return if (p + l > t.len) 1 else l;
                    }
                };
                // alt2: [^\r\n L N]? [L M]+
                var k = j;
                const c0 = Dec.cp(text, k);
                if (c0 != '\r' and c0 != '\n' and !isL(c0) and !isN(c0)) {
                    k += Dec.len(text, k);
                }
                if (k < text.len) {
                    const ck = Dec.cp(text, k);
                    if (isL(ck) or isM(ck)) {
                        k += Dec.len(text, k);
                        while (k < text.len) {
                            const cc = Dec.cp(text, k);
                            if (!(isL(cc) or isM(cc))) break;
                            k += Dec.len(text, k);
                        }
                        j = k;
                        matched = true;
                    }
                }
                // alt3: single number
                if (!matched) {
                    const ck = Dec.cp(text, start);
                    if (isN(ck)) {
                        j = start + Dec.len(text, start);
                        matched = true;
                    }
                }
                // alt4: ' '? [^ \s L M N]+ [\r\n]*
                if (!matched) {
                    var m = start;
                    const c0b = Dec.cp(text, m);
                    if (c0b == ' ') m += Dec.len(text, m);
                    var any = false;
                    while (m < text.len) {
                        const cc = Dec.cp(text, m);
                        if (isS(cc) or isL(cc) or isM(cc) or isN(cc)) break;
                        m += Dec.len(text, m);
                        any = true;
                    }
                    if (any) {
                        while (m < text.len) {
                            const cc = Dec.cp(text, m);
                            if (cc != '\r' and cc != '\n') break;
                            m += Dec.len(text, m);
                        }
                        j = m;
                        matched = true;
                    }
                }
                // whitespace run starting at start
                var run_end = start;
                var prev_pos = start;
                var ws_count: usize = 0;
                var last_nl_end: usize = 0;
                while (run_end < text.len) {
                    const cc = Dec.cp(text, run_end);
                    if (!isS(cc)) break;
                    prev_pos = run_end;
                    run_end += Dec.len(text, run_end);
                    ws_count += 1;
                    if (cc == '\r' or cc == '\n') last_nl_end = run_end;
                }
                // alt5: \s*[\r\n]+  (longest prefix of the run ending with a newline)
                if (!matched and last_nl_end > start) {
                    j = last_nl_end;
                    matched = true;
                }
                // alt6: \s+(?!\S) ; alt7: \s+
                if (!matched and ws_count > 0) {
                    if (run_end == text.len) {
                        j = run_end;
                        matched = true;
                    } else if (ws_count >= 2) {
                        j = prev_pos;
                        matched = true;
                    } else {
                        j = run_end;
                        matched = true;
                    }
                }
                if (!matched) {
                    j = start + Dec.len(text, start);
                }
            }
            if (j <= start) j = start + 1;
            // map raw bytes to unicode byte-mapped form
            mapped.clearRetainingCapacity();
            var q = start;
            while (q < j) : (q += 1) {
                var buf: [4]u8 = undefined;
                const l = std.unicode.utf8Encode(self.b2u[text[q]], &buf) catch 0;
                try mapped.appendSlice(alloc, buf[0..l]);
            }
            try self.bpePiece(mapped.items, out, &n_out);
            i = j;
        }
        return n_out;
    }

    fn bpePiece(self: *const Tokenizer, piece: []const u8, out: []i32, n_out: *usize) !void {
        const alloc = self.alloc;
        // split piece into codepoint boundaries
        var bounds = std.ArrayListUnmanaged(usize).empty;
        defer bounds.deinit(alloc);
        try bounds.append(alloc, 0);
        var p: usize = 0;
        while (p < piece.len) {
            const l = byteLenOfUtf8(piece[p]);
            p += if (p + l <= piece.len) l else 1;
            try bounds.append(alloc, p);
        }
        var n_sym = bounds.items.len - 1;
        var keybuf = std.ArrayListUnmanaged(u8).empty;
        defer keybuf.deinit(alloc);
        while (n_sym > 1) {
            var best_rank: u32 = std.math.maxInt(u32);
            var best_i: usize = 0;
            var idx: usize = 0;
            while (idx + 2 < bounds.items.len) : (idx += 1) {
                const a = piece[bounds.items[idx]..bounds.items[idx + 1]];
                const b = piece[bounds.items[idx + 1]..bounds.items[idx + 2]];
                keybuf.clearRetainingCapacity();
                try keybuf.appendSlice(alloc, a);
                try keybuf.append(alloc, ' ');
                try keybuf.appendSlice(alloc, b);
                if (self.merge_map.get(keybuf.items)) |r| {
                    if (r < best_rank) {
                        best_rank = r;
                        best_i = idx;
                    }
                }
            }
            if (best_rank == std.math.maxInt(u32)) break;
            var z = best_i + 1;
            while (z + 1 < bounds.items.len) : (z += 1) bounds.items[z] = bounds.items[z + 1];
            bounds.items.len -= 1;
            n_sym -= 1;
        }
        // emit tokens
        idxLoop: for (bounds.items[0 .. bounds.items.len - 1], 0..) |bv, si| {
            const ev = bounds.items[si + 1];
            const sym = piece[bv..ev];
            if (self.tok_map.get(sym)) |id| {
                if (n_out.* >= out.len) return error.BufferTooSmall;
                out[n_out.*] = @intCast(id);
                n_out.* += 1;
            } else {
                for (sym) |ch| {
                    _ = ch;
                }
                // fallback: emit byte tokens for each raw mapped char
                var q = bv;
                while (q < ev) {
                    const l = byteLenOfUtf8(piece[q]);
                    const e2 = if (q + l <= ev) q + l else q + 1;
                    if (self.tok_map.get(piece[q..e2])) |id| {
                        if (n_out.* >= out.len) return error.BufferTooSmall;
                        out[n_out.*] = @intCast(id);
                        n_out.* += 1;
                    }
                    q = e2;
                }
                continue :idxLoop;
            }
        }
    }
};
