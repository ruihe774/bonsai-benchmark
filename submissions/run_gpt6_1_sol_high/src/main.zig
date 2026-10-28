const std = @import("std");
const engine = @import("engine.zig");
const Model = engine.Model;
const Context = engine.Context;
const alloc = std.heap.c_allocator;
threadlocal var last_error: ?[*:0]const u8 = null;
fn clear() void {
    last_error = null;
}
fn fail(msg: [*:0]const u8, code: i32) i32 {
    last_error = msg;
    return code;
}
fn err(e: anyerror) i32 {
    last_error = @errorName(e);
    return switch (e) {
        error.OutOfMemory => -6,
        error.ContextCapacity => -2,
        error.InvalidToken => -1,
        else => -6,
    };
}
export fn bonsai_abi_version() u32 {
    clear();
    return 1;
}
export fn bonsai_last_error() ?[*:0]const u8 {
    return last_error;
}
export fn bonsai_model_open(path: ?[*:0]const u8, out: ?*?*Model) i32 {
    clear();
    if (path == null or out == null) return fail("null model argument", -1);
    out.?.* = null;
    const m = Model.open(std.mem.span(path.?)) catch |e| return err(e);
    out.?.* = m;
    return 0;
}
export fn bonsai_model_close(m: ?*Model) void {
    clear();
    if (m) |p| p.close();
}
export fn bonsai_n_vocab(m: ?*const Model) i32 {
    clear();
    return if (m) |p| @intCast(p.n_vocab) else fail("null model", -1);
}
export fn bonsai_n_ctx_train(m: ?*const Model) i32 {
    clear();
    return if (m) |p| @intCast(p.n_ctx_train) else fail("null model", -1);
}
export fn bonsai_tokenize(m: ?*Model, text: ?[*:0]const u8, add_bos: i32, ids: ?[*]i32, capacity: i32) i32 {
    clear();
    if (m == null or text == null or capacity < 0) return fail("invalid tokenizer arguments", -1);
    const tokens = m.?.tokenizer.tokenize(std.mem.span(text.?), add_bos != 0) catch |e| return err(e);
    defer alloc.free(tokens);
    if (ids) |out| {
        if (capacity < tokens.len) return fail("token output capacity", -2);
        @memcpy(out[0..tokens.len], tokens);
    }
    return @intCast(tokens.len);
}
export fn bonsai_tensor_count(m: ?*const Model) i32 {
    clear();
    return if (m) |p| @intCast(p.gguf.tensors.len) else fail("null model", -1);
}
export fn bonsai_tensor_info(m: ?*const Model, index: i32, name: ?*?[*:0]const u8, typ: ?*i32, rank: ?*i32, ne: ?[*]i64) i32 {
    clear();
    if (m == null or index < 0 or index >= m.?.gguf.tensors.len) return fail("invalid tensor index", -1);
    const t = &m.?.gguf.tensors[@intCast(index)];
    if (name) |p| p.* = t.name.ptr;
    if (typ) |p| p.* = @intCast(t.typ);
    if (rank) |p| p.* = @intCast(t.n_dims);
    if (ne) |p| for (0..4) |i| {
        p[i] = @intCast(t.dims[i]);
    };
    return 0;
}
export fn bonsai_tensor_dequant(m: ?*const Model, name: ?[*:0]const u8, out: ?[*]f32, capacity: i64) i64 {
    clear();
    if (m == null or name == null or capacity < 0) return fail("invalid dequant arguments", -1);
    const t = m.?.gguf.findTensor(std.mem.span(name.?)) orelse return fail("tensor not found", -1);
    const n = t.elementCount();
    if (out == null or capacity < n) return fail("dequant output capacity", -2);
    t.dequant(out.?[0..n]) catch |e| return err(e);
    return @intCast(n);
}
export fn bonsai_context_create(m: ?*Model, n_ctx: i32, n_batch: i32, out: ?*?*Context) i32 {
    clear();
    if (m == null or n_ctx < 1 or n_ctx > 262144 or n_batch < 1 or out == null) return fail("invalid context arguments", -1);
    out.?.* = null;
    out.?.* = Context.create(m.?, @intCast(n_ctx)) catch |e| return err(e);
    return 0;
}
export fn bonsai_context_close(ctx: ?*Context) void {
    clear();
    if (ctx) |p| p.close();
}
export fn bonsai_context_reset(ctx: ?*Context) i32 {
    clear();
    if (ctx == null) return fail("null context", -1);
    ctx.?.reset() catch |e| return err(e);
    return 0;
}
export fn bonsai_eval(ctx: ?*Context, tokens: ?[*]const i32, n_tokens: i32, logits: ?[*]f32, capacity: i32) i32 {
    clear();
    if (ctx == null or tokens == null or n_tokens < 1 or capacity < 0) return fail("invalid eval arguments", -1);
    if (logits != null and capacity < ctx.?.model.n_vocab) return fail("logits output capacity", -2);
    ctx.?.eval(tokens.?[0..@intCast(n_tokens)], logits) catch |e| return err(e);
    return 0;
}
export fn bonsai_generate_greedy(ctx: ?*Context, prompt: ?[*]const i32, n_prompt: i32, n_predict: i32, flags: u32, out: ?[*]i32) i32 {
    clear();
    if (ctx == null or n_prompt < 0 or n_predict < 0 or (n_prompt > 0 and prompt == null) or (n_predict > 0 and out == null)) return fail("invalid generation arguments", -1);
    const p = ctx.?;
    if (@as(u64, @intCast(n_prompt)) + @as(u64, @intCast(n_predict)) > p.n_ctx - p.pos) return fail("generation context capacity", -2);
    if (n_prompt > 0) {
        p.eval(prompt.?[0..@intCast(n_prompt)], null) catch |e| return err(e);
    } else if (p.pos == 0 and n_predict > 0) return fail("generation needs prompt or evaluated context", -5);
    var count: i32 = 0;
    while (count < n_predict) {
        const id = p.best() catch |e| return err(e);
        out.?[@intCast(count)] = id;
        count += 1;
        if (id == p.model.eos and (flags & 1) == 0) break;
        const one = [_]i32{id};
        p.eval(&one, null) catch |e| return err(e);
    }
    return count;
}
