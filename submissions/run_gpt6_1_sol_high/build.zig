const std = @import("std");
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const mod = b.createModule(.{ .root_source_file = b.path("src/main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const generated = b.addWriteFiles();
    const names = [_][]const u8{ "matvec", "hadamard", "norm", "element", "embed", "conv", "gdn", "attprep", "attscore", "softmax", "attvalue", "argmax", "gdn_batch", "matvec_fast", "dequant", "cast", "matmul_f16", "flash", "attreduce", "matvec_vocab" };
    var shader_source: std.Io.Writer.Allocating = .init(b.allocator);
    for (names) |name| {
        const compile = b.addSystemCommand(&.{ "glslc", "--target-env=vulkan1.4", "-O", "-Ishaders" });
        compile.addFileInput(b.path("shaders/common.glsl"));
        const source_name = if (std.mem.eql(u8, name, "gdn_batch")) "gdn_batch_parallel" else if (std.mem.eql(u8, name, "gdn")) "gdn_fast" else name;
        compile.addFileArg(b.path(b.fmt("shaders/{s}.comp", .{source_name})));
        compile.addArg("-o");
        const output = compile.addOutputFileArg(b.fmt("{s}.spv", .{name}));
        _ = generated.addCopyFile(output, b.fmt("{s}.spv", .{name}));
        shader_source.writer.print("pub const {s} align(4) = @embedFile(\"{s}.spv\").*;\n", .{ name, name }) catch @panic("OOM");
    }
    const shader_zig = generated.add("shaders.zig", shader_source.written());
    mod.addImport("shaders", b.createModule(.{ .root_source_file = shader_zig }));
    mod.addLibraryPath(.{ .cwd_relative = "/opt/vulkan-sdk/lib/VulkanLoader/lib" });
    mod.addRPath(.{ .cwd_relative = "/opt/vulkan-sdk/lib/VulkanLoader/lib" });
    mod.linkSystemLibrary("vulkan", .{});
    const lib = b.addLibrary(.{ .name = "bonsai", .linkage = .dynamic, .root_module = mod });
    b.installArtifact(lib);
}
