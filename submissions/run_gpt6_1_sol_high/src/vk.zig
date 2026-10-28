const std = @import("std");
pub const c = @import("c.zig");
pub const Buffer = struct { handle: c.VkBuffer = null, memory: c.VkDeviceMemory = null, ptr: [*]u8 = undefined, mapped: bool = false, size: usize = 0, addr: u64 = 0 };
pub const Pipeline = c.VkPipeline;
const profiling = false;
pub const Runtime = struct {
    instance: c.VkInstance = null,
    physical: c.VkPhysicalDevice = null,
    device: c.VkDevice = null,
    queue: c.VkQueue = null,
    family: u32 = 0,
    pool: c.VkCommandPool = null,
    cmd: c.VkCommandBuffer = null,
    fence: c.VkFence = null,
    layout: c.VkPipelineLayout = null,
    mem: c.VkPhysicalDeviceMemoryProperties = undefined,
    queries: c.VkQueryPool = null,
    query_count: u32 = 0,
    profile_op: u8 = 0,
    timer_scale: f64 = 1,
    profile_ids: [8192]u8 = undefined,
    pub fn init() !Runtime {
        var self: Runtime = .{};
        errdefer self.deinit();
        var app: c.VkApplicationInfo = std.mem.zeroes(c.VkApplicationInfo);
        app.sType = c.VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "bonsai-vk";
        app.apiVersion = c.VK_API_VERSION_1_4;
        var ii: c.VkInstanceCreateInfo = std.mem.zeroes(c.VkInstanceCreateInfo);
        ii.sType = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ii.pApplicationInfo = &app;
        try check(c.vkCreateInstance(&ii, null, &self.instance));
        var np: u32 = 0;
        try check(c.vkEnumeratePhysicalDevices(self.instance, &np, null));
        if (np == 0) return error.NoGPU;
        var devs: [16]c.VkPhysicalDevice = undefined;
        np = @min(np, 16);
        try check(c.vkEnumeratePhysicalDevices(self.instance, &np, &devs));
        for (devs[0..np]) |physical| {
            var properties: c.VkPhysicalDeviceProperties = undefined;
            c.vkGetPhysicalDeviceProperties(physical, &properties);
            if (properties.deviceType == c.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                self.physical = physical;
                break;
            }
        }
        if (self.physical == null) return error.NoDiscreteGPU;
        var props: c.VkPhysicalDeviceProperties = undefined;
        c.vkGetPhysicalDeviceProperties(self.physical, &props);
        if (props.apiVersion < c.VK_API_VERSION_1_4) return error.Vulkan14Required;
        self.timer_scale = props.limits.timestampPeriod;
        var nq: u32 = 0;
        c.vkGetPhysicalDeviceQueueFamilyProperties(self.physical, &nq, null);
        var qp: [32]c.VkQueueFamilyProperties = undefined;
        nq = @min(nq, 32);
        c.vkGetPhysicalDeviceQueueFamilyProperties(self.physical, &nq, &qp);
        var found = false;
        for (qp[0..nq], 0..) |q, i| {
            if ((q.queueFlags & c.VK_QUEUE_COMPUTE_BIT) != 0) {
                self.family = @intCast(i);
                found = true;
                break;
            }
        }
        if (!found) return error.NoComputeQueue;
        var f11 = std.mem.zeroes(c.VkPhysicalDeviceVulkan11Features);
        f11.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        var f12 = std.mem.zeroes(c.VkPhysicalDeviceVulkan12Features);
        f12.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        var f13 = std.mem.zeroes(c.VkPhysicalDeviceVulkan13Features);
        f13.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        var f14 = std.mem.zeroes(c.VkPhysicalDeviceVulkan14Features);
        f14.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
        var cooperative = std.mem.zeroes(c.VkPhysicalDeviceCooperativeMatrixFeaturesKHR);
        cooperative.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
        var maintenance7 = std.mem.zeroes(c.VkPhysicalDeviceMaintenance7FeaturesKHR);
        maintenance7.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_7_FEATURES_KHR;
        var maintenance8 = std.mem.zeroes(c.VkPhysicalDeviceMaintenance8FeaturesKHR);
        maintenance8.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR;
        var maintenance9 = std.mem.zeroes(c.VkPhysicalDeviceMaintenance9FeaturesKHR);
        maintenance9.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_9_FEATURES_KHR;
        var uniform = std.mem.zeroes(c.VkPhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR);
        uniform.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_FEATURES_KHR;
        var reconvergence = std.mem.zeroes(c.VkPhysicalDeviceShaderMaximalReconvergenceFeaturesKHR);
        reconvergence.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MAXIMAL_RECONVERGENCE_FEATURES_KHR;
        var quad = std.mem.zeroes(c.VkPhysicalDeviceShaderQuadControlFeaturesKHR);
        quad.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_QUAD_CONTROL_FEATURES_KHR;
        var untyped = std.mem.zeroes(c.VkPhysicalDeviceShaderUntypedPointersFeaturesKHR);
        untyped.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR;
        var explicit_layout = std.mem.zeroes(c.VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR);
        explicit_layout.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR;
        var robust = std.mem.zeroes(c.VkPhysicalDeviceRobustness2FeaturesKHR);
        robust.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_KHR;
        var binary = std.mem.zeroes(c.VkPhysicalDevicePipelineBinaryFeaturesKHR);
        binary.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_BINARY_FEATURES_KHR;
        var clock = std.mem.zeroes(c.VkPhysicalDeviceShaderClockFeaturesKHR);
        clock.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR;
        var derivatives = std.mem.zeroes(c.VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR);
        derivatives.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR;
        f11.pNext = &f12;
        f12.pNext = &f13;
        f13.pNext = &f14;
        f14.pNext = &cooperative;
        cooperative.pNext = &maintenance7;
        maintenance7.pNext = &maintenance8;
        maintenance8.pNext = &maintenance9;
        maintenance9.pNext = &uniform;
        uniform.pNext = &reconvergence;
        reconvergence.pNext = &quad;
        quad.pNext = &untyped;
        untyped.pNext = &explicit_layout;
        explicit_layout.pNext = &robust;
        robust.pNext = &binary;
        binary.pNext = &clock;
        clock.pNext = &derivatives;
        const extensions = [_][*:0]const u8{ "VK_KHR_cooperative_matrix", "VK_KHR_maintenance5", "VK_KHR_maintenance7", "VK_KHR_maintenance8", "VK_KHR_maintenance9", "VK_KHR_shader_float_controls2", "VK_KHR_shader_expect_assume", "VK_KHR_shader_subgroup_rotate", "VK_KHR_shader_subgroup_uniform_control_flow", "VK_KHR_shader_maximal_reconvergence", "VK_KHR_shader_quad_control", "VK_KHR_shader_untyped_pointers", "VK_KHR_workgroup_memory_explicit_layout", "VK_KHR_robustness2", "VK_KHR_pipeline_binary", "VK_KHR_map_memory2", "VK_KHR_shader_clock", "VK_KHR_compute_shader_derivatives" };
        var features = std.mem.zeroes(c.VkPhysicalDeviceFeatures2);
        features.sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features.pNext = &f11;
        c.vkGetPhysicalDeviceFeatures2(self.physical, &features);
        if (features.features.shaderInt64 == 0 or f11.storageBuffer16BitAccess == 0 or f12.storageBuffer8BitAccess == 0 or f12.shaderFloat16 == 0 or f12.shaderInt8 == 0 or f12.bufferDeviceAddress == 0 or f12.scalarBlockLayout == 0 or f12.vulkanMemoryModel == 0 or f12.timelineSemaphore == 0 or f12.descriptorIndexing == 0 or f13.shaderIntegerDotProduct == 0 or f13.subgroupSizeControl == 0 or f12.shaderSubgroupExtendedTypes == 0 or f14.maintenance5 == 0 or f14.shaderFloatControls2 == 0 or f14.shaderExpectAssume == 0 or f14.shaderSubgroupRotate == 0 or cooperative.cooperativeMatrix == 0 or maintenance7.maintenance7 == 0 or maintenance8.maintenance8 == 0 or maintenance9.maintenance9 == 0 or uniform.shaderSubgroupUniformControlFlow == 0 or reconvergence.shaderMaximalReconvergence == 0 or quad.shaderQuadControl == 0 or untyped.shaderUntypedPointers == 0 or explicit_layout.workgroupMemoryExplicitLayout == 0 or robust.robustBufferAccess2 == 0 or robust.robustImageAccess2 == 0 or robust.nullDescriptor == 0 or binary.pipelineBinaries == 0 or clock.shaderDeviceClock == 0 or derivatives.computeDerivativeGroupQuads == 0) return error.RequiredComputeFeaturesUnavailable;
        // Query first: enabling only features advertised by the physical device.
        const priority: f32 = 1;
        var qi = std.mem.zeroes(c.VkDeviceQueueCreateInfo);
        qi.sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi.queueFamilyIndex = self.family;
        qi.queueCount = 1;
        qi.pQueuePriorities = &priority;
        var di = std.mem.zeroes(c.VkDeviceCreateInfo);
        di.sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        di.pNext = &features;
        di.queueCreateInfoCount = 1;
        di.pQueueCreateInfos = &qi;
        di.enabledExtensionCount = extensions.len;
        di.ppEnabledExtensionNames = &extensions;
        try check(c.vkCreateDevice(self.physical, &di, null, &self.device));
        c.vkGetDeviceQueue(self.device, self.family, 0, &self.queue);
        c.vkGetPhysicalDeviceMemoryProperties(self.physical, &self.mem);
        var pi = std.mem.zeroes(c.VkCommandPoolCreateInfo);
        pi.sType = c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pi.queueFamilyIndex = self.family;
        pi.flags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        try check(c.vkCreateCommandPool(self.device, &pi, null, &self.pool));
        var ai = std.mem.zeroes(c.VkCommandBufferAllocateInfo);
        ai.sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = self.pool;
        ai.level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        try check(c.vkAllocateCommandBuffers(self.device, &ai, &self.cmd));
        var fi = std.mem.zeroes(c.VkFenceCreateInfo);
        fi.sType = c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        try check(c.vkCreateFence(self.device, &fi, null, &self.fence));
        const range = c.VkPushConstantRange{ .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = 128 };
        var li = std.mem.zeroes(c.VkPipelineLayoutCreateInfo);
        li.sType = c.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        li.pushConstantRangeCount = 1;
        li.pPushConstantRanges = &range;
        try check(c.vkCreatePipelineLayout(self.device, &li, null, &self.layout));
        if (profiling) {
            var query = std.mem.zeroes(c.VkQueryPoolCreateInfo);
            query.sType = c.VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            query.queryType = c.VK_QUERY_TYPE_TIMESTAMP;
            query.queryCount = 16384;
            try check(c.vkCreateQueryPool(self.device, &query, null, &self.queries));
        }
        return self;
    }
    pub fn deinit(self: *Runtime) void {
        if (self.device != null) {
            _ = c.vkDeviceWaitIdle(self.device);
            if (self.queries != null) c.vkDestroyQueryPool(self.device, self.queries, null);
            if (self.layout != null) c.vkDestroyPipelineLayout(self.device, self.layout, null);
            if (self.fence != null) c.vkDestroyFence(self.device, self.fence, null);
            if (self.pool != null) c.vkDestroyCommandPool(self.device, self.pool, null);
            c.vkDestroyDevice(self.device, null);
        }
        if (self.instance != null) c.vkDestroyInstance(self.instance, null);
    }
    pub fn buffer(self: *Runtime, size: usize) !Buffer {
        var out: Buffer = .{ .size = size };
        errdefer self.freeBuffer(&out);
        var bi = std.mem.zeroes(c.VkBufferCreateInfo);
        bi.sType = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = size;
        bi.usage = c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | c.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | c.VK_BUFFER_USAGE_TRANSFER_DST_BIT | c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bi.sharingMode = c.VK_SHARING_MODE_EXCLUSIVE;
        try check(c.vkCreateBuffer(self.device, &bi, null, &out.handle));
        var req: c.VkMemoryRequirements = undefined;
        c.vkGetBufferMemoryRequirements(self.device, out.handle, &req);
        var ix: ?u32 = null;
        for (0..self.mem.memoryTypeCount) |i| {
            const flags = self.mem.memoryTypes[i].propertyFlags;
            if ((req.memoryTypeBits & (@as(u32, 1) << @intCast(i))) != 0 and (flags & (c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) == (c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                ix = @intCast(i);
                break;
            }
        }
        if (ix == null) for (0..self.mem.memoryTypeCount) |i| {
            const flags = self.mem.memoryTypes[i].propertyFlags;
            if ((req.memoryTypeBits & (@as(u32, 1) << @intCast(i))) != 0 and (flags & (c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) == (c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                ix = @intCast(i);
                break;
            }
        };
        var flags = std.mem.zeroes(c.VkMemoryAllocateFlagsInfo);
        flags.sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flags.flags = c.VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        var mi = std.mem.zeroes(c.VkMemoryAllocateInfo);
        mi.sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mi.pNext = &flags;
        mi.allocationSize = req.size;
        mi.memoryTypeIndex = ix orelse return error.NoMemoryType;
        try check(c.vkAllocateMemory(self.device, &mi, null, &out.memory));
        try check(c.vkBindBufferMemory(self.device, out.handle, out.memory, 0));
        var ptr: ?*anyopaque = null;
        try check(c.vkMapMemory(self.device, out.memory, 0, c.VK_WHOLE_SIZE, 0, &ptr));
        out.ptr = @ptrCast(ptr.?);
        out.mapped = true;
        var addr = std.mem.zeroes(c.VkBufferDeviceAddressInfo);
        addr.sType = c.VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addr.buffer = out.handle;
        out.addr = c.vkGetBufferDeviceAddress(self.device, &addr);
        return out;
    }
    pub fn freeBuffer(self: *Runtime, b: *Buffer) void {
        if (b.mapped) c.vkUnmapMemory(self.device, b.memory);
        if (b.handle != null) c.vkDestroyBuffer(self.device, b.handle, null);
        if (b.memory != null) c.vkFreeMemory(self.device, b.memory, null);
        b.* = .{};
    }
    pub fn pipeline(self: *Runtime, bytes: []align(4) const u8) !Pipeline {
        var sm = std.mem.zeroes(c.VkShaderModuleCreateInfo);
        sm.sType = c.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm.codeSize = bytes.len;
        sm.pCode = @ptrCast(bytes.ptr);
        var module: c.VkShaderModule = null;
        try check(c.vkCreateShaderModule(self.device, &sm, null, &module));
        defer c.vkDestroyShaderModule(self.device, module, null);
        var stage = std.mem.zeroes(c.VkPipelineShaderStageCreateInfo);
        stage.sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = c.VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = module;
        stage.pName = "main";
        var subgroup = std.mem.zeroes(c.VkPipelineShaderStageRequiredSubgroupSizeCreateInfo);
        subgroup.sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
        subgroup.requiredSubgroupSize = 32;
        stage.pNext = &subgroup;
        var ci = std.mem.zeroes(c.VkComputePipelineCreateInfo);
        ci.sType = c.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage = stage;
        ci.layout = self.layout;
        var out: c.VkPipeline = null;
        try check(c.vkCreateComputePipelines(self.device, null, 1, &ci, null, &out));
        return out;
    }
    pub fn begin(self: *Runtime) !void {
        try check(c.vkResetCommandBuffer(self.cmd, 0));
        var bi = std.mem.zeroes(c.VkCommandBufferBeginInfo);
        bi.sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        try check(c.vkBeginCommandBuffer(self.cmd, &bi));
        if (profiling) {
            c.vkCmdResetQueryPool(self.cmd, self.queries, 0, 16384);
            self.query_count = 0;
        }
    }
    pub fn dispatch(self: *Runtime, p: Pipeline, params: anytype, x: u32, y: u32, z: u32) void {
        c.vkCmdBindPipeline(self.cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, p);
        c.vkCmdPushConstants(self.cmd, self.layout, c.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(@TypeOf(params)), &params);
        if (profiling and self.query_count < 16384) c.vkCmdWriteTimestamp(self.cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, self.queries, self.query_count);
        c.vkCmdDispatch(self.cmd, x, y, z);
        if (profiling and self.query_count < 16384) {
            c.vkCmdWriteTimestamp(self.cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, self.queries, self.query_count + 1);
            self.profile_ids[self.query_count / 2] = self.profile_op;
            self.query_count += 2;
        }
        var b = std.mem.zeroes(c.VkMemoryBarrier);
        b.sType = c.VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        b.srcAccessMask = c.VK_ACCESS_SHADER_WRITE_BIT;
        b.dstAccessMask = c.VK_ACCESS_SHADER_READ_BIT | c.VK_ACCESS_SHADER_WRITE_BIT;
        c.vkCmdPipelineBarrier(self.cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, null, 0, null);
    }
    pub fn finish(self: *Runtime) !void {
        // Make shader results visible to mapped coherent host memory before the fence.
        var host = std.mem.zeroes(c.VkMemoryBarrier);
        host.sType = c.VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        host.srcAccessMask = c.VK_ACCESS_SHADER_WRITE_BIT;
        host.dstAccessMask = c.VK_ACCESS_HOST_READ_BIT;
        c.vkCmdPipelineBarrier(self.cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, null, 0, null);
        try check(c.vkEndCommandBuffer(self.cmd));
        try check(c.vkResetFences(self.device, 1, &self.fence));
        var si = std.mem.zeroes(c.VkSubmitInfo);
        si.sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &self.cmd;
        try check(c.vkQueueSubmit(self.queue, 1, &si, self.fence));
        try check(c.vkWaitForFences(self.device, 1, &self.fence, c.VK_TRUE, std.math.maxInt(u64)));
        if (profiling and self.query_count > 0) {
            var times: [16384]u64 = undefined;
            try check(c.vkGetQueryPoolResults(self.device, self.queries, 0, self.query_count, self.query_count * 8, &times, 8, c.VK_QUERY_RESULT_64_BIT | c.VK_QUERY_RESULT_WAIT_BIT));
            var sums: [23]f64 = @splat(0);
            var counts: [23]u32 = @splat(0);
            for (0..self.query_count / 2) |i| {
                const id = self.profile_ids[i];
                sums[id] += @as(f64, @floatFromInt(times[i * 2 + 1] - times[i * 2])) * self.timer_scale / 1000000;
                counts[id] += 1;
            }
            std.debug.print("GPU_PROFILE ms={any} counts={any}\n", .{ sums, counts });
        }
    }
};
pub fn check(result: c.VkResult) !void {
    if (result != c.VK_SUCCESS) return error.VulkanFailure;
}
