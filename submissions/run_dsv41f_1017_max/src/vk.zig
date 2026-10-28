// Minimal hand-written Vulkan 1.4 compute bindings (loader based, no headers).
const std = @import("std");

pub const VkResult = i32;
pub const VK_SUCCESS: VkResult = 0;
pub const VkFlags = u32;
pub const VkBool32 = u32;
pub const VkDeviceSize = u64;

pub const VkInstance = ?*anyopaque;
pub const VkPhysicalDevice = ?*anyopaque;
pub const VkDevice = ?*anyopaque;
pub const VkQueue = ?*anyopaque;
pub const VkCommandBuffer = ?*anyopaque;
pub const VkBuffer = u64;
pub const VkDeviceMemory = u64;
pub const VkShaderModule = u64;
pub const VkPipeline = u64;
pub const VkPipelineLayout = u64;
pub const VkDescriptorSetLayout = u64;
pub const VkDescriptorPool = u64;
pub const VkDescriptorSet = u64;
pub const VkCommandPool = u64;
pub const VkFence = u64;

pub const VK_STRUCTURE_TYPE_APPLICATION_INFO = 0;
pub const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1;
pub const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2;
pub const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3;
pub const VK_STRUCTURE_TYPE_SUBMIT_INFO = 4;
pub const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5;
pub const VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 8;
pub const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12;
pub const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16;
pub const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18;
pub const VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29;
pub const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34;
pub const VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35;
pub const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39;
pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40;
pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42;
pub const VK_STRUCTURE_TYPE_MEMORY_BARRIER = 46;

pub const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: u32 = 0x20;
pub const VK_BUFFER_USAGE_TRANSFER_SRC_BIT: u32 = 0x1;
pub const VK_BUFFER_USAGE_TRANSFER_DST_BIT: u32 = 0x2;
pub const VK_SHARING_MODE_EXCLUSIVE: u32 = 0;
pub const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: u32 = 0x1;
pub const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: u32 = 0x2;
pub const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: u32 = 0x4;
pub const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: u32 = 7;
pub const VK_SHADER_STAGE_COMPUTE_BIT: u32 = 0x20;
pub const VK_PIPELINE_BIND_POINT_COMPUTE: u32 = 1;
pub const VK_COMMAND_BUFFER_LEVEL_PRIMARY: u32 = 0;
pub const VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: u32 = 0x2;
pub const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: u32 = 0x1;
pub const VK_QUEUE_COMPUTE_BIT: u32 = 0x2;
pub const VK_ACCESS_SHADER_READ_BIT: u32 = 0x20;
pub const VK_ACCESS_SHADER_WRITE_BIT: u32 = 0x40;
pub const VK_ACCESS_TRANSFER_WRITE_BIT: u32 = 0x1000;
pub const VK_ACCESS_TRANSFER_READ_BIT: u32 = 0x800;
pub const VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT: u32 = 0x800;
pub const VK_PIPELINE_STAGE_TRANSFER_BIT: u32 = 0x1000;
pub const VK_PIPELINE_STAGE_ALL_COMMANDS_BIT: u32 = 0x10000;

pub const VkApplicationInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    pNext: ?*const anyopaque = null,
    pApplicationName: ?[*:0]const u8 = null,
    applicationVersion: u32 = 0,
    pEngineName: ?[*:0]const u8 = null,
    engineVersion: u32 = 0,
    apiVersion: u32 = 0,
};
pub const VkInstanceCreateInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    pApplicationInfo: ?*const VkApplicationInfo = null,
    enabledLayerCount: u32 = 0,
    ppEnabledLayerNames: ?*const anyopaque = null,
    enabledExtensionCount: u32 = 0,
    ppEnabledExtensionNames: ?*const anyopaque = null,
};
pub const VkQueueFamilyProperties = extern struct {
    queueFlags: u32,
    queueCount: u32,
    timestampValidBits: u32,
    minImageTransferGranularity: [3]u32,
};
pub const VkDeviceQueueCreateInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    queueFamilyIndex: u32 = 0,
    queueCount: u32 = 1,
    pQueuePriorities: ?*const f32 = null,
};
pub const VkDeviceCreateInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    queueCreateInfoCount: u32 = 0,
    pQueueCreateInfos: ?*const VkDeviceQueueCreateInfo = null,
    enabledLayerCount: u32 = 0,
    ppEnabledLayerNames: ?*const anyopaque = null,
    enabledExtensionCount: u32 = 0,
    ppEnabledExtensionNames: ?*const anyopaque = null,
    pEnabledFeatures: ?*const anyopaque = null,
};
pub const VkMemoryType = extern struct { propertyFlags: u32, heapIndex: u32 };
pub const VkMemoryHeap = extern struct { size: u64, flags: u32 };
pub const VkPhysicalDeviceMemoryProperties = extern struct {
    memoryTypeCount: u32,
    memoryTypes: [32]VkMemoryType,
    memoryHeapCount: u32,
    memoryHeaps: [16]VkMemoryHeap,
};
pub const VkBufferCreateInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    size: u64 = 0,
    usage: u32 = 0,
    sharingMode: u32 = VK_SHARING_MODE_EXCLUSIVE,
    queueFamilyIndexCount: u32 = 0,
    pQueueFamilyIndices: ?*const u32 = null,
};
pub const VkMemoryRequirements = extern struct { size: u64, alignment: u64, memoryTypeBits: u32 };
pub const VkMemoryAllocateInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    pNext: ?*const anyopaque = null,
    allocationSize: u64 = 0,
    memoryTypeIndex: u32 = 0,
};
pub const VkShaderModuleCreateInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    codeSize: usize = 0,
    pCode: ?[*]const u32 = null,
};
pub const VkDescriptorSetLayoutBinding = extern struct {
    binding: u32,
    descriptorType: u32,
    descriptorCount: u32,
    stageFlags: u32,
    pImmutableSamplers: ?*const anyopaque = null,
};
pub const VkDescriptorSetLayoutCreateInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    bindingCount: u32 = 0,
    pBindings: ?[*]const VkDescriptorSetLayoutBinding = null,
};
pub const VkPushConstantRange = extern struct { stageFlags: u32, offset: u32, size: u32 };
pub const VkPipelineLayoutCreateInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    setLayoutCount: u32 = 0,
    pSetLayouts: ?[*]const VkDescriptorSetLayout = null,
    pushConstantRangeCount: u32 = 0,
    pPushConstantRanges: ?[*]const VkPushConstantRange = null,
};
pub const VkPipelineShaderStageCreateInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    stage: u32 = VK_SHADER_STAGE_COMPUTE_BIT,
    module: VkShaderModule = 0,
    pName: ?[*:0]const u8 = null,
    pSpecializationInfo: ?*const anyopaque = null,
};
pub const VkComputePipelineCreateInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    stage: VkPipelineShaderStageCreateInfo = .{},
    layout: VkPipelineLayout = 0,
    basePipelineHandle: VkPipeline = 0,
    basePipelineIndex: i32 = -1,
};
pub const VkDescriptorPoolSize = extern struct { type: u32, descriptorCount: u32 };
pub const VkDescriptorPoolCreateInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    maxSets: u32 = 0,
    poolSizeCount: u32 = 0,
    pPoolSizes: ?[*]const VkDescriptorPoolSize = null,
};
pub const VkDescriptorSetAllocateInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    pNext: ?*const anyopaque = null,
    descriptorPool: VkDescriptorPool = 0,
    descriptorSetCount: u32 = 0,
    pSetLayouts: ?[*]const VkDescriptorSetLayout = null,
};
pub const VkDescriptorBufferInfo = extern struct { buffer: VkBuffer = 0, offset: u64 = 0, range: u64 = 0 };
pub const VkWriteDescriptorSet = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    pNext: ?*const anyopaque = null,
    dstSet: VkDescriptorSet = 0,
    dstBinding: u32 = 0,
    dstArrayElement: u32 = 0,
    descriptorCount: u32 = 1,
    descriptorType: u32 = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    pImageInfo: ?*const anyopaque = null,
    pBufferInfo: ?*const VkDescriptorBufferInfo = null,
    pTexelBufferView: ?*const anyopaque = null,
};
pub const VkCommandPoolCreateInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    queueFamilyIndex: u32 = 0,
};
pub const VkCommandBufferAllocateInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    pNext: ?*const anyopaque = null,
    commandPool: VkCommandPool = 0,
    level: u32 = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    commandBufferCount: u32 = 1,
};
pub const VkCommandBufferBeginInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    pInheritanceInfo: ?*const anyopaque = null,
};
pub const VkSubmitInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    pNext: ?*const anyopaque = null,
    waitSemaphoreCount: u32 = 0,
    pWaitSemaphores: ?*const anyopaque = null,
    pWaitDstStageMask: ?*const anyopaque = null,
    commandBufferCount: u32 = 0,
    pCommandBuffers: ?[*]const VkCommandBuffer = null,
    signalSemaphoreCount: u32 = 0,
    pSignalSemaphores: ?*const anyopaque = null,
};
pub const VkFenceCreateInfo = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
};
pub const VkMemoryBarrier = extern struct {
    sType: u32 = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    pNext: ?*const anyopaque = null,
    srcAccessMask: u32 = 0,
    dstAccessMask: u32 = 0,
};
pub const VkBufferCopy = extern struct { srcOffset: u64 = 0, dstOffset: u64 = 0, size: u64 = 0 };

pub const FnCreateInstance = *const fn (*const VkInstanceCreateInfo, ?*const anyopaque, *VkInstance) callconv(.c) VkResult;
pub const FnGetInstanceProcAddr = *const fn (VkInstance, [*:0]const u8) callconv(.c) ?*const anyopaque;
pub const FnEnumeratePhysicalDevices = *const fn (VkInstance, *u32, ?[*]VkPhysicalDevice) callconv(.c) VkResult;
pub const FnGetPhysicalDeviceProperties = *const fn (VkPhysicalDevice, *anyopaque) callconv(.c) void;
pub const FnGetPhysicalDeviceMemoryProperties = *const fn (VkPhysicalDevice, *VkPhysicalDeviceMemoryProperties) callconv(.c) void;
pub const FnGetPhysicalDeviceQueueFamilyProperties = *const fn (VkPhysicalDevice, *u32, ?[*]VkQueueFamilyProperties) callconv(.c) void;
pub const FnCreateDevice = *const fn (VkPhysicalDevice, *const VkDeviceCreateInfo, ?*const anyopaque, *VkDevice) callconv(.c) VkResult;
pub const FnGetDeviceProcAddr = *const fn (VkDevice, [*:0]const u8) callconv(.c) ?*const anyopaque;
pub const FnGetDeviceQueue = *const fn (VkDevice, u32, u32, *VkQueue) callconv(.c) void;
pub const FnCreateBuffer = *const fn (VkDevice, *const VkBufferCreateInfo, ?*const anyopaque, *VkBuffer) callconv(.c) VkResult;
pub const FnGetBufferMemoryRequirements = *const fn (VkDevice, VkBuffer, *VkMemoryRequirements) callconv(.c) void;
pub const FnAllocateMemory = *const fn (VkDevice, *const VkMemoryAllocateInfo, ?*const anyopaque, *VkDeviceMemory) callconv(.c) VkResult;
pub const FnBindBufferMemory = *const fn (VkDevice, VkBuffer, VkDeviceMemory, u64) callconv(.c) VkResult;
pub const FnMapMemory = *const fn (VkDevice, VkDeviceMemory, u64, u64, u32, *?*anyopaque) callconv(.c) VkResult;
pub const FnUnmapMemory = *const fn (VkDevice, VkDeviceMemory) callconv(.c) void;
pub const FnCreateShaderModule = *const fn (VkDevice, *const VkShaderModuleCreateInfo, ?*const anyopaque, *VkShaderModule) callconv(.c) VkResult;
pub const FnCreateDescriptorSetLayout = *const fn (VkDevice, *const VkDescriptorSetLayoutCreateInfo, ?*const anyopaque, *VkDescriptorSetLayout) callconv(.c) VkResult;
pub const FnCreatePipelineLayout = *const fn (VkDevice, *const VkPipelineLayoutCreateInfo, ?*const anyopaque, *VkPipelineLayout) callconv(.c) VkResult;
pub const FnCreateComputePipelines = *const fn (VkDevice, u64, u32, [*]const VkComputePipelineCreateInfo, ?*const anyopaque, [*]VkPipeline) callconv(.c) VkResult;
pub const FnCreateDescriptorPool = *const fn (VkDevice, *const VkDescriptorPoolCreateInfo, ?*const anyopaque, *VkDescriptorPool) callconv(.c) VkResult;
pub const FnAllocateDescriptorSets = *const fn (VkDevice, *const VkDescriptorSetAllocateInfo, [*]VkDescriptorSet) callconv(.c) VkResult;
pub const FnResetDescriptorPool = *const fn (VkDevice, VkDescriptorPool, u32) callconv(.c) VkResult;
pub const FnUpdateDescriptorSets = *const fn (VkDevice, u32, [*]const VkWriteDescriptorSet, u32, ?*const anyopaque) callconv(.c) void;
pub const FnCreateCommandPool = *const fn (VkDevice, *const VkCommandPoolCreateInfo, ?*const anyopaque, *VkCommandPool) callconv(.c) VkResult;
pub const FnAllocateCommandBuffers = *const fn (VkDevice, *const VkCommandBufferAllocateInfo, [*]VkCommandBuffer) callconv(.c) VkResult;
pub const FnBeginCommandBuffer = *const fn (VkCommandBuffer, *const VkCommandBufferBeginInfo) callconv(.c) VkResult;
pub const FnEndCommandBuffer = *const fn (VkCommandBuffer) callconv(.c) VkResult;
pub const FnCmdBindPipeline = *const fn (VkCommandBuffer, u32, VkPipeline) callconv(.c) void;
pub const FnCmdBindDescriptorSets = *const fn (VkCommandBuffer, u32, VkPipelineLayout, u32, u32, [*]const VkDescriptorSet, u32, ?[*]const u32) callconv(.c) void;
pub const FnCmdPushConstants = *const fn (VkCommandBuffer, VkPipelineLayout, u32, u32, u32, *const anyopaque) callconv(.c) void;
pub const FnCmdDispatch = *const fn (VkCommandBuffer, u32, u32, u32) callconv(.c) void;
pub const FnCmdPipelineBarrier = *const fn (VkCommandBuffer, u32, u32, u32, u32, ?[*]const VkMemoryBarrier, u32, ?*const anyopaque, u32, ?*const anyopaque) callconv(.c) void;
pub const FnCmdCopyBuffer = *const fn (VkCommandBuffer, VkBuffer, VkBuffer, u32, [*]const VkBufferCopy) callconv(.c) void;
pub const FnCmdFillBuffer = *const fn (VkCommandBuffer, VkBuffer, u64, u64, u32) callconv(.c) void;
pub const FnQueueSubmit = *const fn (VkQueue, u32, ?[*]const VkSubmitInfo, VkFence) callconv(.c) VkResult;
pub const FnQueueWaitIdle = *const fn (VkQueue) callconv(.c) VkResult;
pub const FnCreateFence = *const fn (VkDevice, *const VkFenceCreateInfo, ?*const anyopaque, *VkFence) callconv(.c) VkResult;
pub const FnWaitForFences = *const fn (VkDevice, u32, [*]const VkFence, u32, u64) callconv(.c) VkResult;
pub const FnResetFences = *const fn (VkDevice, u32, [*]const VkFence) callconv(.c) VkResult;
pub const FnDestroyBuffer = *const fn (VkDevice, VkBuffer, ?*const anyopaque) callconv(.c) void;
pub const FnFreeMemory = *const fn (VkDevice, VkDeviceMemory, ?*const anyopaque) callconv(.c) void;
pub const FnDestroyShaderModule = *const fn (VkDevice, VkShaderModule, ?*const anyopaque) callconv(.c) void;
pub const FnDestroyPipeline = *const fn (VkDevice, VkPipeline, ?*const anyopaque) callconv(.c) void;
pub const FnDestroyPipelineLayout = *const fn (VkDevice, VkPipelineLayout, ?*const anyopaque) callconv(.c) void;
pub const FnDestroyDescriptorSetLayout = *const fn (VkDevice, VkDescriptorSetLayout, ?*const anyopaque) callconv(.c) void;
pub const FnDestroyDescriptorPool = *const fn (VkDevice, VkDescriptorPool, ?*const anyopaque) callconv(.c) void;
pub const FnDestroyCommandPool = *const fn (VkDevice, VkCommandPool, ?*const anyopaque) callconv(.c) void;
pub const FnDeviceWaitIdle = *const fn (VkDevice) callconv(.c) VkResult;
pub const FnDestroyDevice = *const fn (VkDevice, ?*const anyopaque) callconv(.c) void;
pub const FnDestroyInstance = *const fn (VkInstance, ?*const anyopaque) callconv(.c) void;

pub const Api = struct {
    create_instance: FnCreateInstance = undefined,
    get_instance_proc_addr: FnGetInstanceProcAddr = undefined,
    enumerate_physical_devices: FnEnumeratePhysicalDevices = undefined,
    get_physical_device_properties: FnGetPhysicalDeviceProperties = undefined,
    get_physical_device_memory_properties: FnGetPhysicalDeviceMemoryProperties = undefined,
    get_physical_device_queue_family_properties: FnGetPhysicalDeviceQueueFamilyProperties = undefined,
    create_device: FnCreateDevice = undefined,
    get_device_proc_addr: FnGetDeviceProcAddr = undefined,
    get_device_queue: FnGetDeviceQueue = undefined,
    create_buffer: FnCreateBuffer = undefined,
    get_buffer_memory_requirements: FnGetBufferMemoryRequirements = undefined,
    allocate_memory: FnAllocateMemory = undefined,
    bind_buffer_memory: FnBindBufferMemory = undefined,
    map_memory: FnMapMemory = undefined,
    unmap_memory: FnUnmapMemory = undefined,
    create_shader_module: FnCreateShaderModule = undefined,
    create_descriptor_set_layout: FnCreateDescriptorSetLayout = undefined,
    create_pipeline_layout: FnCreatePipelineLayout = undefined,
    create_compute_pipelines: FnCreateComputePipelines = undefined,
    create_descriptor_pool: FnCreateDescriptorPool = undefined,
    allocate_descriptor_sets: FnAllocateDescriptorSets = undefined,
    reset_descriptor_pool: FnResetDescriptorPool = undefined,
    update_descriptor_sets: FnUpdateDescriptorSets = undefined,
    create_command_pool: FnCreateCommandPool = undefined,
    allocate_command_buffers: FnAllocateCommandBuffers = undefined,
    begin_command_buffer: FnBeginCommandBuffer = undefined,
    end_command_buffer: FnEndCommandBuffer = undefined,
    cmd_bind_pipeline: FnCmdBindPipeline = undefined,
    cmd_bind_descriptor_sets: FnCmdBindDescriptorSets = undefined,
    cmd_push_constants: FnCmdPushConstants = undefined,
    cmd_dispatch: FnCmdDispatch = undefined,
    cmd_pipeline_barrier: FnCmdPipelineBarrier = undefined,
    cmd_copy_buffer: FnCmdCopyBuffer = undefined,
    cmd_fill_buffer: FnCmdFillBuffer = undefined,
    queue_submit: FnQueueSubmit = undefined,
    queue_wait_idle: FnQueueWaitIdle = undefined,
    create_fence: FnCreateFence = undefined,
    wait_for_fences: FnWaitForFences = undefined,
    reset_fences: FnResetFences = undefined,
    destroy_buffer: FnDestroyBuffer = undefined,
    free_memory: FnFreeMemory = undefined,
    destroy_shader_module: FnDestroyShaderModule = undefined,
    destroy_pipeline: FnDestroyPipeline = undefined,
    destroy_pipeline_layout: FnDestroyPipelineLayout = undefined,
    destroy_descriptor_set_layout: FnDestroyDescriptorSetLayout = undefined,
    destroy_descriptor_pool: FnDestroyDescriptorPool = undefined,
    destroy_command_pool: FnDestroyCommandPool = undefined,
    device_wait_idle: FnDeviceWaitIdle = undefined,
    destroy_device: FnDestroyDevice = undefined,
    destroy_instance: FnDestroyInstance = undefined,
};

pub const Buffer = struct {
    buf: VkBuffer = 0,
    mem: VkDeviceMemory = 0,
    size: u64 = 0,
};

pub const Ctx = struct {
    api: Api = .{},
    instance: VkInstance = null,
    phys: VkPhysicalDevice = null,
    device: VkDevice = null,
    queue: VkQueue = null,
    qfam: u32 = 0,
    memprops: VkPhysicalDeviceMemoryProperties = undefined,
    cmdpool: VkCommandPool = 0,
    cmd: VkCommandBuffer = null,
    dsl: VkDescriptorSetLayout = 0,
    pl: VkPipelineLayout = 0,
    pool: VkDescriptorPool = 0,
    err: [512]u8 = undefined, // unused
};

fn loadPtr(comptime T: type, p: ?*const anyopaque) T {
    return @as(T, @ptrCast(@alignCast(p.?)));
}

pub fn init(c: *Ctx) !void {
    var dl = try std.DynLib.open("libvulkan.so.1");
    const create_instance_p = dl.lookup(*const anyopaque, "vkCreateInstance") orelse return error.NoVulkan;
    const get_proc_p = dl.lookup(*const anyopaque, "vkGetInstanceProcAddr") orelse return error.NoVulkan;
    c.api.create_instance = loadPtr(FnCreateInstance, create_instance_p);
    c.api.get_instance_proc_addr = loadPtr(FnGetInstanceProcAddr, get_proc_p);

    var app = VkApplicationInfo{ .apiVersion = (1 << 22) | (4 << 12) };
    var ci = VkInstanceCreateInfo{ .pApplicationInfo = &app };
    var inst: VkInstance = null;
    if (c.api.create_instance(&ci, null, &inst) != VK_SUCCESS) return error.InstanceFailed;
    c.instance = inst;

    inline for (.{
        .{ "enumerate_physical_devices", "vkEnumeratePhysicalDevices" },
        .{ "get_physical_device_properties", "vkGetPhysicalDeviceProperties" },
        .{ "get_physical_device_memory_properties", "vkGetPhysicalDeviceMemoryProperties" },
        .{ "get_physical_device_queue_family_properties", "vkGetPhysicalDeviceQueueFamilyProperties" },
        .{ "create_device", "vkCreateDevice" },
        .{ "destroy_instance", "vkDestroyInstance" },
    }) |pair| {
        const p = c.api.get_instance_proc_addr(inst, pair[1]) orelse return error.NoProc;
        @field(c.api, pair[0]) = loadPtr(@TypeOf(@field(c.api, pair[0])), p);
    }

    var ndev: u32 = 0;
    if (c.api.enumerate_physical_devices(inst, &ndev, null) != VK_SUCCESS or ndev == 0) return error.NoDevice;
    var devs: [8]VkPhysicalDevice = undefined;
    if (ndev > 8) ndev = 8;
    _ = c.api.enumerate_physical_devices(inst, &ndev, &devs);
    c.phys = devs[0];
    c.api.get_physical_device_memory_properties(c.phys, &c.memprops);

    var nq: u32 = 0;
    c.api.get_physical_device_queue_family_properties(c.phys, &nq, null);
    var qf: [32]VkQueueFamilyProperties = undefined;
    if (nq > 32) nq = 32;
    c.api.get_physical_device_queue_family_properties(c.phys, &nq, &qf);
    var found = false;
    for (0..nq) |i| {
        if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
            c.qfam = @intCast(i);
            found = true;
            break;
        }
    }
    if (!found) return error.NoQueue;

    var prio: f32 = 1.0;
    var qci = VkDeviceQueueCreateInfo{ .queueFamilyIndex = c.qfam, .queueCount = 1, .pQueuePriorities = &prio };
    var dci = VkDeviceCreateInfo{ .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci };
    var dev: VkDevice = null;
    if (c.api.create_device(c.phys, &dci, null, &dev) != VK_SUCCESS) return error.DeviceFailed;
    c.device = dev;

    inline for (.{
        .{ "get_device_proc_addr", "vkGetDeviceProcAddr" },
    }) |pair| {
        const p = c.api.get_instance_proc_addr(inst, pair[1]) orelse return error.NoProc;
        @field(c.api, pair[0]) = loadPtr(@TypeOf(@field(c.api, pair[0])), p);
    }
    const gdpa = c.api.get_device_proc_addr;
    inline for (.{
        .{ "get_device_queue", "vkGetDeviceQueue" },
        .{ "create_buffer", "vkCreateBuffer" },
        .{ "get_buffer_memory_requirements", "vkGetBufferMemoryRequirements" },
        .{ "allocate_memory", "vkAllocateMemory" },
        .{ "bind_buffer_memory", "vkBindBufferMemory" },
        .{ "map_memory", "vkMapMemory" },
        .{ "unmap_memory", "vkUnmapMemory" },
        .{ "create_shader_module", "vkCreateShaderModule" },
        .{ "create_descriptor_set_layout", "vkCreateDescriptorSetLayout" },
        .{ "create_pipeline_layout", "vkCreatePipelineLayout" },
        .{ "create_compute_pipelines", "vkCreateComputePipelines" },
        .{ "create_descriptor_pool", "vkCreateDescriptorPool" },
        .{ "allocate_descriptor_sets", "vkAllocateDescriptorSets" },
        .{ "reset_descriptor_pool", "vkResetDescriptorPool" },
        .{ "update_descriptor_sets", "vkUpdateDescriptorSets" },
        .{ "create_command_pool", "vkCreateCommandPool" },
        .{ "allocate_command_buffers", "vkAllocateCommandBuffers" },
        .{ "begin_command_buffer", "vkBeginCommandBuffer" },
        .{ "end_command_buffer", "vkEndCommandBuffer" },
        .{ "cmd_bind_pipeline", "vkCmdBindPipeline" },
        .{ "cmd_bind_descriptor_sets", "vkCmdBindDescriptorSets" },
        .{ "cmd_push_constants", "vkCmdPushConstants" },
        .{ "cmd_dispatch", "vkCmdDispatch" },
        .{ "cmd_pipeline_barrier", "vkCmdPipelineBarrier" },
        .{ "cmd_copy_buffer", "vkCmdCopyBuffer" },
        .{ "cmd_fill_buffer", "vkCmdFillBuffer" },
        .{ "queue_submit", "vkQueueSubmit" },
        .{ "queue_wait_idle", "vkQueueWaitIdle" },
        .{ "create_fence", "vkCreateFence" },
        .{ "wait_for_fences", "vkWaitForFences" },
        .{ "reset_fences", "vkResetFences" },
        .{ "destroy_buffer", "vkDestroyBuffer" },
        .{ "free_memory", "vkFreeMemory" },
        .{ "destroy_shader_module", "vkDestroyShaderModule" },
        .{ "destroy_pipeline", "vkDestroyPipeline" },
        .{ "destroy_pipeline_layout", "vkDestroyPipelineLayout" },
        .{ "destroy_descriptor_set_layout", "vkDestroyDescriptorSetLayout" },
        .{ "destroy_descriptor_pool", "vkDestroyDescriptorPool" },
        .{ "destroy_command_pool", "vkDestroyCommandPool" },
        .{ "device_wait_idle", "vkDeviceWaitIdle" },
        .{ "destroy_device", "vkDestroyDevice" },
    }) |pair| {
        const p = gdpa(dev, pair[1]) orelse return error.NoProc;
        @field(c.api, pair[0]) = loadPtr(@TypeOf(@field(c.api, pair[0])), p);
    }

    c.api.get_device_queue(dev, c.qfam, 0, &c.queue);

    var cpci = VkCommandPoolCreateInfo{ .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = c.qfam };
    var pool: VkCommandPool = 0;
    if (c.api.create_command_pool(dev, &cpci, null, &pool) != VK_SUCCESS) return error.PoolFailed;
    c.cmdpool = pool;
    var cba = VkCommandBufferAllocateInfo{ .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
    var cmd: VkCommandBuffer = null;
    if (c.api.allocate_command_buffers(dev, &cba, @ptrCast(&cmd)) != VK_SUCCESS) return error.CmdFailed;
    c.cmd = cmd;

    // descriptor layout: 8 storage buffers
    var binds: [8]VkDescriptorSetLayoutBinding = undefined;
    for (0..8) |i| {
        binds[i] = .{ .binding = @intCast(i), .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .pImmutableSamplers = null };
    }
    var dslci = VkDescriptorSetLayoutCreateInfo{ .bindingCount = 8, .pBindings = &binds };
    var dsl: VkDescriptorSetLayout = 0;
    if (c.api.create_descriptor_set_layout(dev, &dslci, null, &dsl) != VK_SUCCESS) return error.DslFailed;
    c.dsl = dsl;
    const pcr = VkPushConstantRange{ .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = 64 };
    var dsl_arr: [1]VkDescriptorSetLayout = .{dsl};
    var pcr_arr: [1]VkPushConstantRange = .{pcr};
    var plci = VkPipelineLayoutCreateInfo{ .setLayoutCount = 1, .pSetLayouts = &dsl_arr, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr_arr };
    var pl: VkPipelineLayout = 0;
    if (c.api.create_pipeline_layout(dev, &plci, null, &pl) != VK_SUCCESS) return error.PlFailed;
    c.pl = pl;

    var psize_arr: [1]VkDescriptorPoolSize = .{.{ .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 8 * 2048 }};
    var dpci = VkDescriptorPoolCreateInfo{ .maxSets = 2048, .poolSizeCount = 1, .pPoolSizes = &psize_arr };
    var dpool: VkDescriptorPool = 0;
    if (c.api.create_descriptor_pool(dev, &dpci, null, &dpool) != VK_SUCCESS) return error.DpFailed;
    c.pool = dpool;
}

fn memTypeIndex(c: *Ctx, bits: u32, props: u32) ?u32 {
    const m = c.memprops;
    var i: u32 = 0;
    while (i < m.memoryTypeCount) : (i += 1) {
        if ((bits & (@as(u32, 1) << @intCast(i))) != 0 and (m.memoryTypes[i].propertyFlags & props) == props) return i;
    }
    return null;
}

pub fn createBuffer(c: *Ctx, size: u64, usage: u32, props: u32) !Buffer {
    var bci = VkBufferCreateInfo{ .size = size, .usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
    var buf: VkBuffer = 0;
    if (c.api.create_buffer(c.device, &bci, null, &buf) != VK_SUCCESS) return error.BufferFailed;
    var req: VkMemoryRequirements = undefined;
    c.api.get_buffer_memory_requirements(c.device, buf, &req);
    const mt = memTypeIndex(c, req.memoryTypeBits, props) orelse return error.NoMemType;
    var mai = VkMemoryAllocateInfo{ .allocationSize = req.size, .memoryTypeIndex = mt };
    var mem: VkDeviceMemory = 0;
    if (c.api.allocate_memory(c.device, &mai, null, &mem) != VK_SUCCESS) return error.AllocFailed;
    if (c.api.bind_buffer_memory(c.device, buf, mem, 0) != VK_SUCCESS) return error.BindFailed;
    return .{ .buf = buf, .mem = mem, .size = size };
}

pub fn mapBuffer(c: *Ctx, b: Buffer) ![]u8 {
    var p: ?*anyopaque = null;
    if (c.api.map_memory(c.device, b.mem, 0, b.size, 0, &p) != VK_SUCCESS) return error.MapFailed;
    return @as([*]u8, @ptrCast(p.?))[0..b.size];
}

pub fn beginCommands(c: *Ctx) !void {
    var bi = VkCommandBufferBeginInfo{ .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    if (c.api.begin_command_buffer(c.cmd, &bi) != VK_SUCCESS) return error.BeginFailed;
}

pub fn endSubmitWait(c: *Ctx) !void {
    if (c.api.end_command_buffer(c.cmd) != VK_SUCCESS) return error.EndFailed;
    var si_arr: [1]VkSubmitInfo = .{.{ .commandBufferCount = 1, .pCommandBuffers = @ptrCast(&c.cmd) }};
    if (c.api.queue_submit(c.queue, 1, &si_arr, 0) != VK_SUCCESS) return error.SubmitFailed;
    if (c.api.queue_wait_idle(c.queue) != VK_SUCCESS) return error.WaitFailed;
}

pub fn barrier(c: *Ctx) void {
    var mb = VkMemoryBarrier{ .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
    c.api.cmd_pipeline_barrier(c.cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, @ptrCast(&mb), 0, null, 0, null);
}

pub fn copyBuffer(c: *Ctx, src: Buffer, dst: Buffer, dstOff: u64, size: u64) void {
    copyBufferO(c, src, 0, dst, dstOff, size);
}
pub fn copyBufferO(c: *Ctx, src: Buffer, srcOff: u64, dst: Buffer, dstOff: u64, size: u64) void {
    var bc = VkBufferCopy{ .srcOffset = srcOff, .dstOffset = dstOff, .size = size };
    c.api.cmd_copy_buffer(c.cmd, src.buf, dst.buf, 1, @ptrCast(&bc));
}

pub fn fillBuffer(c: *Ctx, dst: Buffer, off: u64, size: u64, val: u32) void {
    c.api.cmd_fill_buffer(c.cmd, dst.buf, off, size, val);
}
