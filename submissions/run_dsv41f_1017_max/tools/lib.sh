#!/bin/sh
echo "--- sdk lib ---"
ls /opt/vulkan-sdk/lib
echo "--- system libvulkan ---"
ls /usr/lib/x86_64-linux-gnu/libvulkan* 2>&1
ls /usr/lib64/libvulkan* 2>&1
echo "--- headers ---"
ls /opt/vulkan-sdk/include/vulkan | head
echo "--- device ---"
/opt/vulkan-sdk/bin/vulkaninfo --summary 2>&1 | head -40
