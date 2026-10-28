#!/bin/sh
echo "--- vulkaninfo ---"
which vulkaninfo glslc glslangValidator spirv-as 2>&1
echo "--- dri ---"
ls /dev/dri 2>&1
echo "--- headers ---"
ls /usr/include/vulkan 2>&1 | head -5
echo "--- SDK ---"
ls /usr/local 2>&1
echo "--- ldconfig ---"
ldconfig -p 2>/dev/null | grep -i vulkan
