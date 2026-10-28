#!/bin/sh
cd /workspace
mkdir -p shaders
for f in shaders/*.comp; do
  /opt/vulkan-sdk/bin/glslc --target-env=vulkan1.3 -O -o "${f%.comp}.spv" "$f" || exit 1
done
ls -la shaders/*.spv
