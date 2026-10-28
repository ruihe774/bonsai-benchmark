from pathlib import Path
import shutil,subprocess
root=Path.cwd();dest=root/'tools'/'clean_build_check'
if dest.exists():raise RuntimeError('clean build directory already exists')
dest.mkdir();shutil.copyfile(root/'build.zig',dest/'build.zig');shutil.copytree(root/'src',dest/'src');shutil.copytree(root/'shaders',dest/'shaders')
# This new directory contains no data/, docs/, ref/, or model files.
result=subprocess.run(['zig','build','-Doptimize=ReleaseFast'],cwd=dest)
assert result.returncode==0,'clean source-only build failed'
assert (dest/'zig-out/lib/libbonsai.so').is_file()
print('Clean source-only offline build passed without data/, docs/, ref/')
