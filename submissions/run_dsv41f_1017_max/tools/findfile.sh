#!/bin/sh
grep -rn "pub const File" /opt/zig/lib/std/fs.zig /opt/zig/lib/std/Io.zig /opt/zig/lib/std/Io/Dir.zig 2>/dev/null | head -10
echo ---
grep -rn "pub const File" /opt/zig/lib/std/fs/ | head -10
echo ---
ls /opt/zig/lib/std | head -40
