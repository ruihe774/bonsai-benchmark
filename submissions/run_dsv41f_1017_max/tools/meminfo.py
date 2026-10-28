import os
for line in open('/proc/meminfo').read().splitlines()[:3]:
    print(line)
print("cpus", os.cpu_count())
