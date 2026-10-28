import os,subprocess,sys
if len(sys.argv)==1:
 env=os.environ.copy()
 for name in ['LD_LIBRARY_PATH','VULKAN_SDK','VK_ICD_FILENAMES','VK_DRIVER_FILES','VK_INSTANCE_LAYERS']:env.pop(name,None)
 raise SystemExit(subprocess.run([sys.executable,'-I',__file__,'child'],env=env).returncode)
import ctypes as C
l=C.CDLL('zig-out/lib/libbonsai.so');assert l.bonsai_abi_version()==1
l.bonsai_last_error.restype=C.c_char_p;l.bonsai_model_open.argtypes=[C.c_char_p,C.POINTER(C.c_void_p)];l.bonsai_model_close.argtypes=[C.c_void_p]
m=C.c_void_p();r=l.bonsai_model_open(b'data/Ternary-Bonsai-2-4B-PTQ1_0.gguf',C.byref(m));assert r==0,(r,l.bonsai_last_error());l.bonsai_model_close(m)
print('dlopen and Vulkan model initialization work without SDK, loader, or ICD environment variables')
