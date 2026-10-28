import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import ctypes as C,numpy as np,time,sys,threading,re,json
P=C.c_void_p;I=C.c_int32;F=C.c_float
l=C.CDLL('zig-out/lib/libbonsai.so')
l.bonsai_model_open.argtypes=[C.c_char_p,C.POINTER(P)];l.bonsai_context_create.argtypes=[P,I,I,C.POINTER(P)];l.bonsai_eval.argtypes=[P,C.POINTER(I),I,C.POINTER(F),I];l.bonsai_tokenize.argtypes=[P,C.c_char_p,I,C.POINTER(I),I];l.bonsai_context_close.argtypes=[P];l.bonsai_model_close.argtypes=[P];l.bonsai_context_reset.argtypes=[P];l.bonsai_generate_greedy.argtypes=[P,C.POINTER(I),I,I,C.c_uint32,C.POINTER(I)];l.bonsai_last_error.restype=C.c_char_p

def ck(r):
 if r<0:raise RuntimeError(r,l.bonsai_last_error())
 return r
peak={'rss_kib':0,'vram_bytes':0};running=True
def monitor():
 while running:
  try:
   status=open('/proc/self/status').read();rss=int(re.search(r'VmHWM:\s+(\d+)',status)[1]);peak['rss_kib']=max(peak['rss_kib'],rss)
   clients={}
   for name in os.listdir('/proc/self/fdinfo'):
    try:
     data=open('/proc/self/fdinfo/'+name).read();client=re.search(r'drm-client-id:\s+(\d+)',data);v=re.search(r'drm-memory-vram:\s+(\d+)\s+(\w+)',data)
     if client and v:clients[client[1]]=int(v[1])*({'KiB':1024,'MiB':1048576,'B':1}.get(v[2],1))
    except (OSError,ValueError):pass
   peak['vram_bytes']=max(peak['vram_bytes'],sum(clients.values()))
  except OSError:pass
  time.sleep(.01)
th=threading.Thread(target=monitor);th.start()
m=P();t=time.perf_counter();ck(l.bonsai_model_open(b'data/Ternary-Bonsai-2-4B-PTQ1_0.gguf',C.byref(m)));print('load_seconds',time.perf_counter()-t,flush=True)
n=int(sys.argv[1]) if len(sys.argv)>1 else 512
c=P();ck(l.bonsai_context_create(m,max(16384,n+128),512,C.byref(c)))
text=('The quick brown fox jumps over the lazy dog. Paris is the capital of France. Write a Python function to compute Fibonacci numbers.\n'*max(1,n//20+1)).encode();nt=ck(l.bonsai_tokenize(m,text,0,None,0));tokens=(I*nt)();ck(l.bonsai_tokenize(m,text,0,tokens,nt));ids=list(tokens)[:n];arr=(I*n)(*ids);logits=np.empty(248320,np.float32)
predict=int(sys.argv[3]) if len(sys.argv)>3 else 128
results=[]
for repeat in range(int(sys.argv[2]) if len(sys.argv)>2 else 3):
 ck(l.bonsai_context_reset(c));t=time.perf_counter();ck(l.bonsai_eval(c,arr,n,logits.ctypes.data_as(C.POINTER(F)),248320));pp=time.perf_counter()-t
 out=(I*predict)();t=time.perf_counter();ck(l.bonsai_generate_greedy(c,None,0,predict,1,out));tg=time.perf_counter()-t
 print('BENCH',repeat,'n',n,'pp_seconds',pp,'pp_tps',n/pp,'tg_seconds',tg,'tg_tps',predict/tg,'tokens',list(out)[:8],flush=True);results.append({'pp':n/pp,'tg':predict/tg})
running=False;th.join();print('PEAK',peak,flush=True);json.dump({'n':n,'results':results,'peak':peak},open('tools/bench_latest.json','w'));l.bonsai_context_close(c);l.bonsai_model_close(m)
