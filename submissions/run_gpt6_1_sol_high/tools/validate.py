import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import ctypes as C,subprocess,numpy as np,time,json,sys,threading
P=C.c_void_p;I=C.c_int32;F=C.c_float
lib=C.CDLL('zig-out/lib/libbonsai.so')
lib.bonsai_model_open.argtypes=[C.c_char_p,C.POINTER(P)];lib.bonsai_context_create.argtypes=[P,I,I,C.POINTER(P)];lib.bonsai_eval.argtypes=[P,C.POINTER(I),I,C.POINTER(F),I];lib.bonsai_tokenize.argtypes=[P,C.c_char_p,I,C.POINTER(I),I];lib.bonsai_context_close.argtypes=[P];lib.bonsai_model_close.argtypes=[P];lib.bonsai_context_reset.argtypes=[P];lib.bonsai_last_error.restype=C.c_char_p

def ck(r):
 if r<0:raise RuntimeError(r,lib.bonsai_last_error())
 return r

def metrics(a,b):
 af=a.astype(np.float64);bf=b.astype(np.float64);cos=np.dot(af,bf)/np.sqrt(np.dot(af,af)*np.dot(bf,bf));ap=af-af.max();bp=bf-bf.max();ap-=np.log(np.exp(ap).sum());bp-=np.log(np.exp(bp).sum());kl=(np.exp(ap)*(ap-bp)).sum()
 return dict(cos=float(cos),kl=float(kl),top1=int(np.argmax(a))==int(np.argmax(b)),ours=int(np.argmax(b)),ref=int(np.argmax(a)),rmse=float(np.sqrt(np.mean((a-b)**2))))
backend=sys.argv[1] if len(sys.argv)>1 else 'cpu'
n=int(sys.argv[2]) if len(sys.argv)>2 else 64
m=P();ck(lib.bonsai_model_open(b'data/Ternary-Bonsai-2-4B-PTQ1_0.gguf',C.byref(m)));c=P();ck(lib.bonsai_context_create(m,max(512,n+128),512,C.byref(c)))
text=('The quick brown fox jumps over the lazy dog. Paris is the capital of France. Write a Python function to compute Fibonacci numbers.\n'*max(1,n//20+1)).encode();nt=ck(lib.bonsai_tokenize(m,text,0,None,0));tokens=(I*nt)();ck(lib.bonsai_tokenize(m,text,0,tokens,nt));ids=list(tokens)[:n]
ref=subprocess.Popen(['ref/bin/bonsai-ref'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=open('tools/validate_ref_stderr.txt','w'),text=True,bufsize=1)
def call(cmd):
 ref.stdin.write(cmd+'\n');ref.stdin.flush();r=ref.stdout.readline().strip()
 if not r.startswith('ok'):raise RuntimeError(cmd[:100],r)
 return r
call('load data/Ternary-Bonsai-2-4B-PTQ1_0.gguf');call(f'ctx 0 {max(512,n+128)} {backend}')
a=np.empty(248320,np.float32);t=time.time();ck(lib.bonsai_eval(c,(I*len(ids))(*ids),len(ids),a.ctypes.data_as(C.POINTER(F)),248320));dur=time.time()-t;print('OUR pp',n,dur,n/dur,flush=True)
t=time.time();call('eval 0 '+str(len(ids))+' '+' '.join(map(str,ids)));print('REF pp',time.time()-t,flush=True);call('logits 0 tools/validate_ref_logits.bin');b=np.fromfile('tools/validate_ref_logits.bin',np.float32);res=[metrics(b,a)];print('prompt',res[-1],flush=True)
t0=time.time();our_time=0
steps=int(sys.argv[3]) if len(sys.argv)>3 else 16
for step in range(steps):
 token=int(b.argmax());one=(I*1)(token);t=time.time();ck(lib.bonsai_eval(c,one,1,a.ctypes.data_as(C.POINTER(F)),248320));our_time+=time.time()-t;call('eval 0 1 '+str(token));call('logits 0 tools/validate_ref_logits.bin');b=np.fromfile('tools/validate_ref_logits.bin',np.float32);res.append(metrics(b,a));print('step',step,res[-1],flush=True)
print('SUMMARY',backend,n,'agree',sum(r['top1'] for r in res)/len(res),'KL',sum(r['kl'] for r in res)/len(res),'cos',sum(r['cos'] for r in res)/len(res),'tg',steps/our_time,flush=True)
# Re-evaluate incrementally to verify batch/state consistency.
if n<=512:
 ck(lib.bonsai_context_reset(c));t=time.time()
 for token in ids:ck(lib.bonsai_eval(c,(I*1)(token),1,a.ctypes.data_as(C.POINTER(F)),248320))
 call('reset 0');call('eval 0 '+str(len(ids))+' '+' '.join(map(str,ids)));call('logits 0 tools/validate_ref_logits.bin');b=np.fromfile('tools/validate_ref_logits.bin',np.float32);print('incremental',time.time()-t,metrics(b,a),flush=True)
ref.stdin.write('quit\n');ref.stdin.flush();ref.wait();lib.bonsai_context_close(c);lib.bonsai_model_close(m)
