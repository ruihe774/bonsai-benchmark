import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import ctypes as C,json,threading,numpy as np,tempfile
P=C.c_void_p;I=C.c_int32;F=C.c_float;L=C.c_int64
l=C.CDLL('zig-out/lib/libbonsai.so')
l.bonsai_last_error.restype=C.c_char_p
l.bonsai_model_open.argtypes=[C.c_char_p,C.POINTER(P)];l.bonsai_model_close.argtypes=[P]
l.bonsai_n_vocab.argtypes=[P];l.bonsai_n_ctx_train.argtypes=[P]
l.bonsai_tensor_count.argtypes=[P]
l.bonsai_tensor_info.argtypes=[P,I,C.POINTER(C.c_char_p),C.POINTER(I),C.POINTER(I),C.POINTER(L)]
l.bonsai_tensor_dequant.argtypes=[P,C.c_char_p,C.POINTER(F),L];l.bonsai_tensor_dequant.restype=L
l.bonsai_tokenize.argtypes=[P,C.c_char_p,I,C.POINTER(I),I]
l.bonsai_context_create.argtypes=[P,I,I,C.POINTER(P)];l.bonsai_context_close.argtypes=[P];l.bonsai_context_reset.argtypes=[P]
l.bonsai_eval.argtypes=[P,C.POINTER(I),I,C.POINTER(F),I]
l.bonsai_generate_greedy.argtypes=[P,C.POINTER(I),I,I,C.c_uint32,C.POINTER(I)]
assert l.bonsai_abi_version()==1
assert l.bonsai_model_open(None,None)==-1
assert l.bonsai_last_error()
assert l.bonsai_model_open(b'tools/no-such-file',C.byref(P()))<0
assert l.bonsai_n_vocab(None)<0
assert l.bonsai_context_reset(None)<0
l.bonsai_abi_version();assert l.bonsai_last_error() is None
seen=[]
def worker():
 assert l.bonsai_n_vocab(None)<0
 seen.append(l.bonsai_last_error())
t=threading.Thread(target=worker);t.start();t.join();assert seen[0] and l.bonsai_last_error() is None
m=P();assert l.bonsai_model_open(b'data/Ternary-Bonsai-2-4B-PTQ1_0.gguf',C.byref(m))==0,l.bonsai_last_error()
assert l.bonsai_n_vocab(m)==248320 and l.bonsai_n_ctx_train(m)==262144
assert l.bonsai_tensor_count(m)==426
name=C.c_char_p();typ=I();rank=I();shape=(L*4)()
assert l.bonsai_tensor_info(m,0,C.byref(name),C.byref(typ),C.byref(rank),shape)==0
assert name.value==b'output_norm.weight' and typ.value==0 and rank.value==1 and list(shape)==[2560,1,1,1]
a=np.full(2560,123,np.float32)
assert l.bonsai_tensor_dequant(m,name,a.ctypes.data_as(C.POINTER(F)),2559)==-2 and (a==123).all()
assert l.bonsai_tensor_dequant(m,name,a.ctypes.data_as(C.POINTER(F)),2560)==2560
assert np.isfinite(a).all() and a[0]==.669921875
assert l.bonsai_tensor_info(m,-1,None,None,None,None)==-1
assert l.bonsai_tensor_info(m,426,None,None,None,None)==-1
assert l.bonsai_tensor_dequant(m,b'nonexistent',None,0)==-1
vectors=json.load(open('TOKENIZER-VECTORS.json'));vectors=vectors if isinstance(vectors,list) else vectors.get('vectors',vectors.get('cases',[]))
for v in vectors:
 text=v.get('text',v.get('input','')).encode();expected=v['ids'];n=l.bonsai_tokenize(m,text,0,None,0);assert n==len(expected)
 ids=(I*n)();assert l.bonsai_tokenize(m,text,0,ids,n)==n;assert list(ids)==expected,(v,list(ids));assert l.bonsai_tokenize(m,text,1,ids,n)==n and list(ids)==expected
 if n:assert l.bonsai_tokenize(m,text,0,ids,n-1)==-2
# Exercise short, unaligned cache sizes and boundary errors.
for n_ctx in [1,17,65,513]:
 c=P();assert l.bonsai_context_create(m,n_ctx,1,C.byref(c))==0,l.bonsai_last_error()
 logits=np.empty(248320,np.float32);ids=(I*1)(9419)
 assert l.bonsai_eval(c,ids,1,logits.ctypes.data_as(C.POINTER(F)),1)==-2
 assert l.bonsai_eval(c,ids,1,logits.ctypes.data_as(C.POINTER(F)),248320)==0,l.bonsai_last_error();before=logits.copy()
 assert l.bonsai_eval(c,(I*1)(-1),1,None,0)<0
 assert l.bonsai_eval(c,ids,0,None,0)==-1
 assert l.bonsai_eval(c,ids,2147483647,None,0)==-2
 assert l.bonsai_context_reset(c)==0
 assert l.bonsai_generate_greedy(c,None,0,1,0,(I*1)())==-5
 assert l.bonsai_eval(c,ids,1,logits.ctypes.data_as(C.POINTER(F)),248320)==0
 assert np.array_equal(before,logits),'reset determinism'
 if n_ctx in (17,65):
  assert l.bonsai_context_reset(c)==0
  many=(I*n_ctx)(*([9419]*n_ctx))
  assert l.bonsai_eval(c,many,n_ctx,logits.ctypes.data_as(C.POINTER(F)),248320)==0,l.bonsai_last_error()
  assert np.isfinite(logits).all();first=logits.copy()
  assert l.bonsai_eval(c,ids,1,None,0)==-2
  assert l.bonsai_context_reset(c)==0
  assert l.bonsai_eval(c,many,n_ctx,logits.ctypes.data_as(C.POINTER(F)),248320)==0
  assert np.array_equal(first,logits),'batch reset determinism'
 l.bonsai_context_close(c)
# Contexts retain their model even if the caller releases its model handle.
c=P();assert l.bonsai_context_create(m,17,1,C.byref(c))==0
l.bonsai_model_close(m)
assert l.bonsai_eval(c,(I*1)(9419),1,None,0)==0
l.bonsai_context_close(c)
l.bonsai_model_close(None);l.bonsai_context_close(None)
print('ABI checks passed; tokenizer vectors',len(vectors),'; malformed/capacity/TLS/reset tests passed')
