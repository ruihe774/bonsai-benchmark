import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import ctypes as C,subprocess,numpy as np,json,random,time
P=C.c_void_p;I=C.c_int32;F=C.c_float
l=C.CDLL('zig-out/lib/libbonsai.so')
l.bonsai_model_open.argtypes=[C.c_char_p,C.POINTER(P)];l.bonsai_context_create.argtypes=[P,I,I,C.POINTER(P)];l.bonsai_eval.argtypes=[P,C.POINTER(I),I,C.POINTER(F),I];l.bonsai_tokenize.argtypes=[P,C.c_char_p,I,C.POINTER(I),I];l.bonsai_context_close.argtypes=[P];l.bonsai_model_close.argtypes=[P];l.bonsai_context_reset.argtypes=[P];l.bonsai_last_error.restype=C.c_char_p

def ck(r):
 if r<0:raise RuntimeError(r,l.bonsai_last_error())
 return r

def metric(a,b):
 a=a.astype(np.float64);b=b.astype(np.float64);cos=float(a@b/np.sqrt((a@a)*(b@b)));ap=a-a.max();bp=b-b.max();ap-=np.log(np.exp(ap).sum());bp-=np.log(np.exp(bp).sum());kl=float((np.exp(ap)*(ap-bp)).sum());return dict(cos=cos,kl=kl,match=int(a.argmax())==int(b.argmax()),ref=int(a.argmax()),ours=int(b.argmax()))
texts=[
 'Hello! How are you today?',
 'What is the capital of France? Explain briefly.',
 'The following is a proof by induction. Let n be a positive integer.',
 'Compute 12345 * 6789 and explain each step.\nAnswer:',
 'def fibonacci(n):\n    if n <= 1:\n        return n\n    return ',
 '#include <stdio.h>\nint main(void) {\n    printf("Hello, world!\\n");\n',
 'Once upon a time, in a village surrounded by mountains, a traveler arrived',
 '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nExplain Vulkan compute shaders.<|im_end|>\n<|im_start|>assistant\n',
 '请用中文解释什么是机器学习，并给出三个例子。',
 '日本語の文章を書いてください。桜の季節について。',
 'Привет! Расскажи мне о космосе и звёздах.',
 'مرحبا، ما هي عاصمة مصر؟',
 'नमस्ते! कृपया पाइथन में एक छोटा उदाहरण दें।',
 'café näive élève 👩🏽‍💻 🦊\n\t  text\r\n',
 'A B C D E F G\n1 2 3 4 5 6 7\nThe next character is',
 'SELECT customer_id, SUM(total) FROM orders GROUP BY customer_id ORDER BY',
 '```zig\nconst std = @import("std");\npub fn main() void {\n',
 'Translate to English: Bonjour, je voudrais acheter un billet pour Paris.',
 'Two trains leave different stations at the same time. One travels at 60 km/h',
 'The mitochondrion is an organelle found in most eukaryotic cells. It produces',
 'Write a JSON object with fields name, age, and city.\n{',
 'Rust ownership prevents data races because',
 'The chemical formula of water is H2O, whereas carbon dioxide is',
 '    \n\n\t \r\n punctuation?![]{}<> -- == +++\n',
]
corpus=open('data/samples/corpus.txt').read();rng=random.Random(9724)
for length in [128,512,1024,2048,4096,8192]:
 start=rng.randrange(max(1,len(corpus)-length));texts.append(corpus[start:start+length])
ref=subprocess.Popen(['ref/bin/bonsai-ref-cpu'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=open('tools/suite_ref_stderr.txt','w'),text=True,bufsize=1)
def call(cmd):
 ref.stdin.write(cmd+'\n');ref.stdin.flush();r=ref.stdout.readline().strip()
 if not r.startswith('ok'):raise RuntimeError(cmd[:100],r)
 return r
call('load data/Ternary-Bonsai-2-4B-PTQ1_0.gguf');call('ctx 0 16384 cpu')
m=P();ck(l.bonsai_model_open(b'data/Ternary-Bonsai-2-4B-PTQ1_0.gguf',C.byref(m)));c=P();ck(l.bonsai_context_create(m,16384,512,C.byref(c)))
a=np.empty(248320,np.float32);results=[];t0=time.time()
for case,text in enumerate(texts):
 ck(l.bonsai_context_reset(c));call('reset 0');btext=text.encode();n=ck(l.bonsai_tokenize(m,btext,0,None,0));ids=(I*n)();ck(l.bonsai_tokenize(m,btext,0,ids,n));ri=call('tokenize '+btext.hex());assert list(ids)==list(map(int,ri.split()[2:]))
 ck(l.bonsai_eval(c,ids,n,a.ctypes.data_as(C.POINTER(F)),248320));call('eval 0 '+str(n)+' '+' '.join(map(str,ids)));call('logits 0 tools/suite_ref_logits.bin');b=np.fromfile('tools/suite_ref_logits.bin',np.float32);r=metric(b,a);r.update(case=case,step=-1,n=n);results.append(r);print(r,flush=True)
 for step in range(4):
  token=int(b.argmax());ck(l.bonsai_eval(c,(I*1)(token),1,a.ctypes.data_as(C.POINTER(F)),248320));call('eval 0 1 '+str(token));call('logits 0 tools/suite_ref_logits.bin');b=np.fromfile('tools/suite_ref_logits.bin',np.float32);r=metric(b,a);r.update(case=case,step=step,n=n);results.append(r);print(r,flush=True)
summary={'comparisons':len(results),'top1':sum(r['match'] for r in results)/len(results),'mean_kl':sum(r['kl'] for r in results)/len(results),'max_kl':max(r['kl'] for r in results),'mean_cos':sum(r['cos'] for r in results)/len(results),'min_cos':min(r['cos'] for r in results),'seconds':time.time()-t0}
print('SUMMARY',summary,flush=True);json.dump({'summary':summary,'results':results},open('tools/suite_results.json','w'))
ref.stdin.write('quit\n');ref.stdin.flush();ref.wait();l.bonsai_context_close(c);l.bonsai_model_close(m)
