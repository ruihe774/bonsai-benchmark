import os,subprocess,threading,re,time,json,sys
n=int(sys.argv[1]);predict=128;repeats=3
ref=subprocess.Popen(['ref/bin/bonsai-ref'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=open(f'tools/ref_bench_{n}_stderr.txt','w'),text=True,bufsize=1)
peak={'rss_kib':0,'vram_bytes':0};running=True
def monitor():
 while running:
  try:
   status=open(f'/proc/{ref.pid}/status').read();rss=int(re.search(r'VmHWM:\s+(\d+)',status)[1]);peak['rss_kib']=max(peak['rss_kib'],rss);clients={}
   for name in os.listdir(f'/proc/{ref.pid}/fdinfo'):
    try:
     data=open(f'/proc/{ref.pid}/fdinfo/'+name).read();client=re.search(r'drm-client-id:\s+(\d+)',data);v=re.search(r'drm-memory-vram:\s+(\d+)\s+(\w+)',data)
     if client and v:clients[client[1]]=int(v[1])*({'KiB':1024,'MiB':1048576,'B':1}.get(v[2],1))
    except (OSError,ValueError):pass
   peak['vram_bytes']=max(peak['vram_bytes'],sum(clients.values()))
  except (OSError,TypeError):pass
  time.sleep(.01)
th=threading.Thread(target=monitor);th.start()
def call(cmd):
 ref.stdin.write(cmd+'\n');ref.stdin.flush();r=ref.stdout.readline().strip()
 if not r.startswith('ok'):raise RuntimeError(cmd[:100],r)
 return r
call('load data/Ternary-Bonsai-2-4B-PTQ1_0.gguf');text=('The quick brown fox jumps over the lazy dog. Paris is the capital of France. Write a Python function to compute Fibonacci numbers.\n'*max(1,n//20+1)).encode();r=call('tokenize '+text.hex());ids=list(map(int,r.split()[2:]))[:n];call('ctx 0 16384 vulkan')
results=[]
for repeat in range(repeats):
 call('reset 0');t=time.perf_counter();call('eval 0 '+str(n)+' '+' '.join(map(str,ids)));pp=time.perf_counter()-t;t=time.perf_counter();tokens=call('gen 0 '+str(predict)+' 1');tg=time.perf_counter()-t;print('REFBENCH',repeat,'n',n,'pp',n/pp,'tg',predict/tg,'tokens',tokens[:100],flush=True);results.append({'pp':n/pp,'tg':predict/tg})
running=False;th.join();print('PEAK',peak,flush=True);json.dump({'n':n,'results':results,'peak':peak},open(f'tools/ref_bench_{n}.json','w'));ref.stdin.write('quit\n');ref.stdin.flush();ref.wait()
