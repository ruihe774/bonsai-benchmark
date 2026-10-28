import subprocess,sys,json
results={'ours':{},'reference':{}}
for n in [1,512,2048,4096,8192]:
 print('Benchmark ours',n,flush=True)
 with open(f'tools/perf_ours_{n}.txt','w') as out,open(f'tools/perf_ours_{n}_stderr.txt','w') as err:subprocess.run([sys.executable,'-I','tools/bench.py',str(n),'3','128'],stdout=out,stderr=err,check=True)
 results['ours'][str(n)]=json.load(open('tools/bench_latest.json'));json.dump(results,open('tools/performance.json','w'))
for n in [1,512,2048,4096,8192]:
 print('Benchmark reference',n,flush=True)
 with open(f'tools/perf_ref_{n}.txt','w') as out:subprocess.run([sys.executable,'-I','tools/ref_bench.py',str(n)],stdout=out,check=True)
 results['reference'][str(n)]=json.load(open(f'tools/ref_bench_{n}.json'));json.dump(results,open('tools/performance.json','w'))
print('Performance measurements complete',flush=True)
