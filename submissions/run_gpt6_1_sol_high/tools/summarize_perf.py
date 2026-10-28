import json,statistics,math
x=json.load(open('tools/performance.json'));ratios=[]
for n in ['1','512','2048','4096','8192']:
 a=x['ours'][n];b=x['reference'][n];ap=statistics.median(r['pp'] for r in a['results']);at=statistics.median(r['tg'] for r in a['results']);bp=statistics.median(r['pp'] for r in b['results']);bt=statistics.median(r['tg'] for r in b['results']);print(n,'pp',round(ap,1),'ref',round(bp,1),'tg',round(at,1),'ref',round(bt,1),'ratios',round(ap/bp,3),round(at/bt,3));ratios.append(at/bt)
 if n!='1':ratios.append(ap/bp)
a=max(v['peak']['vram_bytes'] for v in x['ours'].values());b=max(v['peak']['vram_bytes'] for v in x['reference'].values());ar=max(v['peak']['rss_kib'] for v in x['ours'].values());br=max(v['peak']['rss_kib'] for v in x['reference'].values());gate=max(1.5*b,b+512*2**20)
print('SCORE_GEOMEAN',math.exp(sum(map(math.log,ratios))/len(ratios)))
print('VRAM_MiB',a/2**20,b/2**20,'GATE',gate/2**20,'PASS',a<=gate)
print('RSS_MiB',ar/1024,br/1024,'GATE',br/1024+2048,'PASS',ar<=br+2048*1024)
