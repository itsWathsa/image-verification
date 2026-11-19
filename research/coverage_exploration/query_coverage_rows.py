import urllib.request,urllib.parse,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
root=Path(__file__).resolve().parents[2]/'.models/coverage'
common={'dataset':'KhangTruong/Tampered-Images-Combined-Dataset-2','config':'default','split':'cross_test','offset':0,'length':100}
queries=[('search',{'query':'coverage'}),('filter',{'where':"dataset = 'Coverage'"}),('filter',{'where':"\"dataset\" = 'Coverage'"})]
def fetch(item):
 endpoint,extra=item
 try:
  u='https://datasets-server.huggingface.co/'+endpoint+'?'+urllib.parse.urlencode(common|extra)
  with urllib.request.urlopen(u,timeout=60) as response: data=response.read()
  name=endpoint+str(queries.index(item)); (root/(name+'.json')).write_bytes(data)
  d=json.loads(data)
  return name,d.get('num_rows_total'),[(r['row']['dataset'],r['row']['id'],r['row']['image']['width'],r['row']['image']['height']) for r in d['rows'][:3]]
 except Exception as error:return item,str(error)
for result in ThreadPoolExecutor(3).map(fetch,queries): print(result,flush=True)
