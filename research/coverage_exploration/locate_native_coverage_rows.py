import urllib.request,urllib.parse,json,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
root=Path(__file__).resolve().parents[2]/'.models/coverage'
found=threading.Event()
def fetch(offset):
 if found.is_set():return None
 q={'dataset':'KhangTruong/Tampered-Images-Combined-Dataset-2','config':'default','split':'cross_test','offset':offset,'length':1}
 try:
  with urllib.request.urlopen('https://datasets-server.huggingface.co/rows?'+urllib.parse.urlencode(q),timeout=35) as response: data=response.read()
  d=json.loads(data); row=d['rows'][0]['row']
  result={'offset':offset,'dataset':row['dataset'],'id':row['id']}
  if row['dataset'].lower()=='coverage':
   found.set();(root/'native_location.json').write_text(json.dumps({'offset':offset,'row':row},indent=2));print('FOUND COVERAGE',offset,row['image']['width'],row['image']['height'],flush=True)
  return result
 except Exception as error:return {'offset':offset,'error':str(error)}
results=[]
with ThreadPoolExecutor(10) as pool:
 futures=[pool.submit(fetch,index) for index in range(0,31103,200)]
 for future in as_completed(futures):
  result=future.result()
  if result:results.append(result)
  if len(results)%20==0 and result:print('Checked metadata positions:',len(results),flush=True)
(root/'native_position_inventory.json').write_text(json.dumps(results,indent=2))
