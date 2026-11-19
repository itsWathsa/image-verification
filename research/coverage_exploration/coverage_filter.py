import json
from pathlib import Path
import urllib.request
import urllib.parse
root=Path(__file__).resolve().parents[2]/'.models/coverage'
query={'dataset':'ThaneJoss/COVERAGE','config':'coverage','split':'full','offset':0,'length':100}
url='https://datasets-server.huggingface.co/rows?'+urllib.parse.urlencode(query)
try:
 with urllib.request.urlopen(url,timeout=60) as response: data=response.read()
 (root/'filtered.json').write_bytes(data)
 d=json.loads(data)
 print('total',d.get('num_rows_total'),'rows',len(d.get('rows',[])),flush=True)
 for item in d.get('rows',[])[:2]:
  row=item['row']; print({k:v for k,v in row.items() if k not in {'image','mask'}},flush=True); print('image',row['image'].get('width'),row['image'].get('height'),flush=True)
except Exception as error: print(type(error).__name__,str(error),flush=True)
