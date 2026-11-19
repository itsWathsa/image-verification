import urllib.request, json
from pathlib import Path
root=Path(__file__).resolve().parents[2]/'.models/coverage'
for term in ['coverage','coverage forgery','coverage dataset']:
 try:
  u='https://www.kaggle.com/api/v1/datasets/list?search='+urllib.parse.quote(term)
  with urllib.request.urlopen(u,timeout=30) as r: data=r.read()
  d=json.loads(data)
  print(json.dumps([(v.get('ref'),v.get('title'),v.get('totalBytes')) for v in d]),flush=True)
 except Exception as e: print(type(e).__name__,str(e)[:100],flush=True)
