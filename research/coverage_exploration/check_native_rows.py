import urllib.request,urllib.parse,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
root=Path(__file__).resolve().parents[2]/'.models/coverage'
def fetch(offset):
 q={'dataset':'KhangTruong/Tampered-Images-Combined-Dataset-2','config':'default','split':'cross_test','offset':offset,'length':10}
 try:
  with urllib.request.urlopen('https://datasets-server.huggingface.co/rows?'+urllib.parse.urlencode(q),timeout=50) as response: data=response.read()
  (root/f'native_rows_{offset}.json').write_bytes(data)
  d=json.loads(data)
  return offset,[(row['row']['dataset'],row['row']['id'],row['row']['image']['width'],row['row']['image']['height']) for row in d['rows']]
 except Exception as error:return offset,str(error)
for result in ThreadPoolExecutor(3).map(fetch,[0,7119,7000]):print(result,flush=True)
