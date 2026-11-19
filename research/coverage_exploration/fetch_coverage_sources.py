import base64
import json
from pathlib import Path
import urllib.request

root = Path(__file__).resolve().parents[2] / '.models/coverage'
urls = {
 'hf': 'https://huggingface.co/api/datasets?search=coverage&limit=50',
 'onedrive': 'https://api.onedrive.com/v1.0/shares/u!' + base64.urlsafe_b64encode(b'https://1drv.ms/f/s!AggVhXcCj1FLhUUyUrqSpV_yI_GH').decode().rstrip('=') + '/root/children',
 'hifi': 'https://raw.githubusercontent.com/CHELSEA234/HiFi_IFDL/main/README.md',
 'hf_coverage': 'https://huggingface.co/api/datasets/ThaneJoss/COVERAGE/tree/main?recursive=true',
}
for name, url in urls.items():
 try:
  with urllib.request.urlopen(url, timeout=45) as response:
   data = response.read(3_000_000)
  (root / (name + '.txt')).write_bytes(data)
  if name == 'hf': print(name, [row['id'] for row in json.loads(data)], flush=True)
  else: print(name, len(data), flush=True)
 except Exception as error: print(name, type(error).__name__, str(error)[:150], flush=True)
