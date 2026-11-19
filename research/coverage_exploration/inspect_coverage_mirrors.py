from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import urllib.request

root = Path(__file__).resolve().parents[2] / '.models/coverage'
urls = {
 'hifi_data': 'https://drive.usercontent.google.com/download?id=1RYXTg0Q82KEvkeOtaaR5AZ0FBx5219SY&export=download&confirm=t',
 'hf_metadata': 'https://huggingface.co/api/datasets/KhangTruong/Tampered-Images-Combined-Dataset-2',
 'hf_rows': 'https://datasets-server.huggingface.co/first-rows?dataset=KhangTruong/Tampered-Images-Combined-Dataset-2&config=default&split=train',
 'hf_packaged': 'https://huggingface.co/api/datasets/ThaneJoss/COVERAGE/tree/main?recursive=true',
}
def fetch(item):
 name, url = item
 try:
  request = urllib.request.Request(url, headers={'Range': 'bytes=0-65535'} if name == 'hifi_data' else {})
  with urllib.request.urlopen(request, timeout=45) as response:
   data = response.read(3_000_000 if name != 'hifi_data' else 65536)
   headers = {key: response.headers.get(key) for key in ('Content-Type','Content-Length','Content-Range','Content-Disposition')}
  (root / (name + '.txt')).write_bytes(data)
  return name, headers, len(data)
 except Exception as error: return name, type(error).__name__, str(error)[:160]
for result in ThreadPoolExecutor(4).map(fetch, urls.items()): print(result, flush=True)
