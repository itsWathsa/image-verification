import urllib.request,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[2]/'.models/coverage'
url='https://huggingface.co/datasets/ThaneJoss/COVERAGE/resolve/main/coverage/parquet/full-00000-of-00001.parquet?download=true'
path=root/'processed.parquet'
if not path.exists():
 with urllib.request.urlopen(url,timeout=90) as response,(root/'processed.parquet.part').open('wb') as output:
  total=0
  while chunk:=response.read(1024**2):
   total+=len(chunk)
   if total>180_000_000:raise ValueError('Unexpected data size')
   output.write(chunk)
   if total%(20*1024**2)==0:print('Downloaded',total//1024**2,'MiB',flush=True)
 (root/'processed.parquet.part').replace(path)
with path.open('rb') as source:digest=hashlib.file_digest(source,'sha256').hexdigest()
assert digest=='8079bd790c028c20e733028c70d74ec12bee58718770118454ae6c2672bf0cdd'
print('Verified processed COVERAGE archive',path.stat().st_size,digest,flush=True)
