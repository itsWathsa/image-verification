import io,json,urllib.request,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq
root=Path(__file__).resolve().parents[2]/'.models/coverage'
class Remote(io.RawIOBase):
 def __init__(self,url):
  self.url,self.pos,self.transferred=url,0,0
  request=urllib.request.Request(url+'&r=init',headers={'Range':'bytes=0-0'})
  with urllib.request.urlopen(request,timeout=45) as response:
   self.length=int(response.headers['Content-Range'].split('/')[-1])
   response.read()
 def readable(self): return True
 def seekable(self): return True
 def tell(self): return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.length+offset
  return self.pos
 def read(self,count=-1):
  end=self.length-1 if count<0 else min(self.length-1,self.pos+count-1)
  if self.pos>end:return b''
  request=urllib.request.Request(self.url+f'&r={self.pos}-{end}',headers={'Range':f'bytes={self.pos}-{end}'})
  with urllib.request.urlopen(request,timeout=60) as response: data=response.read()
  if len(data)!=end-self.pos+1:raise ValueError('Range not honored')
  self.pos+=len(data);self.transferred+=len(data)
  return data
def inspect(index):
 try:
  url=f'https://huggingface.co/datasets/KhangTruong/Tampered-Images-Combined-Dataset-2/resolve/main/data/cross_test-{index:05d}.parquet?download=true'
  file=Remote(url); parquet=pq.ParquetFile(file)
  columns=parquet.schema_arrow.names
  metadata=parquet.read(columns=['dataset','id','label'])
  rows=metadata.to_pylist(); found=[{'row':i,**row} for i,row in enumerate(rows) if row['dataset'].lower()=='coverage']
  report={'index':index,'bytes':file.length,'transferred':file.transferred,'found':found}
  (root/f'shard-{index}.json').write_text(json.dumps(report))
  print(index,'COVERAGE',len(found),'metadata bytes',file.transferred,flush=True)
  return report
 except Exception as e:
  print(index,type(e).__name__,str(e)[:120],flush=True); return {'index':index,'error':str(e)}
if __name__=='__main__':
 results=list(ThreadPoolExecutor(8).map(inspect,range(64)))
 (root/'shard_inventory.json').write_text(json.dumps(results,indent=2))
