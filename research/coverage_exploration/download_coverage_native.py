"""Stream the HiFi-IFDL research bundle, saving only COVERAGE members."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import time
import urllib.request

root = Path(__file__).resolve().parents[2] / '.models/coverage'
url = 'https://drive.usercontent.google.com/download?id=1RYXTg0Q82KEvkeOtaaR5AZ0FBx5219SY&export=download&confirm=t'
class Progress:
 def __init__(self, stream): self.stream, self.total, self.last, self.buffer = stream, 0, 0, b''
 def read(self, count=-1):
  if len(self.buffer) < count:
   fetched = self.stream.read(max(1024**2, count-len(self.buffer)))
   self.total += len(fetched)
   self.buffer += fetched
  data, self.buffer = self.buffer[:count], self.buffer[count:]
  if self.total - self.last >= 20 * 1024**2:
   print(f'Scanned {self.total / 1024**2:.0f} MiB of research archive', flush=True); self.last = self.total
  if self.total > 6_000_000_000: raise ValueError('Unexpected download size')
  return data

saved, directories = [], set()
with urllib.request.urlopen(url, timeout=90) as response:
 stream = Progress(response)
 with tarfile.open(fileobj=stream, mode='r|gz') as archive:
  for member in archive:
   parts = PurePosixPath(member.name).parts
   if len(parts) >= 2 and parts[1] not in directories:
    directories.add(parts[1]); print('Archive section:', parts[1], flush=True)
   if not any(part.lower() in {'coverage', 'cover'} for part in parts):
    if saved and len(saved) >= 300: break
    continue
   if not member.isfile(): continue
   if member.size > 40_000_000: raise ValueError('Unexpected large COVERAGE member')
   clean = PurePosixPath(*[part for part in parts if part not in {'/', '.', '..'}])
   path = root / 'native' / str(clean)
   resolved = path.resolve()
   if not resolved.is_relative_to((root / 'native').resolve()): raise ValueError('Unsafe archive member')
   path.parent.mkdir(parents=True, exist_ok=True)
   data = archive.extractfile(member).read()
   path.write_bytes(data)
   saved.append({'member': member.name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
   if len(saved) % 50 == 0: print('Saved COVERAGE files:', len(saved), flush=True)
   (root / 'download_progress.json').write_text(json.dumps({'files':len(saved),'downloaded_bytes':stream.total}))
report = {'source':'https://github.com/CHELSEA234/HiFi_IFDL','download':url,'bytes_streamed':stream.total,'files':saved}
(root / 'download.json').write_text(json.dumps(report, indent=2))
print('Completed:',len(saved),'files; streamed',stream.total,'bytes', flush=True)
