import concurrent.futures
import email
import hashlib
import html.parser
import json
from pathlib import Path
import subprocess
import urllib.parse
import zipfile
from packaging.requirements import Requirement
from packaging.markers import default_environment

dest=Path('/Users/botlv/cup-deploy-assets/rtx5090_wheels_20261115');dest.mkdir(exist_ok=True)
def fetch(url):return subprocess.check_output(['curl','-fsSL','--retry','2','--connect-timeout','20',url])
class Links(html.parser.HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':self.links += [v for k,v in attrs if k=='href']
def pytorch_link(name,version):
    url='https://download.pytorch.org/whl/cu128/'+name+'/';p=Links();p.feed(fetch(url).decode())
    candidates=[urllib.parse.urljoin(url,u) for u in p.links if name+'-'+version+'-cp312-cp312-' in urllib.parse.unquote(u) and 'x86_64.whl' in u]
    assert len(candidates)==1,(name,len(candidates));u=candidates[0];parts=urllib.parse.urlsplit(u)
    return {'name':name,'version':version,'url':urllib.parse.urlunsplit(parts._replace(fragment='')),'filename':urllib.parse.unquote(parts.path.rsplit('/',1)[1]),'sha256':parts.fragment.split('sha256=')[1]}
def pypi_link(name,version):
    data=json.loads(fetch('https://pypi.org/pypi/'+name+'/'+version+'/json'))
    xs=[x for x in data['urls'] if x['filename'].endswith('.whl') and ('x86_64' in x['filename'] or x['filename'].endswith('py3-none-any.whl')) and ('cp312' in x['filename'] or 'py3-none-' in x['filename'])]
    assert xs,(name,version);x=xs[0]
    return {'name':name,'version':version,'url':x['url'],'filename':x['filename'],'sha256':x['digests']['sha256']}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def download(x):
    target=dest/x['filename']
    if not target.exists():
        part=target.with_suffix('.whl.partial')
        subprocess.run(['curl','-fLsS','--retry','2','--connect-timeout','20','--max-time','1800',x['url'],'-o',str(part)],check=True)
        assert sha(part)==x['sha256'],x['filename'];part.rename(target)
    assert sha(target)==x['sha256'];x['bytes']=target.stat().st_size;print('WHEEL_VERIFIED',x['filename'],x['bytes'],flush=True);return x
items=[pytorch_link(n,v) for n,v in [('torch','2.7.1+cu128'),('torchvision','0.22.1+cu128'),('torchaudio','2.7.1+cu128')]]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:items=list(pool.map(download,items))
torchfile=dest/next(x['filename'] for x in items if x['name']=='torch')
with zipfile.ZipFile(torchfile) as z:
    meta=email.message_from_bytes(z.read(next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))))
env=default_environment();env.update(python_version='3.12',python_full_version='3.12.6',sys_platform='linux',platform_system='Linux',platform_machine='x86_64',extra='')
deps=[]
for value in meta.get_all('Requires-Dist'):
    req=Requirement(value)
    if (req.marker is None or req.marker.evaluate(env)) and (req.name.startswith('nvidia-') or req.name in ['triton','sympy']):
        version=next((s.version for s in req.specifier if s.operator=='=='), '1.13.3' if req.name=='sympy' else None);assert version;deps.append(pypi_link(req.name,version))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:items+=list(pool.map(download,deps))
(dest/'download_manifest.json').write_text(json.dumps(items,indent=2)+'\n')
print('ALL_RTX_WHEELS_VERIFIED',len(items),sum(x['bytes'] for x in items),flush=True)
