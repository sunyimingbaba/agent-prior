"""Reopen the archive and verify each candidate group's logged digest before release."""
from pathlib import Path
import gzip,hashlib,io,json,tarfile
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/"archive";out.mkdir(exist_ok=True)
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def read(path):return json.loads(path.read_bytes())
index={}
for tag in [s+"_"+a for s in "ab" for a in ("r2","strict")]:
    model=ROOT/"models"/tag
    records=read(model/"steps.json")
    files={model/"trajectories"/(r["image"]+".pt"):r["trajectory"] for r in records}
    files[model/"steps.json"]=read(model/"complete.json")["scores"]
    files[model/"complete.json"]=sha(model/"complete.json")
    signatures={}
    final=out/(tag+".tar.gz")
    with final.open("wb") as stream,gzip.GzipFile(fileobj=stream,mode="wb",filename="",mtime=0) as g:
        with tarfile.open(fileobj=g,mode="w|",format=tarfile.USTAR_FORMAT) as tar:
            for f,h in sorted(files.items()):
                b=f.read_bytes();assert hashlib.sha256(b).hexdigest()==h
                name=f.relative_to(ROOT).as_posix();entry=tarfile.TarInfo(name);entry.size=len(b)
                tar.addfile(entry,io.BytesIO(b));signatures[name]=h
    with tarfile.open(final) as tar:
        assert signatures=={x.name:hashlib.sha256(tar.extractfile(x).read()).hexdigest() for x in tar}
    index[tag]=dict(file=final.name,sha256=sha(final),members=signatures,slots=8*len(records))
(out/"SHA256.json").write_text(json.dumps(index,indent=2)+"\n")
