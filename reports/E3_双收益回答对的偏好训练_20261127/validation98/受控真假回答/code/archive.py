"""Check every member of the raw outputs, native projections and scores."""
from pathlib import Path
import gzip
import hashlib
import io
import json
import tarfile

from analyze import ROOT, digest, read, write


def main():
    generation = read(ROOT/"generation.json")
    scoring = read(ROOT/"scoring.json")
    files = {}
    for name,sig in generation["files"].items():
        files["raw/"+name+".pt"] = sig
        files["readouts/"+name+".pt"] = scoring["readouts"][name]
        files["scores/"+name+".json"] = scoring["files"][name]
    assert len(files)==294
    for name in ("calibration.pt","calibration.json","calibration_readouts.json","generation.json","scoring.json"):
        files[name]=digest(ROOT/name)
    folder = ROOT/"archive"
    folder.mkdir(exist_ok=True)
    out = folder/"results.tar.gz"
    tmp = out.with_suffix(".gz.part")
    with tmp.open("wb") as handle,gzip.GzipFile(filename="",mode="wb",fileobj=handle,mtime=0) as gz:
        with tarfile.open(fileobj=gz,mode="w|",format=tarfile.USTAR_FORMAT) as tar:
            for rel, sig in sorted(files.items()):
                content=(ROOT/rel).read_bytes()
                assert hashlib.sha256(content).hexdigest()==sig
                member=tarfile.TarInfo(rel)
                member.size=len(content)
                tar.addfile(member,io.BytesIO(content))
    checked={}
    with tarfile.open(tmp,"r|gz") as tar:
        for m in tar:
            assert m.isfile() and m.name not in checked
            with tar.extractfile(m) as handle:
                checked[m.name]=hashlib.file_digest(handle,"sha256").hexdigest()
    assert checked==files
    tmp.replace(out)
    write(folder/"index.json",dict(archive=out.name,sha256=digest(out),bytes=out.stat().st_size,members=files))


if __name__=="__main__":
    main()
