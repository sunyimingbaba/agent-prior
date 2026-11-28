"""Bind both possible reuse sources and the RSITMD donor permutation before reading UCM outcomes."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
SEM=ROOT.parent;STAGE=SEM.parent;BASE_CROSS=STAGE/"cross_replication"
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def read(p):return json.loads(p.read_bytes())
def save(p,z):p.write_text(json.dumps(z,ensure_ascii=False,indent=2)+"\n")
if __name__=="__main__":
    oldp=BASE_CROSS/"inputs/manifest.json";old=read(oldp)
    for name in ("donors.json","roster.json"):
        copy=ROOT/"inputs"/name;orig=BASE_CROSS/"inputs"/name
        assert sha(orig)==old["files"]["inputs/"+name]
        copy.write_bytes(orig.read_bytes())
    files=sorted((ROOT/"code").glob("*.py"))+[ROOT/"code/queue.sh",ROOT/"跨库事前口径.md",ROOT/"inputs/roster.json",ROOT/"inputs/donors.json"]
    spec={k:old[k] for k in ("query","label","images","captions","overlap","donor_shift","prior_gallery","arrow","tool_cache","head","scorer_sources","external")}
    spec.update(cross_plan=sha(oldp),cross_generator=sha(BASE_CROSS/"code/generate.py"),previous_protocol=sha(SEM/"事前口径.md"),
                first_train=sha(STAGE/"first_round/inputs/train_lock.json"),files={str(p.relative_to(ROOT)):sha(p) for p in files})
    save(ROOT/"inputs/manifest.json",spec)
