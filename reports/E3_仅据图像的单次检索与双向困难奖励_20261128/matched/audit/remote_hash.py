"""Hash the exact active directory of each completed model on the training server."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

if __name__=="__main__":
    out={}
    for s in "ab":
        for a in ("r2","strict"):
            tag=s+"_"+a
            model=ROOT/"models"/tag
            out[tag]={str(p.relative_to(model)):sha(p) for p in sorted(model.rglob("*")) if p.is_file()}
            if (model/"complete.json").exists():
                cp=json.loads((model/"complete.json").read_bytes())
                assert 1<=cp["steps"]<=cp["scheduled_slots"]==24
                assert out[tag]["adapter/adapter_model.safetensors"]==cp["final"]
            else:
                assert json.loads((model/"no_eligible.json").read_bytes())["steps"]==0
    (ROOT/"audit/远端模型摘要.json").write_text(json.dumps(out,sort_keys=True,indent=2)+"\n")
