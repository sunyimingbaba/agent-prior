from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
POOL=ROOT.parent/"aftertool_joint_pool_20261127"
CAL=ROOT.parent/"hard_reward_calibration_20261127"
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def main():
    files=["阶段口径.md","training/训练口径.md","training/run.py","training/lock.py","training/run.sh"]
    res=dict(pool_lock=sha(POOL/"training/train_lock.json"),redistribution=sha(CAL/"code/reward.py"),
             baseline_history={},n=24,files={f:sha(ROOT/f) for f in files})
    for seed in "ab":
        folder=POOL/"models"/(seed+"_r2")
        cp=json.loads((folder/"complete.json").read_bytes())
        assert cp["scores"]==sha(folder/"steps.json") and cp["steps"]==24
        assert cp["final"]==sha(folder/"adapter/adapter_model.safetensors")
        res["baseline_history"][seed]=cp["scores"]
    p=ROOT/"training/strict_lock.json"
    if p.exists():
        assert json.loads(p.read_bytes())==res
    else:p.write_text(json.dumps(res,indent=2,ensure_ascii=False)+"\n")

if __name__=="__main__":main()
