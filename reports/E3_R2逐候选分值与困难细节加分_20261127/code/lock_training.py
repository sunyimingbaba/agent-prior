"""Lock the exact R2 implementations, contexts and bonus assets used by every training arm."""
from pathlib import Path
import hashlib
import json

stage=Path(__file__).resolve().parents[1]
hist=stage.parent/"second_answer_5090_20261124"


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


files=("code/lock_training.py","code/graded_reader.py","code/run_graded.py","code/train_queue.sh","code/detail_reward.py","inputs/hard.pt","训练复现口径.md")
lock={"stage":{f:sha(stage/f) for f in files}}
for seed,source in (("a",hist),("b",hist/"repeat")):
    used=("code/train.py","code/policy.py","code/reader_worker.py","code/reader_client.py","code/readouts.py","code/shared.py",
          "initial.json","prepare.json","contexts.pt","R0/adapter/adapter_model.safetensors","R0/adapter/adapter_config.json")
    lock[seed]={f:sha(source/f) for f in used}
path=stage/"inputs/train_lock.json"
if path.exists():
    assert json.loads(path.read_text())==lock
else:
    path.write_text(json.dumps(lock,indent=2)+"\n")
