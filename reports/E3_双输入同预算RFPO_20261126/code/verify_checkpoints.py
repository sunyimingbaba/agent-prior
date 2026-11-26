import sys
from pathlib import Path
import torch
from study import STAGE, read, dump, sha


def main(location):
    stage = STAGE
    backup = Path(location)
    manifest = {}
    for line in (stage / "results/resume_sha256.txt").read_text().splitlines():
        checksum, remote = line.split()
        arm = Path(remote).parent.name
        path = backup / arm
        assert sha(path / "resume.pt") == checksum
        cp = read(stage / "models" / arm / "complete.json")
        assert sha(path / "adapter/adapter_model.safetensors") == cp["final"]
        state = torch.load(path / "resume.pt", weights_only=True, mmap=True, map_location="cpu")
        assert len(state["history"]) == 24 and state["decision"] == cp["receipt"] and state["initial"] == cp["initial"]
        assert state["opt"]["state"] and state["history"] == read(stage / "models" / arm / "steps.json")
        manifest[arm] = dict(adapter=cp["final"], resume=checksum, optimizer_tensors=len(state["opt"]["state"]), steps=24)
    assert len(manifest) == 8
    dump(stage / "results/training_backup.json", manifest)


if __name__ == "__main__":
    main(sys.argv[1])
