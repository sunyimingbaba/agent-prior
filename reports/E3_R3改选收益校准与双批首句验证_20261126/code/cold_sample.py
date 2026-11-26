"""Reuse the verified sampler and CPU reader on the matched roster with cold weights."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "soft_guard_20261125/code"))
import real_remote as original
import torch


def main(action, slot):
    original.ROOT = ROOT / "cold"
    original.setup()
    if action == "sample":
        from runtime import initial
        receipt = original.read(original.ROOT / "inputs/real_lock.json")
        def weights(name):
            seed = name.split("_")[0]
            start = initial(seed)
            digest = original.sha(start / "adapter/adapter_model.safetensors")
            assert digest == receipt["starts"][seed]
            return start, digest
        original.weights = weights
        torch.set_num_threads(1)
        original.sample(slot)
    elif action == "project":
        torch.set_num_threads(3)
        original.project(int(slot))
    else:
        raise ValueError(action)


if __name__ == "__main__":
    main(*sys.argv[1:])
