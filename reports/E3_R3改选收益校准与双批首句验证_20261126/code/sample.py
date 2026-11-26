"""Use the original sampling and readout implementations with a new output root."""
import sys
from pathlib import Path

stage = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(stage.parent/'soft_guard_20261125/code'))
import real_remote as original
import torch

if __name__ == '__main__':
    original.ROOT = stage
    original.setup()
    torch.set_num_threads(1 if sys.argv[1]=='sample' else 3)
    if sys.argv[1]=='sample':
        original.sample(sys.argv[2])
    else:
        original.project(int(sys.argv[2]))
