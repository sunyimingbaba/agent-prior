"""Repeat the previous prespecified 56-image word transfer check on new outputs."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch

STAGE = Path(__file__).resolve().parents[2]
PREVIOUS = STAGE.parent/"E3_R2逐候选分值与困难细节加分_20261127"


def main():
    source = PREVIOUS/"analysis/extra_wording.py"
    spec = spec_from_file_location("previous_held_wording",source)
    audit = module_from_spec(spec)
    spec.loader.exec_module(audit)
    audit.ROOT = STAGE
    torch.set_num_threads(1)
    audit.main()


if __name__ == "__main__":
    main()
