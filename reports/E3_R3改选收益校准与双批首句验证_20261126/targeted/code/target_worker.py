"""Use the same frozen CPU readout as the original training worker."""
import sys
from pathlib import Path
from target_reward import TargetReward
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "code"))
import policy_worker

if __name__ == "__main__":
    policy_worker.ROOT = ROOT
    policy_worker.Scorer = TargetReward
    policy_worker.main(sys.argv[1])
