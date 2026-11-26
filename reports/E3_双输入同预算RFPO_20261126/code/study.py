from pathlib import Path
import sys

STAGE = Path(__file__).resolve().parents[1]
PREV = STAGE.parent / 'E3_单次先验的细节分差蒸馏_20261126'
sys.path.insert(0, str(PREV/'code'))
from local import read, dump, load, sha, feature_path, banks, bank_for, digest, FORM, RFPO, OLD, calculate, summary_by_seed, paired
from same_policy import rr
from single_scores import fixed
from transfer_scores import counts, summary as retrieval_summary, scalar_values

ARMS = ('retained', 'singleplain', 'singlejoint', 'dualplain', 'dualjoint')
CONTRASTS = (('dualjoint','singlejoint'), ('dualjoint','dualplain'), ('dualjoint','retained'),
             ('dualplain','singleplain'), ('singlejoint','singleplain'), ('dualplain','retained'))
