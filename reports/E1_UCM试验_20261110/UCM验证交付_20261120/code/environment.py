"""Private server paths and the frozen E1 experiment interface."""
import json
import hashlib
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PREV = ROOT.parent/'e1_three_dimensions_20261120'
sys.path.insert(0,str(PREV/'code'))
from common import ASSETS, BASE, PLAIN_CKPT, QUERY, image_for, digest, read, dump
from common import tables as old_tables

def val_data():
    import pyarrow as pa
    path = ASSETS/'ucm/ucm_caption_karpathy_label_val.arrow'
    return {'val':pa.ipc.open_file(str(path)).read_all()}
