from pathlib import Path
import hashlib
import io
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/private/AGSE_E1_20261110/legacy")
ASSETS = BASE/"assets"
PLAIN_CKPT = Path("/private/AGSE_E3_1_rebuild_20261112/scorer_resume_20261113/run_01/checkpoints/last.ckpt")
QUERY = "Describe the visible scene."
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HOME=str(ROOT/"scratch/hf"),
                  XDG_CACHE_HOME=str(ROOT/"scratch/cache"), MPLCONFIGDIR=str(ROOT/"scratch/mpl"),
                  TMPDIR=str(ROOT/"scratch"), OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
sys.path[:0]=[str(BASE/"CUP/agent"),str(BASE/"CUP")]

def read(path):
    return json.loads(Path(path).read_bytes())

def dump(path, data):
    temp=path.with_suffix(path.suffix+".part")
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    temp.replace(path)

def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def tables():
    import pyarrow as pa
    return {s:pa.ipc.open_file(str(ASSETS/f"ucm/ucm_caption_karpathy_label_{s}.arrow")).read_all() for s in ("train","val")}

def image_for(record, data):
    from PIL import Image
    table=data[record["split"]]; i=record["row"]
    assert table["path"][i].as_py()==record["image"]
    image=Image.open(io.BytesIO(table["image"][i].as_py())).convert("RGB")
    assert hashlib.sha256(image.tobytes()).hexdigest()==record["rgb_sha256"]
    return image
