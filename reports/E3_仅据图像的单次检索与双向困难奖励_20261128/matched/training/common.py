"""Shared image-only checks for the real visual tensors and tokenized model prefix."""
from pathlib import Path
import hashlib
import io
import json

from PIL import Image
import torch

STAGE = Path(__file__).resolve().parents[1]
QUERY = "Describe the visible features of this remote sensing image."


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def pixels(table, case):
    assert set(case) == {"image","row","rgb_sha256","split"}
    assert table["path"][case["row"]].as_py() == case["image"]
    image = Image.open(io.BytesIO(table["image"][case["row"]].as_py())).convert("RGB")
    assert hashlib.sha256(image.tobytes()).hexdigest() == case["rgb_sha256"]
    return image


def locked():
    lock = read(STAGE/"inputs/train_lock.json")
    assert lock["query"] == QUERY and all(sha(STAGE/f)==h for f,h in lock["files"].items())
    rows = read(STAGE/"inputs/roster.json")["rows"]
    assert len(rows)==24 and rows==sorted(rows,key=lambda r:r["rgb_sha256"])
    return lock, rows


class PromptCheck:
    def __init__(self,processor):
        self.processor=processor
        self.template=None
        self.first_ids=None
    def __getattr__(self,key):
        return getattr(self.processor,key)
    def apply_chat_template(self,messages,*args,**kw):
        assert messages[-1] == {"role":"user","content":[{"type":"image"},{"type":"text","text":"查询文本: "+QUERY}]}
        assert self.template is None
        self.template=self.processor.apply_chat_template(messages,*args,**kw)
        return self.template
    def __call__(self,*args,**kw):
        data=self.processor(*args,**kw)
        if self.first_ids is None:
            assert len(kw["images"]) == 1 and kw["text"]==[self.template]
            self.first_ids=data.input_ids.cpu().clone()
        return data
