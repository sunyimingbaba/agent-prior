"""Capture actual installed native cache/generation/LoRA implementations without environment edits."""
import datetime
import hashlib
import importlib
import importlib.metadata as md
import json
from pathlib import Path
import shutil
r=Path(__file__).resolve().parent.parent;out=r/'runtime_source_snapshot';out.mkdir();modules=['transformers.cache_utils','transformers.generation.utils','transformers.models.qwen2_5_vl.modeling_qwen2_5_vl','peft.peft_model','peft.tuners.lora.layer'];files={}
for name in modules:
 module=importlib.import_module(name);src=Path(module.__file__);dest=out/(name.replace('.','_')+'.py');shutil.copy2(src,dest);files[str(dest.relative_to(r))]={'source_module':name,'installed_path':str(src),'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
meta={'ok':True,'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'packages':{k:md.version(k) for k in ['torch','transformers','peft']},'files':files,'purpose':'read-only source witness for native generation prep, DynamicCache autograd, and LoRA; captured during continuation without installing or editing packages'};(out/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n');print(json.dumps(meta,ensure_ascii=False))
