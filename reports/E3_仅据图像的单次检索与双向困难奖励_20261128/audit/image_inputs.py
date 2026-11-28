"""Build the expected fixed prompt/token signatures and original per-photo evidence locally."""
from collections import Counter
from pathlib import Path
import ast
import hashlib
import io
import json
import sys

import numpy as np
import pyarrow as pa
from PIL import Image
import torch
from transformers import AutoProcessor
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor

ROOT=Path(__file__).resolve().parents[1]
IMAGE=ROOT/'image_only'
REPORTS=ROOT.parent
CUP=REPORTS.parent
ASSETS=Path('/Users/botlv/cup-deploy-assets')
OLD=REPORTS/'E3_视觉细节复核与RFPO_20261127'
FORM=REPORTS/'E3-2_UCM与RSITMD独立复验_20261124'
FINE=REPORTS/'E3_UCM多描述奖励同预算RFPO_20261124'
POOL=REPORTS/'E3_回填后双收益训练图筛选_20261127'
QUERY='Describe the visible features of this remote sensing image.'
READOUT=ASSETS/'e3-source-transfer-20261125/readout'
MODEL=ASSETS/'models/Qwen2.5-VL-7B-Instruct'
CONVERT=Compose([Resize(224,interpolation=InterpolationMode.BICUBIC),CenterCrop(224),ToTensor(),
                 Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
def read(p):return json.loads(p.read_bytes())
def load(p):return torch.load(p,weights_only=True,map_location='cpu')
def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def dump(name,content):
    (ROOT/'audit'/name).write_text(json.dumps(content,ensure_ascii=False,indent=2)+'\n')
def tables():
    return {s:pa.ipc.open_file(pa.memory_map(str(ASSETS/'arrow/ucm'/('ucm_caption_karpathy_label_'+s+'.arrow')))).read_all() for s in ('train','val','test')}
def image(r,t):
    table=t[r['split']];j=r['row'];assert table['path'][j].as_py()==r['image']
    im=Image.open(io.BytesIO(table['image'][j].as_py())).convert('RGB')
    assert hashlib.sha256(im.tobytes()).hexdigest()==r['rgb']
    return im
def expected_messages():
    path=CUP/'CUP/agent/rollout.py'
    tree=ast.parse(path.read_text())
    system=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SYSTEM_PROMPT' for t in n.targets))
    method=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='rollout_one')
    start=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='messages')
    end=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='text')
    routine=compile(ast.Module(body=method.body[start:end],type_ignores=[]),str(path),'exec')
    env=dict(SYSTEM_PROMPT=system,system_prompt=system,caption=QUERY)
    exec(routine,env)
    return env['messages']
def main():
    lock=read(IMAGE/'inputs/manifest.json');roster=read(IMAGE/'inputs/roster.json');assert QUERY==lock['query']
    assert all(sha(IMAGE/f)==d for f,d in lock['files'].items()) and all(sha(REPORTS/f)==d for f,d in lock['score_inputs'].items())
    qwen=AutoProcessor.from_pretrained(MODEL,local_files_only=True);msgs=expected_messages()
    assert msgs[-1]['content']==[dict(type='image'),dict(type='text',text='查询文本: '+QUERY)] and len(msgs)==6
    template=qwen.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True);message_hash=hashlib.sha256(template.encode()).hexdigest()
    source={r['rgb_sha256']:r for r in read(FORM/'inputs/ucm.json')+read(OLD/'extra164/inputs/rows.json')}
    official=read(FORM/'records/tools.json');t=tables();tokens={};sizes={};held={};pixels={}
    train={r['rgb_sha256'] for r in read(POOL/'scoring/locks/training.json')['rows']};assert len(train)==24
    assert len(roster)==len(source)==374 and not train&set(source)
    for r in roster:
        assert set(r)==set(('row','rgb','split','image','panel','feedback')) and r['rgb'] in source
        ori=source[r['rgb']];assert ori['row']==r['row'] and ori['image']==r['image'] and ori.get('split','test')==r['split']
        im=image(r,t)
        caps=t[r['split']]['caption'][r['row']].as_py()
        assert caps==ori['captions'] and QUERY not in caps
        if r['panel']=='official':assert r['feedback']==official[r['rgb']][0]
        else:assert r['feedback']==ori['feedback']
        wh='x'.join(map(str,im.size));sizes[r['rgb']]=wh
        if wh not in tokens:
            ids=qwen(text=[template],images=[im],return_tensors='pt')['input_ids'].contiguous().numpy()
            tokens[wh]=dict(digest=hashlib.sha256(ids.tobytes()).hexdigest(),length=ids.shape[1])
    heads=[ASSETS/'experiment_archive_2026-09/models_aux/ucm_scene_cls.pt',ASSETS/'models/ViT-L-14.pt']
    proof=read(READOUT/'proof.json')
    assert sha(READOUT/'cup_native_trace.pt')==proof['trace'] and sha(heads[0])==lock['scene_head']
    weight_proofs={}
    for tag,record in lock['policies'].items():
        s=tag.split('_')[1]
        base=OLD if s=='strict' else POOL
        store=ASSETS/('e3-visual-detail-rfpo-20261127' if s=='strict' else 'e3-aftertool-joint-20261127')
        cp=base/'models'/tag/'complete.json'
        assert sha(cp)==record['proof'] and sha(store/'models'/tag/'adapter/adapter_model.safetensors')==read(cp)['final']==record['adapter']
        weight_proofs[tag]=record['adapter']
    dump('通用问题与原始输入独立核验.json',dict(query=QUERY,images=len(roster),counts=dict(Counter(r['panel'] for r in roster)),
         expected_chat_template_sha256=message_hash,expected_tokens_by_image_size=tokens,image_sizes=dict(Counter(sizes.values())),
         pixels_verified=len(source),num_train_intersections=0,scene_head=sha(heads[0]),clip=sha(heads[1]),
         reader=proof['trace'],reader_source=proof['reader'],adapters=weight_proofs))

if __name__=='__main__':
    torch.set_num_threads(2);main()
