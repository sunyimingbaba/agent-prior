"""Audit train/val inputs and create a text-purged training reward gallery."""
import ast
import collections
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

import pyarrow as pa
import torch

r=Path(__file__).resolve().parent.parent
base=Path('/private/AGSE_E1_20261110/legacy');assets=base/'assets'
assert str(r).startswith('/private/AGSE_E3_1_rebuild_20261112/separated_hard_negative_20261114')
(r/'inputs').mkdir(exist_ok=True)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()

def canon(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s).casefold()).strip().rstrip(' .!?')
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

source=base/'CUP/agent/train_scene_cls.py'
classes=next(ast.literal_eval(n.value) for n in ast.parse(source.read_text()).body
             if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='UCM_CLASSES' for t in n.targets))
assert classes==torch.load(assets/'ucm_scene_cls.pt',map_location='cpu',weights_only=True)['classes']
original=json.loads(Path('/private/AGSE_E3_1_rebuild_20261112/fixed_tool_22_20261113/base_input_manifest.json').read_text())
metadata={};hashes={}
for split,count in [('train',1680),('val',210)]:
    path=assets/'ucm'/f'ucm_caption_karpathy_label_{split}.arrow';hashes[split]=sha(path)
    assert hashes[split]==original['arrow_sha256'][split]
    t=pa.ipc.open_file(str(path)).read_all();assert len(t)==count and set(t.column_names)=={'image','caption','path','split'}
    rows=[]
    for i in range(len(t)):
        name=t['path'][i].as_py();label=(int(Path(name).stem)-1)//100;assert 0<=label<21
        rows.append({'row':i,'image':name,'split':split,'scene_id':label,'scene':classes[label],
                     'captions':t['caption'][i].as_py(),'image_bytes_sha256':hashlib.sha256(t['image'][i].as_py()).hexdigest()})
    metadata[split]=rows
    dump(r/'inputs'/f'{split}_metadata.json',{'arrow_sha256':hashes[split],'rows':rows})
val_keys={canon(c) for x in metadata['val'] for c in x['captions']}
val_hashes={x['image_bytes_sha256'] for x in metadata['val']}
clean_rows=[x for x in metadata['train'] if x['image_bytes_sha256'] not in val_hashes]
texts={}
for x in clean_rows:
    for c in x['captions']:
        if canon(c) in val_keys:continue
        item=texts.setdefault(c,{'text':c,'canonical':canon(c),'source_rows':[],'scene_ids':[]})
        if x['row'] not in item['source_rows']:item['source_rows'].append(x['row'])
        if x['scene_id'] not in item['scene_ids']:item['scene_ids'].append(x['scene_id'])
gallery=list(texts.values());lookup={x['text']:i for i,x in enumerate(gallery)}
eligible=collections.defaultdict(list);by_scene={};maps={}
for x in clean_rows:
    pos=sorted({lookup[c] for c in x['captions'] if c in lookup})
    negative=[i for i,t in enumerate(gallery) if x['scene_id'] in t['scene_ids'] and i not in pos]
    if pos and negative:
        eligible[x['scene_id']].append(x['row'])
        maps[str(x['row'])]={'positive':pos,'same_scene_negative':negative,
                             'global_negative':[i for i in range(len(gallery)) if i not in pos],
                             'prompt_caption':next(c for c in x['captions'] if c in lookup)}
for scene in range(21):
    by_scene[classes[scene]]={'train_images':sum(x['scene_id']==scene for x in metadata['train']),
        'eligible_images':len(eligible[scene]),'retained_unique_captions':sum(scene in x['scene_ids'] for x in gallery)}
rng=random.Random(2026111403)
train_rows=[rng.choice(eligible[s]) for s in range(21) if eligible[s]];rng.shuffle(train_rows)
assert len(train_rows)>=15 and len({metadata['train'][i]['image_bytes_sha256'] for i in train_rows})==len(train_rows)
previous=json.loads((r/'inputs/previous_plan.json').read_text())
seen=set(previous['training_rows'])|set(previous['validation_rows'])
available=[x['row'] for x in metadata['val'] if x['row'] not in seen]
val_rows=sorted(random.Random(2026111404).sample(available,50))
assert not {metadata['train'][i]['image_bytes_sha256'] for i in train_rows}&{metadata['val'][i]['image_bytes_sha256'] for i in val_rows}
assert not {x['canonical'] for x in gallery}&val_keys
dump(r/'inputs/training_text_gallery.json',{'text_order':[x['text'] for x in gallery],'entries':gallery,'maps':maps})
dump(r/'inputs/split_selection.json',{'training_rows':train_rows,'validation_rows':val_rows,
    'training_seed':2026111403,'validation_seed':2026111404,'excluded_previous_val_rows':sorted(seen),
    'unused_val_eligible_count':len(available),'training_scene_count':len(train_rows)})
audit={'ok':True,'train_images':1680,'val_images':210,'train_arrow_sha256':hashes['train'],'val_arrow_sha256':hashes['val'],
       'scene_label_source':str(source),'scene_label_source_sha256':sha(source),
       'scene_label_rule':'(integer filename stem - 1) // 100; UCM_CLASSES equals saved scene-head classes',
       'train_images_shared_with_val_by_bytes':1680-len(clean_rows),
       'original_unique_train_captions':len({c for x in metadata['train'] for c in x['captions']}),
       'val_unique_canonical_captions':len(val_keys),'training_reward_captions':len(gallery),
       'canonical_text_overlap_after_purge':0,'selected_training_images':len(train_rows),
       'eligible_training_images':sum(len(x) for x in eligible.values()),'class_coverage':by_scene,
       'test_arrow_read':False,'normalization':'Unicode NFKC, casefold, collapse whitespace, remove terminal spaces/.!?',
       'prompt_caption':'first original caption surviving validation text purge',
       'selection_sha256':sha(r/'inputs/split_selection.json')}
dump(r/'split_audit.json',audit);print(json.dumps(audit,ensure_ascii=False))
