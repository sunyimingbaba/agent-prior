import sys
import torch
from runtime import ROOT,LAST,FORM,TRAIN,ASSETS,sha,read,table,initial
TAGS=[s+'_'+a for s in 'ab' for a in ('R2','R3','deployed','R0')]
def weights(tag):
    seed,arm=tag.split('_')
    if arm=='R0':p=initial(seed);sig=read(ROOT/'inputs/roster.json')['initial'][seed]
    elif arm=='deployed':
        p=TRAIN/('repeat' if seed=='b' else '')/'R2'
        sig=read(LAST/'records/environment.json')['starting'][seed]
    else:
        p=ROOT/'pilot'/tag;sig=read(p/'complete.json')['final']
    assert sha(p/'adapter/adapter_model.safetensors')==sig
    return p,sig
def datasets():
    lock=read(ROOT/'inputs/panel_lock.json');assert sha(ROOT/'inputs/panel.json')==lock['panel'] and sha(ROOT/'实验口径.md')==lock['protocol']
    audit=read(FORM/'inputs/audit.json')
    for sp in ('val','test'):assert sha(ASSETS/'ucm'/f'ucm_caption_karpathy_label_{sp}.arrow')==lock['arrows'][sp]
    assert sha(FORM/'inputs/ucm.json')==audit['regions']['ucm']['manifest'] and sha(FORM/'tools.json')==read(FORM/'environment.json')['tools']
    assert sha(TRAIN/'tools.json')==read(TRAIN/'prepare.json')['tools']
    tools=read(FORM/'tools.json');val=read(TRAIN/'tools.json');primary=read(ROOT/'inputs/panel.json')
    gallery=[dict(c,split='test') for c in read(FORM/'inputs/ucm.json')]
    data={'primary':(primary,{r['image']:(val if r['split']=='val' else tools)[r['rgb_sha256']][0] for r in primary},'scene_query'),
          'official':(gallery,{r['image']:tools[r['rgb_sha256']][0] for r in gallery},'orig_query')}
    for name,sub,fp in [('prior148','confirm','tools.json'),('prior142','reward_confirm','records/tools.json')]:
        dr=LAST/sub;p=dr/'inputs/panel.json';assert sha(p)==read(dr/'inputs/lock.json')['panel']
        rows=read(p);data[name]=(rows,{n:v[0] for n,v in read(dr/fp).items()},'scene_query')
    return data
def subsets(tag):return ['primary','official'] if tag.endswith('_R0') else ['primary','official','prior148','prior142']
