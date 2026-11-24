from pathlib import Path
import hashlib,os,sys
ROOT=Path(__file__).resolve().parents[1]
HIST=Path('/private/AGSE_E3_1_rebuild_20261112/r3_informative_groups_20261124')
OLD=HIST.parent/'r3_after_tool_trainpanel_20261124'
os.environ['E3_PANEL_ROOT']=str(ROOT)
sys.path.insert(0,str(OLD/'code'))
from base import PREVIOUS,REBUILD,ASSETS,model_path,weight,read,dump,sha,savept,table,rgb,setup
def hashkey(*values):return hashlib.sha256(('|'.join(map(str,values))).encode()).hexdigest()
def rows():
    return {r['image']:r for sub in ('','confirm') for r in read(OLD/sub/'inputs/panel.json')}
def fixtures(seed):
    rr=[r for r in read(ROOT/'inputs/candidates.json')[seed] if r['pairs']]
    plan=[]
    for cycle in range(32):
        for r in sorted(rr,key=lambda x:hashkey('photo',seed,cycle,x['image'])):
            ps=sorted(r['pairs'],key=lambda a:hashkey('pair',seed,r['image'],a))
            i,j=ps[cycle%len(ps)];a,b=[r['alternatives'][k] for k in (i,j)]
            plan.append(dict(image=r['image'],pos=a['ids'],neg=b['ids'],native_sign=1 if a['N']>b['N'] else -1 if a['N']<b['N'] else 0,win=[a[k] for k in ('A','B','N')],lose=[b[k] for k in ('A','B','N')],indices=[i,j]))
            if len(plan)==32:return plan
def inputs(proc,im,q,ids=None):
    import torch
    from policy import prompt
    inp=proc(text=[prompt(proc,q)],images=[im],return_tensors='pt').to('cuda')
    if ids is not None:
        s=inp.input_ids.shape[1];assert torch.equal(inp.input_ids.cpu(),ids[:,:s])
        inp['input_ids']=ids.cuda();inp['attention_mask']=torch.ones_like(inp.input_ids)
    return inp
def captured_prefix(proc,inp,r):
    import torch
    reply=proc.decode(r['first'],skip_special_tokens=True)
    addition=proc(text=[reply+'\n<result>'+r['tool']+'</result>\n'],return_tensors='pt').input_ids
    return torch.cat((inp.input_ids.cpu(),addition),1)

