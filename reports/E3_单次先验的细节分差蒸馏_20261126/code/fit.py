import sys,torch
from torch.nn import functional as F
from model_run import ROOT,read,dump,sha,setup,table,rgb,policy,reader,TRANS,greedy,hidden_replay,parent,locked

def fit(seed,arm):
    assert arm in ('ordinary','guard','teacher')
    lock=locked();targetfile=ROOT/'inputs'/('targets_'+seed+'.pt');assert sha(targetfile)==lock['teacher_files'][seed]
    src=parent(seed);modelhash=sha(src/'adapter/adapter_model.safetensors');assert modelhash==lock['starting'][seed]
    dest=ROOT/'models'/(seed+'_'+arm);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():
        cp=read(dest/'complete.json');assert cp['final']==sha(dest/'adapter/adapter_model.safetensors') and cp['lock']==sha(ROOT/'inputs/lock.json');return
    setup();torch.set_num_threads(1);model,proc=policy(src);params=[p.requires_grad_() for k,p in model.named_parameters() if '.lora_' in k]
    native=reader();opt=torch.optim.AdamW(params,lr=1e-6);bank=torch.load(targetfile,weights_only=True,map_location='cuda');t=table();steps=[]
    for i,row in enumerate(read(ROOT/'inputs/train.json')[seed]):
        im=rgb(t,row);pix=TRANS(im)[None].cuda();opt.zero_grad(set_to_none=True);views={}
        for view in ('scene','first'):
            q=row['scene_query' if view=='scene' else 'orig_query'];fb=row['feedback']
            actual,actions,calls,_=greedy(model,proc,im,q,fb)
            h=hidden_replay(model,proc,im,q,fb,actions,calls)
            actual=actual.float().cuda()
            cos=float(F.cosine_similarity(h.detach(),actual,dim=0));assert cos>0.97,(cos,row['image'])
            prior=actual+(h-h.detach());v=native(pix,prior[None])[0];gt=bank[row['image'],view]
            ordinary=(F.relu(gt['ordinary_target']-gt['ordinary_dir']@v)/.02).square().mean()
            trust=(v-gt['base']).square().sum()
            fine=(F.relu(gt['teacher' if arm=='teacher' else 'guard']-gt['hard_dir']@v)/.02).square().mean()
            loss=trust+ordinary+(fine if arm!='ordinary' else 0)
            (loss/2).backward()
            views[view]=dict(cos=cos,ordinary=ordinary.item(),fine=fine.item(),trust=trust.item(),actions=actions,calls=calls)
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));opt.step()
        steps.append(dict(step=i+1,image=row['image'],views=views,norm=norm));print(seed,arm,i+1,norm,flush=True)
    model.save_pretrained(dest/'adapter');dump(dest/'steps.json',steps)
    dump(dest/'complete.json',dict(initial=modelhash,final=sha(dest/'adapter/adapter_model.safetensors'),lock=sha(ROOT/'inputs/lock.json'),
            logs=sha(dest/'steps.json'),steps=len(steps),nonzero=sum(z['norm']>0 for z in steps)))
if __name__=='__main__':fit(*sys.argv[1:])
