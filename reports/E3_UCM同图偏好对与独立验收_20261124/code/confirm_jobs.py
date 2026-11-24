"""Once-only nominated checkpoint evaluation in the untouched detail panel and original formal UCM gallery."""
import os,sys,time,torch,subprocess
from pathlib import Path
from env_pair import ROOT,ASSETS,REBUILD,read,dump,sha,savept,table,rgb,setup,weight
FORM=REBUILD/'after_tool_holdout_20261124'
def fresh():
    p=ROOT/'confirm/inputs/panel.json';lock=read(p.with_name('lock.json'))
    assert sha(p)==lock['panel'] and sha(ROOT/'验收与比较约定.md')==lock['protocol']
    return read(p)
@torch.inference_mode()
def prepare():
    from prompt_clip import clip
    from tools import scene_classify_with_conf
    m,pre=clip.load(str(ASSETS/'models/ViT-L-14.pt'),device='cpu',jit=False);m=m.float().eval().cuda()
    cp=torch.load(ASSETS/'ucm_scene_cls.pt',map_location='cpu',weights_only=True)
    head=torch.nn.Linear(cp['feat_dim'],len(cp['classes']));head.load_state_dict(cp['head']);head=head.eval().cuda();t=table();out={}
    for c in fresh():
        a,p=scene_classify_with_conf(rgb(t,c),model=m,preprocess=pre,cls_head=head,cls_names=list(cp['classes']));out[c['image']]=[a,float(p)]
    dump(ROOT/'confirm/tools.json',out)
def datasets():
    audit=read(FORM/'inputs/audit.json');assert sha(FORM/'inputs/ucm.json')==audit['regions']['ucm']['manifest']
    assert sha(ASSETS/'ucm/ucm_caption_karpathy_label_test.arrow')==audit['files']['ucm_test']
    vc=read(FORM/'inputs/ucm.json');ans=read(FORM/'tools.json');assert sha(FORM/'tools.json')==read(FORM/'environment.json')['tools']
    cf=fresh()
    return {'official':(table('test'),vc,{r['image']:ans[r['rgb_sha256']][0] for r in vc},'orig_query'),
            'blind':(table(),cf,{k:v[0] for k,v in read(ROOT/'confirm/tools.json').items()},'scene_query')}
@torch.inference_mode()
def reader(tags):
    from readouts import load_model,clip
    m=load_model(False);ds=datasets();cf=fresh();labels=sorted({s for r in cf for k in 'CDE' for s in r[k]});ix={s:i for i,s in enumerate(labels)}
    tt=clip.tokenize(labels,context_length=77-m.prompt_length);assert len(tt.unique(dim=0))==len(tt)
    hh=torch.cat([m.txt_embeds(t)['text_feats'] for t in tt.split(12)])
    savept(ROOT/'confirm/text.pt',dict(texts=labels,hard=hh,maps={r['image']:{k:[ix[q] for q in r[k]] for k in 'CDE'} for r in cf},panel=sha(ROOT/'confirm/inputs/panel.json')))
    for tag in tags:
        seed,arm=tag.split('_',1);sig=weight(seed) if arm=='start' else read(ROOT/'pilot'/tag/'complete.json')['final']
        for sub,(t,rs,ans,qkey) in ds.items():
            src=ROOT/'pilot/greedy'/tag/sub;fs=[];nm=[];hh=[]
            for c in rs:
                n=c['image'];p=src/(n+'.pt')
                while not p.is_file():time.sleep(1)
                obj=torch.load(p,weights_only=True);assert obj['adapter']==sig and obj['rgb']==c['rgb_sha256'] and obj['query']==c[qkey] and obj['image']==n
                v=m.img_embeds(m.preprocess(rgb(t,c))[None],agent_hidden=obj['prior'].float()[None])['image_feats'][0];assert abs(v.norm().item()-1)<1e-4
                fs.append(v);nm.append(n);hh.append(sha(p))
            while not (src/'complete.json').exists():time.sleep(1)
            assert read(src/'complete.json')['files']==dict(zip(nm,hh))
            savept(ROOT/'confirm'/(tag+'_'+sub+'.pt'),dict(features=torch.stack(fs),images=nm,sources=hh,adapter=sig))
if __name__=='__main__':
    setup();nom=read(ROOT/'inputs'/os.environ.get('E3_NOMINATION','nomination.json'));tags=[s+'_'+a for s in 'ab' for a in ('start',nom['arm'])]
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='read':reader(tags)
    elif sys.argv[1]=='deploy':
        for tag in tags:
            subprocess.run([sys.executable,__file__,'one',tag],check=True)
    elif sys.argv[1]=='one':
        import deploy_pair
        tag=sys.argv[2];assert tag in tags
        deploy_pair.datasets=datasets
        seed,arm=tag.split('_',1)
        if arm!='start':assert read(ROOT/'pilot'/tag/'complete.json')['final']==nom['adapters'][seed]
        deploy_pair.main(seed,arm)
    else:raise ValueError(sys.argv[1])
