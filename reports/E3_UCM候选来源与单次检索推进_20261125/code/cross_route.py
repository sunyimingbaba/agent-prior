"""Single-pass transfer: use the same UCM route and frozen UCM readout on the historical RSITMD split."""
import sys,time,tarfile
import torch
import pyarrow as pa
from remote_common import ROOT,BASE,setup,sha,read,dump,savept,rgb
FORM=BASE/'after_tool_holdout_20261124'

def dataset():
    proof=read(FORM/'inputs/audit.json');p=FORM/'inputs/rsitmd_caption_karpathy_label_test.arrow';f=FORM/'inputs/rsitmd.json'
    assert sha(p)==proof['files']['rsitmd_test'] and sha(f)==proof['regions']['rsitmd']['manifest']
    tools=FORM/'tools.json';assert sha(tools)==read(FORM/'environment.json')['tools']
    rows=read(f);assert len(rows)==448
    return pa.ipc.open_file(pa.memory_map(str(p))).read_all(),rows,read(tools)
@torch.inference_mode()
def generate(seed):
    from eval_io import weights
    from prefixes import greedy
    from policy import policy
    setup();torch.set_num_threads(1);table,rows,tools=dataset()
    assert read(ROOT/'results/nomination.json')['selected'] is None
    route=read(ROOT/'inputs/route.json');assert route['protocol']==sha(ROOT/'单次部署核查口径.md')
    path,sig=weights(seed+'_deployed');m,proc=policy(path);folder=ROOT/'records/cross'/seed/'raw'
    folder.mkdir(exist_ok=True,parents=True)
    for j,c in enumerate(rows):
        nm=c['image'];f=folder/(nm+'.pt');feedback=tools[c['rgb_sha256']][0];top=feedback.split(' (',1)[0]
        entry=route['categories'].get(top);query=c['orig_query']+(' '+entry['question'] if entry else '')
        if f.is_file():
            old=torch.load(f,weights_only=True);assert old['query']==query and old['adapter']==sig;continue
        im=rgb(table,c);h,ids,calls,_=greedy(m,proc,im,query,feedback)
        z=dict(image=nm,rgb=c['rgb_sha256'],adapter=sig,top=top,attribute=entry['attribute'] if entry else '',query=query,prior=h,rounds=ids,calls=calls)
        if j==0:
            anchor,_,_,_=greedy(m,proc,im,c['orig_query'],feedback)
            z['baseline']=anchor
        savept(f,z)
        if (j+1)%50==0:print(seed,j+1,len(rows),flush=True)
    dump(folder/'complete.json',dict(adapter=sig,files={c['image']:sha(folder/(c['image']+'.pt')) for c in rows}))
@torch.inference_mode()
def project():
    from readouts import load_model
    setup();torch.set_num_threads(3);t,rows,tools=dataset();m=load_model(False)
    for seed in 'ab':
        folder=ROOT/'records/cross'/seed/'raw';fs=[];history=[];raw_proof={};base=None
        for c in rows:
            nm=c['image'];p=folder/(nm+'.pt')
            while not p.is_file():time.sleep(1)
            z=torch.load(p,weights_only=True);assert z['image']==nm and z['rgb']==c['rgb_sha256']
            pix=m.preprocess(rgb(t,c))[None]
            def embed(h):return m.img_embeds(pix,agent_hidden=h.float()[None])['image_feats'][0].clone()
            fs.append(embed(z['prior']));raw_proof[nm]=sha(p)
            if 'baseline' in z:base=embed(z['baseline'])
            history.append(dict(image=nm,attribute=z['attribute'],top=z['top']))
        while not (folder/'complete.json').is_file():time.sleep(1)
        cp=read(folder/'complete.json');assert raw_proof==cp['files'] and base is not None
        f=folder.parent/'features.pt';savept(f,dict(features=torch.stack(fs),meta=history,baseline=base,adapter=cp['adapter'],route=sha(ROOT/'inputs/route.json')))
        a=folder.parent/'trajectories.tar.gz'
        with tarfile.open(a.with_suffix('.part'),'w:gz',compresslevel=3) as tf:
            for c in rows:tf.add(folder/(c['image']+'.pt'),arcname=c['image']+'.pt')
            tf.add(folder/'complete.json',arcname='complete.json')
        a.with_suffix('.part').replace(a)
        dump(folder.parent/'read.json',dict(features=sha(f),archive=sha(a),raw=sha(folder/'complete.json')))
        print(seed,len(rows),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='project':project()
    else:generate(sys.argv[1])
