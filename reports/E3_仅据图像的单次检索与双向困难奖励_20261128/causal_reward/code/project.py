"""Cross original hidden states with both pixel backgrounds in the exact historical readout environment."""
from pathlib import Path
import os,sys,json
import torch
BASE=Path('/private/AGSE_E3_1_rebuild_20261112')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'second_answer_5090_20261124/code'))
from shared import setup,rgb,table,sha,read,dump
from readouts import load_model
SOURCES={'98':BASE/'joint_answer_pairs_20261127/fresh98','114':BASE/'hard_reward_calibration_20261127/followup_pool'}
def load(p):return torch.load(p,map_location='cpu',weights_only=True)

@torch.inference_mode()
def main(tag):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available()
    lock=read(ROOT/'inputs/lock.json');assert all(sha(ROOT/k)==v for k,v in lock['files'].items())
    g=read(ROOT/'inputs/groups.json')[tag];source=SOURCES[tag];seed=g['seed']
    assert all(sha(source/k)==v for k,v in g['proofs'].items())
    old=read(source/'scoring_complete.json');rs=read(source/(seed+'_complete.json'))
    assert old['threads']==4
    setup();torch.set_num_threads(4);m=load_model(False);t=table('train')
    folder=ROOT/'records'/tag;folder.mkdir(parents=True,exist_ok=True);proof={}
    for j,(pair,rows) in enumerate(g['pairs'].items()):
        f=folder/(pair+'.pt')
        if f.exists():
            zz=load(f);assert zz['lock']==sha(ROOT/'inputs/lock.json') and zz['pair']==pair
        else:
            states=[];anchors=[];pix=[];raw_hash=[];archive_hash=[]
            for row in rows:
                name=row['image'];rawfile=source/'raw'/seed/(name+'.pt');ref=source/'readouts'/(name+'.pt')
                assert sha(rawfile)==rs['files'][name] and sha(ref)==old['readouts'][name]
                r=load(rawfile);v=load(ref)
                assert r['eligible'] and r['query']==row['orig_query'] and r['feedback']==row['feedback']
                assert r['rgb']==v['rgb']==row['rgb_sha256'] and v['source']==sha(rawfile)
                states.append(r['hidden'].float().contiguous());anchors.append(v['visual']);assert states[-1].shape==(8,3584)
                pix.append(m.preprocess(rgb(t,row))[None]);raw_hash.append(sha(rawfile));archive_hash.append(sha(ref))
            q=torch.empty((2,2,8,768),dtype=torch.float32)
            for i,x in enumerate(pix):
                cache={}
                for hside in (i,1-i):
                    for slot,h in enumerate(states[hside]):
                        key=h.numpy().tobytes()
                        if key not in cache:cache[key]=m.img_embeds(x,agent_hidden=h[None])['image_feats'][0].detach().clone().cpu()
                        q[i,hside,slot]=cache[key]
                    if hside==i:assert torch.equal(q[i,i],anchors[i]),(tag,pair,i,float((q[i,i]-anchors[i]).abs().max()))
            assert torch.isfinite(q).all()
            tmp=f.with_suffix('.part')
            torch.save(dict(pair=pair,images=[r['image'] for r in rows],rgb=[r['rgb_sha256'] for r in rows],hidden_sources=raw_hash,
                       native_sources=archive_hash,lock=sha(ROOT/'inputs/lock.json'),visual=q),tmp);tmp.replace(f)
        proof[pair]=sha(f)
        print(tag,j+1,len(g['pairs']),pair,flush=True)
    dump(folder/'complete.json',dict(tag=tag,files=proof,lock=sha(ROOT/'inputs/lock.json'),device='cpu',threads=torch.get_num_threads(),
        torch=str(torch.__version__),anchors_equal=16*len(proof),new_counterfactual_slots=16*len(proof),reader=sha(BASE/'second_answer_5090_20261124/code/readouts.py')))
if __name__=='__main__':main(sys.argv[1])
