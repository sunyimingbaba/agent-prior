"""Recompute candidate image vectors using the verified four-thread CPU native model."""
from pathlib import Path
import hashlib,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parents[1];POOL=BASE/'aftertool_joint_pool_20261127'
ORIG=BASE/'second_answer_5090_20261124/code'
sys.path.insert(0,str(ORIG))
from shared import rgb,setup,sha,read,table
from readouts import load_model

def dump(path,obj):
    part=path.with_suffix('.part')
    part.write_text(json.dumps(obj,ensure_ascii=False)+'\n')
    part.replace(path)
def load(p):return torch.load(p,map_location='cpu',weights_only=True)

@torch.inference_mode()
def run(shard):
    setup();torch.set_num_threads(4)
    inputs=ROOT/'inputs'
    lock=read(inputs/'lock.json')
    assert sha(inputs/'c_vectors.pt')==lock['vectors'] and sha(inputs/'rows.json')==lock['rows']
    model=load_model(False);dataset=table('train');cv=load(inputs/'c_vectors.pt')
    native_lock=read(POOL/'scoring/inputs/manifest.json')
    bank=POOL/'scoring/inputs/vectors.pt'
    assert sha(bank)==native_lock['compiled']['vectors.pt']
    hard=load(bank)
    rows=[r for r in read(inputs/'rows.json') if r['panel']!='external']
    proofs={s:read(POOL/'sampling'/(s+'_complete.json'))['files'] for s in 'ab'}
    for pos,r in enumerate(rows[shard::2]):
        name=r['image'];pixels=model.preprocess(rgb(dataset,r))[None]
        cache={}
        for seed in 'ab':
            folder=ROOT/'results/candidates'/seed;folder.mkdir(parents=True,exist_ok=True)
            dst=folder/(name+'.json')
            rp=POOL/'sampling/raw'/seed/(name+'.pt'); sp=POOL/'scores'/seed/(name+'.json')
            old=read(sp)
            assert old['rgb']==r['rgb_sha256'] and sha(rp)==proofs[seed][name]==old['source']
            if dst.exists():
                assert read(dst)['historical_score']==sha(sp)
                continue
            raw=load(rp)
            assert raw['eligible'] and old['eligible'] and len(old['items'])==8
            margins=[];errors=[]
            for state,it in zip(raw['hidden'].float().contiguous(),old['items']):
                key=state.numpy().tobytes()
                if key not in cache:
                    cache[key]=model.img_embeds(pixels,agent_hidden=state[None])['image_feats'][0].double()
                v=cache[key];z=hard[name].double()@v
                correct=(z[::2]-z[1::2]).tolist()
                ref=it['hard_margins']+it['DE']+(it['FG'] or [])
                err=max(abs(a-b) for a,b in zip(correct,ref))
                assert err<1e-10,(name,seed,err)
                c=cv[name].double()@v
                margins.append(float(c[0]-c[1]));errors.append(err)
            dump(dst,dict(image=name,seed=seed,margin_true=margins,source=sha(rp),native_vectors=lock['vectors'],
                          historical_score=sha(sp),max_ABDEFG_error=max(errors)))
        print(pos+1,name,flush=True)
if __name__=='__main__':run(int(sys.argv[1]))
