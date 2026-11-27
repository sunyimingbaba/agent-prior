"""Score controlled D/E response states with the existing CPU FP32 native reader."""
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parents[1]
sys.path.insert(0,str(BASE/'aftertool_joint_pool_20261127/scoring/code'))
import pool_reader as original
read,save,sha,load=original.read,original.save,original.sha,original.load

@torch.inference_mode()
def main(shard):
    reader=original.Reader()
    assert torch.get_num_threads()==4
    folder=ROOT/'pool270'
    inventory=read(ROOT/'inputs/inventory.json')
    names=[r['image'] for r in reader.rows.values() if r['attribute'] in inventory['common_attributes'] and r['pair'] in inventory['panels']['pool270']['complete_pairs']]
    assert len(names)==238
    (folder/'readouts').mkdir(exist_ok=True,parents=True)
    files={}; projections={}
    for name in names[shard::2]:
        src=folder/'raw'/(name+'.pt'); dst=folder/'scores'/(name+'.json')
        while not src.exists():time.sleep(2)
        proj=folder/'readouts'/(name+'.pt')
        if not dst.exists():
            raw=load(src)
            hidden=raw['states']
            assert hidden.shape==(4,3584)
            original_vis=reader.native.img_embeds
            vectors={}
            def grab(*args,**kwargs):
                item=original_vis(*args,**kwargs)
                key=kwargs['agent_hidden'].float().contiguous().numpy().tobytes()
                vectors[key]=item['image_feats'][0].detach().clone().cpu()
                return item
            reader.native.img_embeds=grab
            try:
                items=reader.score(name,hidden.repeat(2,1))['items']
            finally:reader.native.img_embeds=original_vis
            assert items[:4]==items[4:]
            v=torch.stack([vectors[h.float().contiguous().numpy().tobytes()] for h in hidden])
            tmp=proj.with_suffix('.part')
            torch.save(dict(image=name,source=sha(src),visual=v,rgb=raw['rgb']),tmp)
            tmp.replace(proj)
            save(dst,dict(image=name,source=sha(src),projection=sha(proj),rgb=raw['rgb'],
                          texts=raw['texts'],side=raw['side'],pair=raw['pair'],attribute=raw['attribute'],
                          alt_ids=reader.rows[name]['alt_ids'],items=items[:4]))
        data=read(dst)
        assert data['source']==sha(src) and data['projection']==sha(proj)
        files[name]=sha(dst);projections[name]=sha(proj)
        print(len(files),name,flush=True)
    save(folder/('scoring_'+str(shard)+'.json'),dict(files=files,projections=projections,code=sha(Path(__file__)),
         reader=sha(Path(original.__file__)),bank=sha(original.SCORING/'inputs/manifest.json'),threads=torch.get_num_threads()))
if __name__=='__main__':main(int(sys.argv[1]))
