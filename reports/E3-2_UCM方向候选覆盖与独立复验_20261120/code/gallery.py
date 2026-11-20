"""Replay the historical full gallery, passing each saved Agent prior through the deployed frozen CUP."""
import torch
from shared import ROOT,REBUILD,ASSETS,read,sha,setup,table,rgb,dump
from readouts import load_model
@torch.inference_mode()
def main():
    setup();reader=load_model();prev=REBUILD/'paired_views_20261120'
    proof=read(prev/'eval/base_00/complete.json');val=read(prev/'inputs/validation.json');t=table('val')
    out=ROOT/'gallery';out.mkdir(exist_ok=True);summary={}
    for cue in ('native','generic'):
        path=prev/f'eval/base_00/{cue}.pt';meta=prev/f'eval/base_00/{cue}.json'
        assert sha(meta)==proof['parts'][cue]
        records=read(meta);assert sha(path)==records['bundle']
        bank=torch.load(path,weights_only=True,map_location='cpu');feat=[]
        for row,tr,h in zip(val,records['records'],bank['hidden'],strict=True):
            assert row['image']==tr['image'] and row['rgb_sha256']==tr['rendered_sha']
            im=rgb(t,row)
            v=reader.img_embeds(reader.preprocess(im)[None].cuda(), agent_hidden=h[None].cuda())['image_feats'][0].cpu()
            feat.append(v)
        measured=torch.stack(feat)
        assert torch.equal(measured,bank['visual'])
        dest=out/(cue+'.pt');torch.save(measured,dest)
        summary[cue]=dict(n=len(feat),input=sha(path),replayed=sha(dest),exact=True)
    txt=prev/'inputs/retrieval_text.pt'
    assert sha(txt)==proof['text']
    from shutil import copyfile
    copyfile(txt,out/'text.pt')
    dump(out/'replay.json',dict(parts=summary,text=sha(txt),historical_mR={p:read(prev/'results/states.json')['base_00'][p]['mR'] for p in summary},
             image_order=[r['image'] for r in val],readout=proof['readout']))
if __name__=='__main__':main()
