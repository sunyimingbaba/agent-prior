"""Run the same strictly loaded deployed AGSE retriever for every unselected policy output."""
import sys
from pathlib import Path
import torch
from PIL import Image
from shared import ROOT,REBUILD,ASSETS,BASE,read,dump,sha,table,rgb,setup
from readouts import clip,load_model

def main(arm):
    setup();model=load_model(False);t=table('val')
    meta=read(ROOT/'inputs/validation.json');detail=read(ROOT/'inputs/detail.json')
    previous=REBUILD/'relation_transfer_20261120/val'
    binding=read(previous/'binding.json')
    for file in ['features.pt','index.json']:assert sha(previous/file)==binding['files'][file]
    orig=torch.load(previous/'features.pt',weights_only=True,map_location='cpu')
    tokens=orig['native_tokens'];texts=orig['native']
    token_to_idx={tuple(z):i for i,z in enumerate(tokens.tolist())}
    raw=[s for caps in t['caption'].to_pylist() for s in caps]
    ids=clip.tokenize(raw,context_length=77-model.prompt_length,truncate=False)
    order=[token_to_idx[tuple(tk)] for tk in ids.tolist()]
    assert len(meta)==len(t)==210 and len(order)==1050
    detailtexts=sorted({s for r in detail for w in ('C','D') for s in r[w]})
    dtokens=clip.tokenize(detailtexts,context_length=77-model.prompt_length,truncate=False)
    assert len({tuple(z) for z in dtokens.tolist()})==len(detailtexts)
    with torch.inference_mode():
        rel=torch.cat([model.txt_embeds(z.cuda())['text_feats'].cpu() for z in dtokens.split(24)])
        assert (model.txt_embeds(tokens[:3].cuda())['text_feats'].cpu()-texts[:3]).abs().max()<2e-5
        result={};hist=[];output=ROOT/'eval'/arm
        for part in ('native','generic','detail'):
            folder=output/part;manifest=read(folder/'complete.json')
            vectors=[];keys=[]
            for r in (detail if part=='detail' else meta):
                im=rgb(t,r)
                for mode in (['upright','rotated'] if part=='detail' else ['upright']):
                    im2=im if mode=='upright' else im.transpose(Image.Transpose.ROTATE_180)
                    pixel=model.preprocess(im2)[None].cuda()
                    file=folder/(Path(r['image']).stem+'_'+mode+'.pt')
                    proof=read(file.with_suffix('.json'))
                    assert proof['image']==r['image'] and proof['sha256']==sha(file) and proof['model']==manifest['model_sha256']
                    hs=torch.load(file,map_location='cpu',weights_only=True)
                    if arm=='base' and part=='native':
                        archived=BASE.parent/'ucm_prior_archive/agent_cache_domain_tool/ucm_caption_karpathy_label_val'/f"{r['row']}.pt"
                        other=torch.load(archived,weights_only=True,map_location='cpu').reshape_as(hs)
                        hist.append(dict(image=r['image'],equal=torch.equal(hs,other),archive=sha(archived),error=float((hs.float()-other.float()).abs().max())))
                    for tr,h in zip(proof['traces'],hs,strict=True):
                        vectors.append(model.img_embeds(pixel,agent_hidden=h[None].cuda())['image_feats'].cpu())
                        keys.append((r['image'],mode,tr['k']))
            assert len(keys)==manifest['trajectories']
            result[part]=dict(visual=torch.cat(vectors),order=keys)
        torch.save(result,output/'vectors.pt')
        # All arms use the same frozen text encoders. Independently save each tiny relation matrix for equality checking.
        torch.save(dict(relation=rel,texts=detailtexts,native=texts,native_tokens=tokens,caption_order=order),output/'text.pt')
        dump(output/'scored.json',dict(vectors=sha(output/'vectors.pt'),text=sha(output/'text.pt'),model=sha(ASSETS/'ucm_main_domain_tool.ckpt'),
                input_gallery=binding['files']['features.pt'],prior_archive_comparison=hist))
if __name__=='__main__':
    from run_once import main as synchronized
    synchronized(sys.argv[1],'score')
