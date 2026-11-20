"""Cross upright/rotated pixels and same-source, naturally generated Agent hidden states."""
import torch
from PIL import Image
from shared import ROOT, dump, read, rgb, setup, sha, table
from readouts import load_model

@torch.inference_mode()
def main():
    setup(); reader=load_model()
    t=table('val')
    meta=read(ROOT/'inputs/detail.json')
    plan=read(ROOT/'inputs/prior_swap_protocol.json')
    target=ROOT/'readout';target.mkdir(exist_ok=True)
    for state in plan['states']:
        folder=ROOT/'eval'/state
        records=read(folder/'detail.json')
        assert records['bundle']==sha(folder/'detail.pt')
        natural=torch.load(folder/'detail.pt',weights_only=True,map_location='cpu')
        cells=[];err=0.
        for k,item in enumerate(meta):
            orig=rgb(t,item)
            images=[orig,orig.transpose(Image.Transpose.ROTATE_180)]
            h=natural['hidden'][2*k:2*k+2].cuda()
            for i,img in enumerate(images):
                assert records['records'][2*k+i]['image']==item['image']
                pixels=reader.preprocess(img)[None].cuda()
                for j in range(2):
                    v=reader.img_embeds(pixels,agent_hidden=h[j:j+1])['image_feats'][0].cpu()
                    if i==j:
                        err=max(err,float((v-natural['visual'][2*k+i]).abs().max()))
                    cells.append(v)
        assert err==0.,(state,err)
        path=target/(state+'.pt')
        torch.save(torch.stack(cells).reshape(len(meta),2,2,-1),path)
        dump(target/(state+'.json'),dict(state=state,bundle=sha(path),input_bundle=records['bundle'],
                 protocol=sha(ROOT/'inputs/prior_swap_protocol.json'),max_deployment_error=err,images=[r['image'] for r in meta]))
        print(state,err,flush=True)
if __name__=='__main__':main()
