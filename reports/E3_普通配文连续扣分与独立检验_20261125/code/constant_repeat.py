"""Fix the train-only mean shift and evaluate the same single prior on a disjoint image panel."""
import sys
from collections import defaultdict
import hashlib,io
import torch
from PIL import Image
from torchvision import transforms as T
from head_fit import ROOT,OUT,reader,pack,read,load,save,sha,head,params_on_device,merge
from head_data import RFPO,ASSETS
import pyarrow as pa
@torch.inference_mode()
def main(seed):
    torch.set_num_threads(3);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    l=read(ROOT/'inputs/head_nomination.json');p=OUT/seed/'hard'/('e%d.pt'%l['epoch']);assert sha(p)==l['checkpoints'][seed]['hard']
    checkpoint=load(p);sig=checkpoint['adapter'];base,w=params_on_device(checkpoint);train=merge(pack('first',seed),pack('second',seed))
    means={f:(head(train[f].cuda(),base,w)-train[f].cuda()).mean(0) for f in ('ordinary','scene')}
    del train;m=reader();d=pack('cross',seed);folder=OUT/'test/cross'/seed;x=d['ordinary'].cuda();vec=[]
    for j in range(0,len(x),12):vec.append(m(d['pixels'][j:j+12].cuda(),x[j:j+12]+means['ordinary']).cpu())
    save(folder/'constant.pt',dict(images=[r['image'] for r in d['rows']],adapter=d['adapter'],nomination=sha(ROOT/'inputs/head_nomination.json'),ordinary=torch.cat(vec)))
    transform=T.Compose([T.Resize(224,interpolation=T.InterpolationMode.BICUBIC),T.CenterCrop(224),T.ToTensor(),
                          T.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
    tb=pa.ipc.open_file(pa.memory_map(str(ASSETS/'ucm/ucm_caption_karpathy_label_train.arrow'))).read_all()
    for tag in ('saved','extra'):
        rows=read(ROOT/'inputs'/('covered_'+tag+'.json'));pixels=[];hs=[]
        for c in rows:
            n=c['image'];assert tb['path'][c['row']].as_py()==n
            im=Image.open(io.BytesIO(tb['image'][c['row']].as_py())).convert('RGB');assert hashlib.sha256(im.tobytes()).hexdigest()==c['rgb_sha256']
            pixels.append(transform(im))
            if tag=='extra':
                z=load(ROOT/'records/covered/extra'/seed/'raw'/(n+'.pt'));assert z['query']==c['scene_query'];h=z['hidden'][-1]
            else:
                upstream=ROOT.parent;inputs=[upstream/nm/'records'/seed/(n+'.pt') for nm in ('attribute_questions_20261125','attribute_questions_repeat_20261125')]
                q,=[p for p in inputs if p.is_file()];z=read(q.with_suffix('.json'))
                assert z['states']==sha(q) and z['groups']['scene']['query']==c['scene_query'];h=load(q)['scene'][-1]
            assert z['rgb']==c['rgb_sha256'] and z['adapter']==sig
            hs.append(h)
        pix=torch.stack(pixels);x=torch.stack(hs).cuda().float();out=OUT/'test'/('covered_'+tag)/seed;out.mkdir(parents=True,exist_ok=True)
        for arm,h in [('baseline',x),('hard',head(x,base,w)),('constant',x+means['scene'])]:
            vec=[]
            for j in range(0,len(x),12):vec.append(m(pix[j:j+12].cuda(),h[j:j+12]).cpu())
            save(out/(arm+'.pt'),dict(images=[c['image'] for c in rows],adapter=z['adapter'],nomination=sha(ROOT/'inputs/head_nomination.json'),scene=torch.cat(vec)))
        print(tag,seed,flush=True)
if __name__=='__main__':main(sys.argv[1])
