"""Calculate the actual image-derived tool feedback with independent local assets."""
import sys
import torch

from image_inputs import ASSETS, CUP, IMAGE, ROOT, dump, image, load, read, sha, tables
sys.path.insert(0,str(CUP/'CUP'))
from prompt_clip import clip

@torch.inference_mode()
def main():
    rows=read(IMAGE/'inputs/roster.json');source=tables()
    m,prep=clip.load(str(ASSETS/'models/ViT-L-14.pt'),device='cpu',jit=False)
    visual=m.float().eval().visual
    h=load(ASSETS/'experiment_archive_2026-09/models_aux/ucm_scene_cls.pt')
    head=torch.nn.Linear(h['feat_dim'],len(h['classes']));head.load_state_dict(h['head']);head.eval()
    result={}
    for i in range(0,len(rows),4):
        batch=rows[i:i+4]
        v=visual(torch.stack([prep(image(r,source)) for r in batch]),torch.zeros(len(batch),16,visual.conv1.weight.shape[0]))
        prob=head(torch.nn.functional.normalize(v,dim=-1)).softmax(-1)
        for j,r in enumerate(batch):
            indices=prob[j].topk(3).indices
            feedback=', '.join(f'{h["classes"][a]} ({prob[j,a].item():.2f})' for a in indices)
            assert feedback==r['feedback'],(r['image'],feedback,r['feedback'])
            result[r['rgb']]=float(prob[j].max())
    dump('场景分类器逐像素独立回算.json',dict(count=len(result),confidence=result))

if __name__=='__main__':
    torch.set_num_threads(4);main()
