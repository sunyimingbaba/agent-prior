from pathlib import Path
import hashlib,json,sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'multicap_rfpo_20261124/code'))
from runtime import setup,sha
from readouts import load_model,clip

@torch.inference_mode()
def main():
    setup();torch.set_num_threads(2)
    p=ROOT/'inputs/独立措辞.json'
    phrases=json.loads(p.read_text());rows=json.loads((ROOT/'inputs/real_roster.json').read_text())
    words=sorted({s for arm in phrases.values() for pair in arm for s in pair})
    known=sorted({s for r in rows for k in 'ABCDE' for s in r[k]})
    m=load_model(False);tok=clip.tokenize(words,context_length=77-m.prompt_length)
    old=clip.tokenize(known,context_length=77-m.prompt_length)
    assert len(torch.unique(tok,dim=0))==108 and not {tuple(v) for v in tok.tolist()} & {tuple(v) for v in old.tolist()}
    vec=torch.cat([m.txt_embeds(t)['text_feats'] for t in tok.split(12)])
    assert torch.isfinite(vec).all() and torch.allclose(vec.norm(dim=1),torch.ones(len(vec)),atol=1e-4)
    torch.save(dict(source=sha(p),words=words,features=vec,tokens=tok),ROOT/'inputs/fresh_text.pt')

if __name__=='__main__':main()
