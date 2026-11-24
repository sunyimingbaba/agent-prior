import torch
from common import ROOT,OLD,read,savept,setup,sha
from base import panel
from readouts import load_model,clip
@torch.inference_mode()
def main():
    setup();rows=panel();reader=load_model(False)
    texts=sorted({q for r in rows for k in 'CDE' for q in r[k]});ix={q:i for i,q in enumerate(texts)}
    tok=clip.tokenize(texts,context_length=77-reader.prompt_length)
    assert len(tok.unique(dim=0))==len(texts)
    bank=torch.load(OLD/'records/text.pt',weights_only=True);old=read(OLD/'inputs/panel.json')
    train={q for r in old for k in 'AB' for q in r[k]}
    train_tokens=clip.tokenize(sorted(train),context_length=77-reader.prompt_length)
    banned={tuple(v) for v in torch.cat((train_tokens,bank['native_tokens'])).tolist()}
    assert not banned.intersection(tuple(v) for v in tok.tolist())
    h=torch.cat([reader.txt_embeds(tt)['text_feats'] for tt in tok.split(12)])
    maps={r['image']:{k:[ix[q] for q in r[k]] for k in 'CDE'} for r in rows}
    savept(ROOT/'records/fresh_text.pt',dict(texts=texts,tokens=tok,hard=h,maps=maps,panel=sha(ROOT/'inputs/panel.json')))
if __name__=='__main__':main()
