"""Encode the visually reviewed third wording, blind to candidate scores."""
import csv
from shared import ROOT,locked,sha,dump,setup
from readouts import load_model,clip
import torch
setup()
cases={r['image']:r for r in locked()}
file=ROOT/'review/heldout_C.tsv'
with file.open() as f:
    pairs=list(csv.DictReader(f,delimiter='\t'))
assert len(pairs)==len({r['pair'] for r in cases.values()}) and len({r['pair'] for r in pairs})==len(pairs)
texts=[];columns={}
for p in pairs:
    if p['decision'].lower()!='keep':continue
    start=len(texts);texts.extend([p['C1'],p['C2']])
    for i,side in enumerate(['image1','image2']):
        n=p[side].split('.')[0]+'.tif'
        assert cases[n]['pair']==p['pair'] and n not in columns
        assert all(s not in cases[n]['A']+cases[n]['B'] for s in texts[start:])
        columns[n]=[start+i,start+(1-i)]
m=load_model(False);tok=clip.tokenize(texts,context_length=77-m.prompt_length,truncate=False)
assert len({tuple(row) for row in tok.tolist()})==len(texts)
with torch.inference_mode():
    ft=torch.cat([m.txt_embeds(batch.cuda())['text_feats'].cpu() for batch in tok.split(32)])
out=ROOT/'heldout';out.mkdir(exist_ok=True)
torch.save(dict(tokens=tok,features=ft),out/'text.pt')
dump(out/'index.json',dict(texts=texts,columns=columns,text_sha256=sha(out/'text.pt'),labels_sha256=sha(file),
     protocol_sha256=sha(ROOT/'第三种措辞检验.md')))
print('DONE',len(texts))
