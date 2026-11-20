"""Encode all distinct training-caption token sequences with the frozen final AGSE."""
import torch
from shared import ROOT,table,read,dump,sha,setup,locked
from readouts import load_model,clip
setup()
cases=locked()
t=table()
model=load_model(False)
caps=t['caption'].to_pylist()
unique=list(dict.fromkeys(c for row in caps for c in row))
tokens=clip.tokenize(unique,context_length=77-model.prompt_length,truncate=False)
lookup={};representatives=[];kept=[];column={}
for i,(text,ids) in enumerate(zip(unique,tokens.tolist())):
    code=tuple(ids)
    if code not in lookup:
        lookup[code]=len(kept);kept.append(i);representatives.append(text)
    column[text]=lookup[code]
tokens=tokens[kept]
with torch.inference_mode():
    text=torch.cat([model.txt_embeds(z.cuda())['text_feats'].cpu() for z in tokens.split(32)])
assert (text.norm(dim=1)-1).abs().max()<1e-5
out=ROOT/'native';out.mkdir(exist_ok=True)
torch.save(dict(tokens=tokens,features=text),out/'features.pt')
maps={}
for c in cases:
    row=caps[c['row']]
    assert len(row)==5 and row[0]==c['orig_query']
    maps[c['image']]=dict(first=column[row[0]],positive=sorted({column[v] for v in row}),captions=row)
dump(out/'index.json',dict(raw=sum(map(len,caps)),unique_strings=len(unique),token_groups=len(kept),texts=representatives,
                           columns=column,maps=maps,sha256=sha(out/'features.pt')))
print('DONE',text.shape,flush=True)
