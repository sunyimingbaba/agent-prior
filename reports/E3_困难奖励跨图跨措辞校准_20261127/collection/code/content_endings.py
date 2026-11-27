"""Document which forced true/false statements end with the same actual content token."""
from collections import Counter
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
def main():
    details={}
    counts=Counter()
    for panel,keys in [('train98',['A']),('pool270',['D','E'])]:
        proof=json.loads((ROOT/panel/'generation.json').read_text())
        for name in sorted(proof['files']):
            p=ROOT/panel/'raw'/(name+'.pt')
            r=torch.load(p,weights_only=True,map_location='cpu')
            for k,label in enumerate(keys):
                i=2*k
                ids=[r['ids'][j][r['content_indices'][j]] for j in (i,i+1)]
                strings=[r['texts'][j][slice(*r['content_offsets'][j])] for j in (i,i+1)]
                same=ids[0]==ids[1]
                details.setdefault(name,{})[label]=dict(panel=panel,attribute=r['attribute'],pair=r['pair'],
                    ids=ids,text=strings,same=same)
                counts[panel,label,same]+=1
    result=dict(counts={f'{p}_{k}_{v}':n for (p,k,v),n in sorted(counts.items())},images=details)
    (ROOT/'content_endings.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(result['counts'])
if __name__=='__main__':main()
