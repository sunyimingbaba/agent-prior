from pathlib import Path
import json
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
def main():
    def read(p):return json.loads(p.read_text())
    out={}
    for seed in 'ab':
        rs=read(ROOT/'records'/(seed+'_text_read.json'))['images'];sums=Counter();strict=set();increase=set();per_form={};caps=set()
        for r in rs:
            for i,letter in [(0,'A'),(2,'B')]:
                good,bad=r['scores'][i:i+2];delta=[good[k]-bad[k] for k in 'AB']
                improves=min(delta)>0;clean=min(good[k] for k in 'AB')>0 and max(bad[k] for k in 'AB')<0
                aligned=good['N']>=bad['N'];sums['slots']+=1;sums['improves_both']+=int(improves);sums['stable_crossing']+=int(clean);sums['crossing_native_ok']+=int(clean and aligned)
                if clean:strict.add(r['image'])
                if improves:increase.add(r['image'])
                if clean and aligned:caps.add(r['image'])
        out[seed]=dict(sums,post_images=len(rs),strict_images=len(strict),improved_images=len(increase),strict_preserve_native_images=len(caps))
    (ROOT/'results/text_proposals.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
