from collections import Counter
import re
from transformers import AutoTokenizer
from env_pair import ROOT,ASSETS,read,dump
def main():
    tok=AutoTokenizer.from_pretrained(ASSETS/'models/Qwen2.5-VL-7B-Instruct',local_files_only=True)
    out={}
    for seed,rs in read(ROOT/'inputs/candidates.json').items():
        counts=Counter();examples={}
        for r in rs:
            for role,pos in [('useful',0),('poor',1)]:
                for i in sorted({p[pos] for p in r['pairs']}):
                    ids=r['alternatives'][i]['ids'];display=tok.decode(ids,skip_special_tokens=True);raw=tok.decode(ids,skip_special_tokens=False)
                    counts[role+'_unique_within_image']+=1
                    counts[role+'_empty']+=int(not display.strip());counts[role+'_complete_answer']+=int(bool(re.search(r'<answer>\s*[^<\s][\s\S]*?</answer>',display)))
                    counts[role+'_addcriterion']+=int('addCriterion' in display)
                    if len(examples)<8:examples[r['image']+'_'+role]=raw
        out[seed]=dict(counts=counts,examples=examples)
    dump(ROOT/'results/decoded_actions.json',out)
if __name__=='__main__':main()
