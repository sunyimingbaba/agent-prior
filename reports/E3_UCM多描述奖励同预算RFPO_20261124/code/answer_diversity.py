from collections import Counter
import numpy as np
from report_io import ROOT,steps,read,dump

def main():
    token=read(ROOT/'records/reply_tokens.json')['rows'][0]
    out={}
    for seed in 'ab':
        for arm in ('R2','R3'):
            tag=seed+'_'+arm;rows=steps(tag);groups=Counter();hits=0
            for r in rows:
                answers=[tuple(x['ids']) for x in r['records']]
                counts=Counter(answers);hits+=counts[tuple(token['ids'])]
                same=len(counts)==1;flat=np.ptp(r['retrieval'][arm])==0
                groups['identical']+=int(same);groups['no_signal']+=int(flat)
                groups['identical_and_flat']+=int(same and flat)
            out[tag]=dict(**groups,groups=len(rows),total_draws=16*len(rows),most_common_a_reply_occurrences=hits)
    dump(ROOT/'results/answer_diversity.json',dict(shared_reply=token,arms=out))
if __name__=='__main__':main()
