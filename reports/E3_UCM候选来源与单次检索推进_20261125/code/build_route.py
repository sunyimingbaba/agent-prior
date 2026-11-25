from collections import Counter,defaultdict
from rosters import REP,ROOT,Q,read,dump,sha
def main():
    orig=REP/'E3-1_UCM第二轮R3困难负样本复验_20261124'
    questions=read(Q/'inputs/questions.json')
    grouped=defaultdict(Counter);pairs=defaultdict(set);proof={}
    for sub in ('','confirm'):
        f=orig/sub/'inputs/panel.json';tools=orig/sub/'records/tools.json'
        t=read(tools);proof[sub]=dict(panel=sha(f),tools=sha(tools))
        for row in read(f):
            top=t[row['image']][0].split(' (',1)[0];attr=row['attribute']
            grouped[top][attr]+=1;pairs[top].add(row['pair'])
    route={}
    for top,counts in grouped.items():
        if len(pairs[top])<2:continue
        a=min(counts,key=lambda k:(-counts[k],k));route[top]=dict(attribute=a,question=questions[a],
            counts=dict(counts),training_pairs=len(pairs[top]))
    dump(ROOT/'inputs/route.json',dict(categories=route,source=proof,protocol=sha(ROOT/'单次部署核查口径.md'),
        questions=sha(Q/'inputs/questions.json')))
if __name__=='__main__':main()
