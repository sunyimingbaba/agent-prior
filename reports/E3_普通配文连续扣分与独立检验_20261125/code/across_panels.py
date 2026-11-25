"""Descriptive synthesis of all disjoint photos after freezing the numerical reward; separate primary tests stay intact."""
from multi_view import OUT,read,dump,contrasts,summary
def main():
    method=read(OUT/'inputs/multi_nomination.json')['method']
    tags=['fresh','extension','saved','extra'];parts={}
    for t in tags:parts[t]=read(OUT/'records'/(('multi_'+t if t in ('fresh','extension') else 'covered_'+t+'_images')+'.json'))
    ids={r['image'] for rs in parts.values() for r in rs};assert len(ids)==sum(map(len,parts.values()))//2
    rs=sum(parts.values(),[]);summary_all=summary(rs,[method,'random'])
    summary_all['contrast']={ref:contrasts(rs,method,ref) for ref in ('R2','old','anchor','random')}
    summary_all['by_panel']={t:{ref:contrasts(v,method,ref) for ref in ('R2','old')} for t,v in parts.items()}
    dump(OUT/'results/all_held_images.json',summary_all)
    print('n=',summary_all['images'],'new-old',summary_all['contrast']['old'],'new-R2',summary_all['contrast']['R2'],flush=True)
if __name__=='__main__':main()
