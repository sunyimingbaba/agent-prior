import importlib.metadata as md
import sys,tarfile
from pathlib import Path
from base import RUN,PREVIOUS,REBUILD,ASSETS,sha,read,dump,panel,weight
def main():
    rows=panel();rec=RUN/'records';hashes={}
    extra='--extra' in sys.argv
    for seed in ('a','b'):
        tag=('extra_' if extra else '')+seed
        dest=rec/f'{tag}.tar.gz'
        with tarfile.open(dest,'w:gz',compresslevel=3) as out:
            for key in (tag,tag+'_readouts'):
                folder=rec/key;proof=read(folder/'complete.json')
                assert len(proof['files'])==len(rows) and proof['adapter']==weight(seed)
                for c in rows:assert sha(folder/(c['image']+'.pt'))==proof['files'][c['image']]
                out.add(folder,arcname=key)
        hashes[tag]=sha(dest)
    base=ASSETS.parent/'CUP'
    source_files=[PREVIOUS/'code'/f for f in ('shared.py','train.py','policy.py','readouts.py')]
    source_files.extend([base/'agent/rollout.py',base/'agent/tools.py',base/'agent/grpo/scorer.py',base/'vilt/modules/vilt_module.py'])
    dump(rec/('extra_provenance.json' if extra else 'provenance.json'),dict(archives=hashes,adapters={k:weight(k) for k in ('a','b')},
          native=sha(REBUILD/'relation_candidates_20261120/native/features.pt'),text=sha(rec/'text.pt'),tools=sha(rec/'tools.json'),
          proxy_checkpoint=sha(REBUILD/'scorer_resume_20261113/run_01/checkpoints/last.ckpt'),
          source_files={str(p):sha(p) for p in source_files},gpu_environment=read(REBUILD/'after_tool_holdout_20261124/environment.json')['gpu'],
          reader_python=str(sys.version),reader_torch=md.version('torch')))
if __name__=='__main__':main()
