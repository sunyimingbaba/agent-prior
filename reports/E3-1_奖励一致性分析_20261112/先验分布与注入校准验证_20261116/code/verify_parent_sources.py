from pathlib import Path
from stage_io import read,dump,sha
r=Path(__file__).resolve().parent.parent;plan=read(r/'plan.json');roots={'balanced':Path(plan['parent_balanced_root']),'permutation':Path(plan['parent_permutation_root'])}
for n,z in read(r/'parent_source_hashes.json').items():
    key,rel=n.split('/',1);p=roots[key]/rel;assert p.stat().st_size==z['bytes'] and sha(p)==z['sha256'],n
dump(r/'parent_integrity.json',{'ok':True,'local_pinned_parent_files_verified':6});print('PINNED_PARENT_SOURCES_VERIFIED')
