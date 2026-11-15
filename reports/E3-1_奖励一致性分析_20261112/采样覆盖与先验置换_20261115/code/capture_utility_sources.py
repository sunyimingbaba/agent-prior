"""Record original Python source hashes around frozen-utility scoring."""
import argparse
from pathlib import Path
import datetime
from stage_io import read,dump,sha
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['before','after'],required=True);phase=p.parse_args().phase;r=Path(__file__).resolve().parent.parent;plan=read(r/'plan.json');base=Path(plan['base_root'])/'CUP';files={str(x.relative_to(base)):sha(x) for x in sorted(base.rglob('*.py')) if '__pycache__' not in x.parts}
assert 'vilt/modules/vilt_module.py' in files and 'run_frozen.py' in files
if phase=='after':assert files==read(r/'runtime_evidence/utility_python_sources_before.json')['files']
dump(r/f'runtime_evidence/utility_python_sources_{phase}.json',{'ok':True,'phase':phase,'base_directory':str(base),'files':files,'file_count':len(files),'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});print('UTILITY_PYTHON_SOURCES_RECORDED',phase,len(files))
