from pathlib import Path
import subprocess
import sys
from stage_io import read, dump, sha
r = Path(__file__).resolve().parent.parent
for n, h in read(r/'cache_code_hashes.json').items():
    assert sha(r/n) == h, n
p = subprocess.run([sys.executable, '-u', str(r/'code/generate_cache.py')])
dump(r/'cache_pipeline_result.json', {'status': 'complete' if p.returncode == 0 else 'failed', 'exit_code': p.returncode})
sys.exit(p.returncode)
