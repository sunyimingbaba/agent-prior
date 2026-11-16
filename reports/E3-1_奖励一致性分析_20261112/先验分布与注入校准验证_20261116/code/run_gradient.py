from pathlib import Path
import subprocess
import sys
from stage_io import dump
r=Path(__file__).resolve().parent.parent;p=subprocess.run([sys.executable,'-u',str(r/'code/gradient_witness.py')]);dump(r/'gradient_pipeline_result.json',{'status':'complete' if p.returncode==0 else 'failed','exit_code':p.returncode});assert p.returncode==0
