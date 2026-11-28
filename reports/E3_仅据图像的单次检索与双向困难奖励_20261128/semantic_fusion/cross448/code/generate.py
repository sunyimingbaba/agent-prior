"""Connect a first-round checkpoint to the previously sealed, identical cross-gallery decoder."""
import hashlib,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];STAGE=ROOT.parents[1]
source=STAGE/"cross_replication/code/generate.py"
manifest=json.loads((ROOT/"inputs/manifest.json").read_bytes())
assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest["cross_generator"]
spec=importlib.util.spec_from_file_location("image_cross_generate",source)
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
run.ROOT=ROOT
assert run.STAGE==STAGE
if __name__=="__main__":
    assert not run.read(ROOT/"inputs/gate.json")["first_priors_reused"]
    if sys.argv[1]=="verify":run.verify()
    else:run.generate(sys.argv[1])
