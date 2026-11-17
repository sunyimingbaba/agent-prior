"""Additional authoritative validation identity+caption count guard, before final-policy outcomes."""
from milestone_common import ROOT,ASSETS
import pyarrow as pa
from stage_io import read,dump,sha
p=ASSETS/'ucm/ucm_caption_karpathy_label_val.arrow';ref=read(ROOT/'inputs/dataset_sources_reference.json')['val'];assert sha(p)==ref['arrow_sha256'];t=pa.ipc.open_file(str(p)).read_all();assert t.num_rows==ref['images']==210;counts=[len(x) for x in t['caption'].to_pylist()];assert set(counts)=={5} and sum(counts)==1050;dump(ROOT/'validation_identity_guard.json',{'ok':True,'arrow_sha256':sha(p),'authoritative_reference_sha256':sha(ROOT/'inputs/dataset_sources_reference.json'),'rows':210,'captions_per_image':5,'captions':1050,'analysis_protocol_lock_sha256':sha(ROOT/'analysis_protocol_lock.json')});print('AUTHORITATIVE_VAL_IDENTITY_210x5_PASSED',flush=True)
