#!/usr/bin/env python3
"""Record token ids and hidden-state steps during two original rollouts and one base-model control."""
import contextlib,csv,hashlib,io,json,os,random,sys,time
from pathlib import Path
import pyarrow as pa
import torch
from PIL import Image
root=Path('/private/AGSE_E1_20261110');base=root/'legacy';assets=base/'assets';parent=root/'e1_prior_compat_20261111';out=parent/'token_diagnostic_20261111'
assert not out.exists();out.mkdir();os.environ.update({'HF_HOME':str(out/'hf'),'TMPDIR':str(out),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.set_num_threads(8)
sys.path.insert(0,str(base/'CUP'));sys.path.insert(0,str(base/'CUP/agent'))
from rollout import load_mllm,rollout_one
from tools import scene_classify_with_conf
from prompt_clip import clip
model,processor=load_mllm(str(assets/'models/Qwen2.5-VL-7B-Instruct'),'cuda',str(assets/'ucm_main_lora'))
clipmodel,preprocess=clip.load(str(assets/'models/ViT-L-14.pt'),device='cuda',jit=False);clipmodel=clipmodel.float().cuda().eval()
h=torch.load(assets/'ucm_scene_cls.pt',map_location='cpu',weights_only=True);head=torch.nn.Linear(h['feat_dim'],len(h['classes']));head.load_state_dict(h['head'],strict=True);head=head.float().cuda().eval()
with (root/'e1_wording100_smoke_20261110/audit.csv').open(encoding='utf-8-sig',newline='') as f:audit={r['图像编号']:r for r in csv.DictReader(f)}
table=pa.ipc.open_file(str(assets/'ucm/ucm_caption_karpathy_label_train.arrow')).read_all();real_generate=model.generate
result=[]
for image_name,mode in [('101.tif','lora'),('1937.tif','lora'),('101.tif','base_adapter_disabled')]:
 ref=audit[image_name];idx=int(ref['Arrow行号']);assert table['path'][idx].as_py()==image_name
 image=Image.open(io.BytesIO(table['image'][idx].as_py())).convert('RGB');assert hashlib.sha256(image.tobytes()).hexdigest()==ref['图像RGB_SHA256'];caption=ref['第一条描述'];events=[];rounds=[]
 def scene(im):
  text,conf=scene_classify_with_conf(im,model=clipmodel,preprocess=preprocess,cls_head=head,cls_names=h['classes']);events.append({'text':text,'confidence':float(conf)});return text if conf>=0 else None
 def recorded_generate(*args,**kwargs):
  output=real_generate(*args,**kwargs);input_len=kwargs['input_ids'].shape[1];ids=output.sequences[0,input_len:].tolist()
  rounds.append({'input_length':int(input_len),'token_ids':ids,'tokens':processor.tokenizer.convert_ids_to_tokens(ids),'decoded_keep_special':processor.decode(ids,skip_special_tokens=False),'decoded_skip_special':processor.decode(ids,skip_special_tokens=True),'hidden_steps':len(output.hidden_states),'last_hidden_shape':list(output.hidden_states[-1][-1].shape)})
  return output
 model.generate=recorded_generate;trace=io.StringIO();ctx=model.disable_adapter() if mode=='base_adapter_disabled' else contextlib.nullcontext()
 with ctx,contextlib.redirect_stdout(trace):
  hidden,used,filtered=rollout_one(model,processor,image,caption,{'scene_classify':scene},max_rounds=3,max_new_tokens=64,device='cuda',verbose=True,injected_h_end=False,temperature=0.0)
 assert tuple(hidden.shape)==(3584,) and torch.isfinite(hidden).all()
 rec={'image':image_name,'mode':mode,'caption':caption,'trace':trace.getvalue(),'rounds':rounds,'tool_events':events,'tool_used':used,'tool_filtered':filtered}
 if mode=='lora':
  paired=parent/'paired_six_20261111/priors'/f'{Path(image_name).stem}_original.pt';old=torch.load(paired,map_location='cpu',weights_only=True).float();rec['cosine_to_previous_same_condition']=float(torch.nn.functional.cosine_similarity(hidden.float()[None],old[None]).item());rec['max_abs_diff_to_previous']=float((hidden.float()-old).abs().max())
 result.append(rec);print('DONE',image_name,mode,flush=True)
(out/'token_diagnostic.json').write_text(json.dumps({'special_token_ids':{'eos':processor.tokenizer.eos_token_id,'pad':processor.tokenizer.pad_token_id},'runs':result},ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('COMPLETE',flush=True)
