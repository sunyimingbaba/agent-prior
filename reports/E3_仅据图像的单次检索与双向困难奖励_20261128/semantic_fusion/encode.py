"""Restore the original CUP text branch and validate its outputs against eight archived gallery vectors."""
import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
REPORTS=ROOT.parents[1]
REPO=REPORTS.parent
ASSETS=Path('/Users/botlv/cup-deploy-assets')
sys.path.insert(0,str(REPO/'CUP'))
from prompt_clip import clip
from prompt_clip.model import CLIP,LayerNorm,Transformer

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_bytes())
def save(p,z):p.write_text(json.dumps(z,ensure_ascii=False,indent=2)+'\n')

class TextClip(nn.Module):
    """Only the original checkpoint's text-side modules; call the project's encode_text."""
    encode_text=CLIP.encode_text
    incorporate_prompt=CLIP.incorporate_prompt
    def __init__(self,sd):
        super().__init__()
        vocab,width=sd['model.token_embedding.weight'].shape
        context=sd['model.positional_embedding'].shape[0]
        self.context_length=context
        layers=len({k.split('.')[3] for k in sd if k.startswith('model.transformer.resblocks.')})
        self.transformer=Transformer(width,layers,width//64,CLIP.build_attention_mask(self))
        self.token_embedding=nn.Embedding(vocab,width)
        self.ln_final=LayerNorm(width)
        self.positional_embedding=nn.Parameter(torch.empty_like(sd['model.positional_embedding']))
        self.text_projection=nn.Parameter(torch.empty_like(sd['model.text_projection']))
        self.dtype=torch.float32

class TextReader(nn.Module):
    def __init__(self,sd):
        super().__init__()
        self.model=TextClip(sd)
        self.prompt_embeddings=nn.Parameter(torch.empty_like(sd['prompt_embeddings']))
        dim=sd['prompt_embeddings'].shape[-1]
        self.prompt_dropout=nn.Dropout(.1)
        self.prompt_proj=nn.Linear(dim,dim)
        self.adapter_txt=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,dim))
        self.adapter_txt_mapping=nn.Sequential(nn.Linear(dim,1))
        required=self.state_dict()
        self.load_state_dict({k:sd[k] for k in required},strict=True,assign=True)

def inference(code):
    tree=ast.parse(code.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ViLTransformerSS')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='txt_embeds')
    env={'torch':torch}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(code),'exec'),env)
    return env['txt_embeds']

class CheckpointCallback:pass
class BackboneCallback:pass

@torch.inference_mode()
def main():
    protocol=ROOT/'事前口径.md'
    assert sha(protocol)=='ef57d20c8bc94f0449c18f9505b2f2b0f97806322fe21a5987ab9318a305baac'
    ckpt=ASSETS/'experiment_archive_2026-09/checkpoints/ucm_main_domain_tool.ckpt';digest=sha(ckpt)
    assert digest=='09ef20e0b354efbf8a7d4d49105cd92e1ae9412b3ea58820961265a065a874dc'
    with torch.serialization.safe_globals([(CheckpointCallback,'pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint'),
                                          (BackboneCallback,'__main__.MyBackboneFinetuning')]):
        original=torch.load(ckpt,map_location='cpu',weights_only=True,mmap=True)
    native=TextReader(original['state_dict']).float().eval();forward=inference(REPO/'CUP/vilt/modules/vilt_module.py')
    head=ASSETS/'experiment_archive_2026-09/models_aux/ucm_scene_cls.pt'
    labels=torch.load(head,weights_only=True,map_location='cpu')['classes']
    assert len(labels)==len(set(labels))==21
    sentences=['A remote sensing image of '+label+'.' for label in labels]
    prompt=original['hyper_parameters']['config']['prompt_length'];assert prompt==16
    gallery=REPORTS/'E3-2_UCM与RSITMD独立复验_20261124/results/ucm_text.pt'
    historical=torch.load(gallery,weights_only=True,map_location='cpu');control=historical['native_tokens'][:8]
    expected=historical['native'][:8];assert expected.shape==(8,768)
    tokens=clip.tokenize(sentences,context_length=77-prompt,truncate=True)
    actual=forward(native,control)['text_feats'];err=(actual-expected).abs().amax(1);assert float(err.max())<1e-4
    vectors=torch.cat([forward(native,b)['text_feats'] for b in tokens.split(16)])
    assert (vectors.norm(dim=1)-1).abs().max()<1e-6
    dst=ROOT/'场景类别向量.pt';torch.save(dict(classes=labels,text=sentences,features=vectors,tokens=tokens),dst)
    save(ROOT/'原生文本编码校验.json',dict(protocol=sha(protocol),checkpoint=digest,scene_head=sha(head),gallery=sha(gallery),
        text_module=sha(REPO/'CUP/vilt/modules/vilt_module.py'),clip_model=sha(REPO/'CUP/prompt_clip/model.py'),
        loaded_text_tensors=len(native.state_dict()),control_token_rows=list(range(8)),max_absolute_errors=err.tolist(),
        min_cosine=float(torch.nn.functional.cosine_similarity(actual,expected).min()),class_features=sha(dst)))
if __name__=='__main__':
    torch.set_num_threads(4);main()
