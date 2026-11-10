#!/usr/bin/env python3
"""Safely inspect local AGSE checkpoint metadata and load the CLIP TorchScript archive."""
import argparse
import json
import sys
import types
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--agse-model', type=Path, required=True)
parser.add_argument('--clip-model', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()

# The checkpoint stores two Lightning callback classes in its metadata.
# Use inert stand-ins and PyTorch's restricted weights-only loader.
class MyBackboneFinetuning:
    pass
for name in ('pytorch_lightning', 'pytorch_lightning.callbacks', 'pytorch_lightning.callbacks.model_checkpoint'):
    sys.modules[name] = types.ModuleType(name)
ModelCheckpoint = type('ModelCheckpoint', (), {'__module__':'pytorch_lightning.callbacks.model_checkpoint'})
sys.modules['pytorch_lightning.callbacks.model_checkpoint'].ModelCheckpoint = ModelCheckpoint
with torch.serialization.safe_globals([MyBackboneFinetuning, ModelCheckpoint]):
    state = torch.load(args.agse_model, map_location='cpu', weights_only=True, mmap=True)
config = state['hyper_parameters']['config']
weights = state['state_dict']
result = {'agse':{'weights_only_metadata_loaded':True, 'parameter_entries':len(weights),
                  'use_agent_prior':config.get('use_agent_prior'),
                  'agent_hidden_dim':config.get('agent_hidden_dim'),
                  'clip_model':config.get('clip_model'),
                  'agent_projection_weight_shape':list(weights['agent_proj.0.weight'].shape)},
          'clip':{}}
clip = torch.jit.load(str(args.clip_model), map_location='cpu').eval()
clip_state = clip.state_dict()
result['clip'] = {'torchscript_loaded':True, 'parameter_entries':len(clip_state),
                  'visual_position_shape':list(clip_state['visual.positional_embedding'].shape),
                  'text_embedding_shape':list(clip_state['token_embedding.weight'].shape)}
args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
