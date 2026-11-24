import torch
from safetensors import safe_open
from runtime import ROOT,TRAIN,initial,sha,dump,read
def main():
    torch.set_num_threads(1);out={}
    for seed in 'ab':
        p=initial(seed)/'adapter/adapter_model.safetensors'
        counts={};nonzero={};elements=0
        with safe_open(str(p),framework='pt',device='cpu') as f:
            for key in f.keys():
                layer='A' if '.lora_A.' in key else 'B' if '.lora_B.' in key else None
                assert layer is not None
                x=f.get_tensor(key);assert x.dtype==torch.float32 and torch.isfinite(x).all()
                counts[layer]=counts.get(layer,0)+1;nonzero[layer]=nonzero.get(layer,0)+int(torch.count_nonzero(x)>0);elements+=x.numel()
        assert counts['A']==counts['B'] and nonzero['B']==0 and nonzero['A']==counts['A']
        out[seed]=dict(adapter=sha(p),matrices=counts,nonzero_matrices=nonzero,parameters=elements)
    dump(ROOT/'records/initial_audit.json',out)
if __name__=='__main__':main()
