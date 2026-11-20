from pathlib import Path
from shared import ROOT,sha,read,dump
def main():
    records={}
    for arm in ('old','caption','hard'):
        folder=ROOT/('policy_'+arm);index=read(ROOT/'adapters'/arm/'index.json')
        adapter=folder/'adapter/adapter_model.safetensors';update=read(folder/'update.json')
        assert index['output_sha']==update['adapter_hash']==sha(adapter)
        parts=sum((ROOT/'adapters'/arm/p['file']).stat().st_size for p in index['parts'])
        records[arm]=dict(adapter_sha=index['output_sha'],history_base_sha=index['base_sha'],adapter_bytes=adapter.stat().st_size,
                          portable_xor_compressed_bytes=parts,parts=len(index['parts']),resume_bytes=(folder/'resume.pt').stat().st_size,resume_sha=sha(folder/'resume.pt'))
    dump(ROOT/'checkpoint_index.json',records)
if __name__=='__main__':main()
