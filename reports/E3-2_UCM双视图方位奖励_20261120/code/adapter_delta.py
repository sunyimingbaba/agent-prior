"""Store lossless XOR differences against the historical adapter, with header-aware safe-tensor reconstruction."""
import sys,json,struct,zlib,hashlib
from pathlib import Path
import numpy as np
from shared import sha,dump,read
def header(file):
    size=file.read(8);n=struct.unpack('<Q',size)[0];raw=file.read(n)
    return size+raw,json.loads(raw),n+8
def main(mode,base,path,dst):
    if mode=='pack':
        dst.mkdir(parents=True,exist_ok=True)
        with base.open('rb') as b,path.open('rb') as f:
            _,bh,off=header(b);raw,oh,origin=header(f)
            keys=sorted((k for k in oh if k!='__metadata__'),key=lambda k:oh[k]['data_offsets'][0])
            assert set(keys)==set(bh)-{'__metadata__'}
            index=dict(base_sha=sha(base),output_sha=sha(path),output_bytes=path.stat().st_size,parts=[])
            (dst/'header.bin').write_bytes(raw);index['header_sha']=sha(dst/'header.bin')
            # Split the raw tensor stream by tensor boundaries. Each part is independently decodable.
            pending=[];size=0;delta=bytearray()
            def flush():
                nonlocal pending,size,delta
                if not pending:return
                file=dst/f'xor_{len(index["parts"]):02}.z';file.write_bytes(zlib.compress(delta,6))
                index['parts'].append(dict(file=file.name,sha=sha(file),tensors=pending))
                pending=[];delta=bytearray();size=0
            end=0
            for k in keys:
                lo,hi=oh[k]['data_offsets'];a,c=bh[k]['data_offsets']
                assert lo==end and hi-lo==c-a and all(bh[k][v]==oh[k][v] for v in ('dtype','shape'))
                if size+hi-lo>32*1024*1024:flush()
                f.seek(origin+lo);b.seek(off+a)
                delta.extend((np.frombuffer(b.read(c-a),np.uint8)^np.frombuffer(f.read(hi-lo),np.uint8)).tobytes())
                size+=hi-lo;pending.append(k);end=hi
            flush();dump(dst/'index.json',index)
    elif mode=='restore':
        index=read(path/'index.json');assert sha(base)==index['base_sha'] and not dst.exists()
        assert sha(path/'header.bin')==index['header_sha']
        tmp=dst.with_suffix('.restoring');assert not tmp.exists()
        with base.open('rb') as b,tmp.open('xb') as out:
            _,bh,off=header(b);out.write((path/'header.bin').read_bytes())
            for part in index['parts']:
                file=path/part['file'];assert sha(file)==part['sha']
                delta=np.frombuffer(zlib.decompress(file.read_bytes()),np.uint8);i=0
                for k in part['tensors']:
                    a,c=bh[k]['data_offsets'];b.seek(off+a);n=c-a
                    out.write((np.frombuffer(b.read(n),np.uint8)^delta[i:i+n]).tobytes());i+=n
                assert i==len(delta)
        assert sha(tmp)==index['output_sha'] and tmp.stat().st_size==index['output_bytes']
        tmp.rename(dst)
    else:raise ValueError(mode)
if __name__=='__main__':main(sys.argv[1],*map(Path,sys.argv[2:]))
