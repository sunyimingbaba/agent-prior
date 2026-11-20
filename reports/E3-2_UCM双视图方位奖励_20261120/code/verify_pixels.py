"""Record pixel hashes for the named training and validation inputs, including all rotation views."""
import hashlib
from PIL import Image
from shared import ROOT, dump, read, rgb, sha, table

def main():
    views = {'rotated':Image.Transpose.ROTATE_180, 'cw90':Image.Transpose.ROTATE_270, 'ccw90':Image.Transpose.ROTATE_90}
    output = {}
    for split, name in [('train','train'),('val','validation')]:
        arrow = table(split)
        items = {}
        for source in read(ROOT/'inputs'/f'{name}.json'):
            image = rgb(arrow, source)
            row = {'upright': source['rgb_sha256']}
            row.update({view:hashlib.sha256(image.transpose(turn).tobytes()).hexdigest() for view,turn in views.items()})
            items[source['image']] = row
        output[split] = items
    assert not set(output['train']) & set(output['val'])
    assert not {v for views in output['train'].values() for v in views.values()} & {v for views in output['val'].values() for v in views.values()}
    dump(ROOT/'results/source_pixels.json',output)

if __name__ == '__main__':
    main()
