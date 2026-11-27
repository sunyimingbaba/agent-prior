"""Archive and re-read the exact server-verified trajectory for every roster image."""
from pathlib import Path
import gzip
import hashlib
import io
import json
import sys
import tarfile

from verify_copy import ROOT, read, sha


def archive(seed):
    assert seed == "d"
    folder = ROOT / "archive"
    folder.mkdir(exist_ok=True)
    receipt = read(ROOT / f"{seed}_complete.json")
    manifest_hash = sha(ROOT / f"{seed}_complete.json")
    assert read(ROOT / f"{seed}_copy_verified.json")["manifest"] == manifest_hash
    output = folder / f"{seed}_raw.tar.gz"
    tmp = output.with_suffix(".gz.part")
    index = {}
    with tmp.open("wb") as handle, gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as compressor:
        with tarfile.open(fileobj=compressor, mode="w|", format=tarfile.USTAR_FORMAT) as tape:
            for name, expected in sorted(receipt["files"].items()):
                path = ROOT / "raw" / seed / (name + ".pt")
                content = path.read_bytes()
                assert hashlib.sha256(content).hexdigest() == expected
                member = "raw/" + seed + "/" + name + ".pt"
                item = tarfile.TarInfo(member)
                item.size = len(content)
                item.mode = 0o644
                tape.addfile(item, io.BytesIO(content))
                index[member] = expected
    observed = {}
    with tarfile.open(tmp, "r|gz") as tape:
        for member in tape:
            assert member.isfile() and member.name not in observed
            with tape.extractfile(member) as handle:
                observed[member.name] = hashlib.file_digest(handle, "sha256").hexdigest()
    assert observed == index and len(observed) == receipt["rows"] == read(ROOT/"inputs/roster_lock.json")["images"]
    tmp.replace(output)
    return dict(file=output.name, sha256=sha(output), bytes=output.stat().st_size, members=len(observed),
                server_manifest=f"../{seed}_complete.json", server_manifest_sha256=manifest_hash,
                local_validation=f"../{seed}_copy_verified.json",
                local_validation_sha256=sha(ROOT / f"{seed}_copy_verified.json"),
                files=index)


def main(seed):
    entry = archive(seed)
    p = ROOT / "archive/index.json"
    data = read(p) if p.exists() else {}
    data[seed] = entry
    tmp = p.with_suffix(".json.part")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(p)
    print(seed, entry["members"], entry["sha256"], entry["bytes"])


if __name__ == "__main__":
    main(sys.argv[1])
