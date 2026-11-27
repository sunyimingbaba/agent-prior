"""Verify the CPU readout files from the server and the matching archive members."""
from pathlib import Path
import gzip
import hashlib
import io
import json
import tarfile

from verify_copy import ROOT, read, sha


def main():
    proof_path = ROOT / "scoring_complete.json"
    proof = read(proof_path)
    sample = read(ROOT / "c_complete.json")
    assert proof["sampling"] == sha(ROOT / "c_complete.json")
    assert proof["code"] == sha(ROOT / "code/score.py")
    assert proof["compiled"] == sha(ROOT / "inputs/manifest.json")
    assert set(proof["files"]) == set(proof["readouts"]) == set(sample["files"])
    names = sorted(proof["files"])
    files = {}
    for name in names:
        path = ROOT / "scores" / (name + ".json")
        record = read(path)
        assert record["source"] == sample["files"][name] and record["projections"] == proof["readouts"][name]
        assert record["eligible"] == sample["eligibility"][name]
        if record["eligible"]:
            assert record["all_readouts_verified"] and len(record["items"]) == 8
        files["scores/" + name + ".json"] = proof["files"][name]
        files["readouts/" + name + ".pt"] = proof["readouts"][name]
    archive = ROOT / "archive/scored.tar.gz"
    tmp = archive.with_suffix(".gz.part")
    with tmp.open("wb") as handle, gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.USTAR_FORMAT) as dest:
            for rel, sig in sorted(files.items()):
                raw = (ROOT / rel).read_bytes()
                assert hashlib.sha256(raw).hexdigest() == sig
                member = tarfile.TarInfo(rel)
                member.size = len(raw)
                dest.addfile(member, io.BytesIO(raw))
    hashes = {}
    with tarfile.open(tmp, "r|gz") as source:
        for member in source:
            assert member.isfile() and member.name not in hashes
            with source.extractfile(member) as stream:
                hashes[member.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    assert files == hashes
    tmp.replace(archive)
    index = dict(archive=archive.name, sha256=sha(archive), members=hashes, scoring_receipt=sha(proof_path))
    (archive.parent / "scored_index.json").write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    main()
