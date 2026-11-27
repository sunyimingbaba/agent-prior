"""Store independently verifiable per-image R0 observations and candidate score evidence."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ARCHIVE = ROOT / "archive"


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def make_archive(folder, count):
    dest = ARCHIVE / (folder + ".tar.gz")
    hashes = {}
    files = sorted((RESULTS / folder).rglob("*.json"))
    assert len(files) == count
    with dest.open("wb") as raw, gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as tar:
            for path in files:
                name = path.relative_to(RESULTS).as_posix()
                payload = path.read_bytes()
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
                hashes[name] = sha(payload)
    with tarfile.open(dest) as tar:
        assert hashes == {item.name: sha(tar.extractfile(item).read()) for item in tar}
    return dict(file=dest.name, sha256=sha(dest.read_bytes()), files=hashes)


if __name__ == "__main__":
    ARCHIVE.mkdir(exist_ok=True)
    result = json.loads((RESULTS / "complete.json").read_bytes())
    assert all(sha((RESULTS / name).read_bytes()) == value for name, value in result["files"].items())
    observation = make_archive("visual", 336)
    raw_expected = json.loads((RESULTS / "visual_complete.json").read_bytes())["files"]
    assert observation["files"] == {"visual/" + name + ".json": val for name, val in raw_expected.items()}
    candidate = make_archive("candidates", 218 + 218 + 118)
    manifest = dict(visual=observation, candidates=candidate, results=sha((RESULTS / "complete.json").read_bytes()))
    (ARCHIVE / "SHA256.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
