"""Freeze actual and permuted image-derived scene hints, keeping each panel's text multiset."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT.parent / "image_only"


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    text = json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        assert path.read_text() == text
    else:
        path.write_text(text)


def top(feedback):
    return feedback.split(" (", 1)[0]


def main():
    rows = read(BASELINE/"inputs/roster.json")
    proof = read(BASELINE/"feedback_verified.json")
    assert set(proof["checks"]) == {r["rgb"] for r in rows}
    arms = read(BASELINE/"inputs/manifest.json")["policies"]
    assignment, counts = {}, {}
    for panel in ("official", "extra"):
        selection = sorted([r for r in rows if r["panel"] == panel], key=lambda r: (top(r["feedback"]),r["rgb"]))
        distribution = Counter(top(r["feedback"]) for r in selection)
        shift = max(distribution.values())
        assert shift < len(selection) / 2
        for i, r in enumerate(selection):
            other = selection[(i+shift) % len(selection)]
            assert proof["checks"][r["rgb"]]["feedback"] == r["feedback"]
            assert top(other["feedback"]) != top(r["feedback"]) and other["rgb"] != r["rgb"]
            assignment[r["rgb"]] = other["rgb"]
        counts[panel] = dict(images=len(selection), shift=shift,
                             different_top=len(selection), category_counts=dict(distribution))
    assert Counter(assignment.values()) == Counter(assignment.keys())
    dump(ROOT/"inputs/assignments.json", assignment)
    dump(ROOT/"inputs/manifest.json", dict(protocol=sha(ROOT/"检验口径.md"), assignment=sha(ROOT/"inputs/assignments.json"),
        source=sha(BASELINE/"inputs/manifest.json"), roster=sha(BASELINE/"inputs/roster.json"),
        verified_tool=sha(BASELINE/"feedback_verified.json"), query=read(BASELINE/"inputs/manifest.json")["query"],
        label="Scene classifier output: ", policies=arms, panels=counts))


if __name__ == "__main__":
    main()
