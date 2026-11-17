"""Stream, verify and restore a complete evidence tar.gz into a fresh directory.

Usage:
    python restore_complete_evidence.py archive.tar.gz NEW_DIRECTORY NEW_PROOF.json
    python restore_complete_evidence.py archive.tar.gz.parts.json NEW_DIRECTORY NEW_PROOF.json

Parts are consumed directly in index order. No joined archive or second copy of
checkpoint files is created. The proof covers byte integrity and archive member
completeness; it does not execute or deserialize the archived experiment files.
"""

import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import stat
import sys
import tarfile
import tempfile


CHUNK = 8 * 1024 * 1024
DEFAULT_JSON_LIMIT = 64 * 1024 * 1024
ROOT_MANIFEST = "file_checksums.json"


class VerificationError(ValueError):
    """Archive, index or destination failed a required verification."""


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def parse_json(data, label):
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=unique_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"Invalid UTF-8 JSON in {label}: {exc}") from exc
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def byte_count(value, label):
    require(type(value) is int and value >= 0, f"Invalid byte count: {label}")
    return value


def digest_value(value, label):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value),
            f"Invalid SHA256: {label}")
    return value.lower()


def member_parts(name):
    require(isinstance(name, str) and name, "Empty or non-string member path")
    require("\x00" not in name and "\\" not in name,
            f"Unsupported member path: {name!r}")
    require(not name.startswith("/") and not PureWindowsPath(name).drive,
            f"Absolute or drive-qualified member path: {name!r}")
    parts = name.split("/")
    require(all(part not in ("", ".", "..") for part in parts),
            f"Non-canonical or traversing member path: {name!r}")
    return parts


def absolute_path(path):
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def new_output(path, label):
    path = absolute_path(path)
    require(not os.path.lexists(path), f"{label} already exists: {path}")
    parent = path.parent.resolve(strict=True)
    require(parent.is_dir(), f"{label} parent is not a directory: {parent}")
    return parent / path.name


def open_regular(path):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode), f"Source is not a regular file: {path}")
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


def read_small(path, limit):
    with open_regular(path) as source:
        require(os.fstat(source.fileno()).st_size <= limit, f"JSON exceeds limit: {path}")
        data = source.read(limit + 1)
    require(len(data) <= limit, f"JSON exceeds limit: {path}")
    return data


class ArchiveStream(io.RawIOBase):
    """Sequential compressed bytes, simultaneously hashing the archive and parts."""

    def __init__(self, segments, expected_bytes, expected_sha256=None):
        super().__init__()
        self.segments = segments
        self.expected_bytes = expected_bytes
        self.expected_sha256 = expected_sha256
        self.total_bytes = 0
        self.total_hash = hashlib.sha256()
        self.position = 0
        self.current = None
        self.current_bytes = 0
        self.current_hash = None
        self.part_results = []
        self.finished = False

    def readable(self):
        return True

    def _open_next(self):
        if self.position == len(self.segments):
            return False
        segment = self.segments[self.position]
        self.current = open_regular(segment["path"])
        require(os.fstat(self.current.fileno()).st_size == segment["bytes"],
                f"Source size changed or disagrees with index: {segment['path']}")
        self.current_bytes = 0
        self.current_hash = hashlib.sha256()
        return True

    def _finish_part(self):
        segment = self.segments[self.position]
        actual_sha = self.current_hash.hexdigest()
        require(self.current_bytes == segment["bytes"], f"Part size mismatch: {segment['path']}")
        if segment.get("sha256") is not None:
            require(actual_sha == segment["sha256"], f"Part SHA256 mismatch: {segment['path']}")
        self.part_results.append({
            "name": segment["path"].name,
            "bytes": self.current_bytes,
            "sha256": actual_sha,
            "expected_bytes": segment["bytes"],
            "expected_sha256": segment.get("sha256"),
            "size_verified": True,
            "sha256_verified_against_index": segment.get("sha256") is not None,
        })
        self.current.close()
        self.current = None
        self.position += 1

    def read(self, size=-1):
        require(size >= 0, "Unbounded compressed source reads are disabled")
        if not size:
            return b""
        wanted = min(size, CHUNK)
        chunks = []
        remaining = wanted
        while self.current is not None or self._open_next():
            data = self.current.read(remaining)
            if data:
                self.current_bytes += len(data)
                require(self.current_bytes <= self.segments[self.position]["bytes"],
                        f"Source grew during restore: {self.segments[self.position]['path']}")
                self.current_hash.update(data)
                self.total_bytes += len(data)
                self.total_hash.update(data)
                chunks.append(data)
                remaining -= len(data)
                if not remaining:
                    break
                continue
            self._finish_part()
        # Fill reads across part boundaries (even inside the gzip magic/header).
        return b"".join(chunks)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def finish(self):
        while self.read(CHUNK):
            pass
        require(self.total_bytes == self.expected_bytes, "Combined archive size mismatch")
        if self.expected_sha256 is not None:
            require(self.total_hash.hexdigest() == self.expected_sha256,
                    "Combined archive SHA256 mismatch")
        self.finished = True

    def close(self):
        if self.current is not None:
            self.current.close()
            self.current = None
        super().close()


def prepare_source(source, source_format, json_limit, expected_sha256):
    if source_format == "auto":
        source_format = "parts" if source.name.endswith(".parts.json") else "archive"
    if source_format == "archive":
        with open_regular(source) as stream:
            size = os.fstat(stream.fileno()).st_size
        return ArchiveStream([{"path": source, "bytes": size}], size, expected_sha256), {
            "kind": "archive", "input_path": str(source), "archive_name": source.name,
            "external_expected_sha256": expected_sha256,
        }

    index_bytes = read_small(source, json_limit)
    index = parse_json(index_bytes, "parts index")
    name = index.get("archive_name")
    require(len(member_parts(name)) == 1, "archive_name must be a basename")
    expected_bytes = byte_count(index.get("archive_bytes"), "archive_bytes")
    index_hash = digest_value(index.get("archive_sha256"), "archive_sha256")
    if expected_sha256 is not None:
        require(index_hash == expected_sha256, "Index and external expected archive SHA256 disagree")
    rows = index.get("parts")
    require(isinstance(rows, list) and rows, "Parts index must have a nonempty parts list")
    seen = set()
    segments = []
    for row in rows:
        require(isinstance(row, dict), "Each part must be a JSON object")
        part_name = row.get("name")
        require(len(member_parts(part_name)) == 1, "Part name must be a basename")
        require(part_name not in seen, f"Duplicate part name: {part_name!r}")
        seen.add(part_name)
        segments.append({
            "path": source.parent / part_name,
            "bytes": byte_count(row.get("bytes"), part_name),
            "sha256": digest_value(row.get("sha256"), part_name),
        })
    require(sum(row["bytes"] for row in segments) == expected_bytes,
            "Sum of indexed part sizes disagrees with archive_bytes")
    return ArchiveStream(segments, expected_bytes, index_hash), {
        "kind": "evidence_parts_index", "input_path": str(source), "archive_name": name,
        "index_bytes": len(index_bytes), "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "indexed_archive_sha256": index_hash, "external_expected_sha256": expected_sha256,
    }


def open_destination(root_fd, parts):
    """Create parent directories and a fresh regular file, refusing symlinks."""
    current_fd = os.dup(root_fd)
    try:
        for component in parts[:-1]:
            try:
                os.mkdir(component, mode=0o700, dir_fd=current_fd)
            except FileExistsError:
                pass
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        fd = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=current_fd)
        return os.fdopen(fd, "wb")
    finally:
        os.close(current_fd)


def verify_manifest(data, actual):
    manifest = parse_json(data, ROOT_MANIFEST)
    expected_names = set(manifest)
    for name in expected_names:
        member_parts(name)
    require(ROOT_MANIFEST not in expected_names, "Root checksum manifest contains its own hash")
    actual_names = set(actual)
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - (expected_names | {ROOT_MANIFEST}))
    require(not missing and not extra,
            f"Member set mismatch: missing={missing[:10]!r}; unregistered={extra[:10]!r}")
    comparisons = {}
    for name, row in manifest.items():
        require(isinstance(row, dict), f"Invalid root manifest entry: {name!r}")
        expected_bytes = byte_count(row.get("bytes"), name)
        expected_sha256 = digest_value(row.get("sha256"), name)
        observed = actual[name]
        require(observed["bytes"] == expected_bytes, f"Member byte count mismatch: {name!r}")
        require(observed["sha256"] == expected_sha256, f"Member SHA256 mismatch: {name!r}")
        comparisons[name] = {
            "expected_bytes": expected_bytes, "actual_bytes": observed["bytes"],
            "expected_sha256": expected_sha256, "actual_sha256": observed["sha256"],
            "bytes_match": True, "sha256_match": True,
        }
    return comparisons


def write_new_proof(path, proof):
    # Atomic publication using a hard link also refuses any existing proof path.
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(proof, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def restore(args):
    started = utc_now()
    source = absolute_path(args.input)
    require(not source.is_symlink(), f"Source symlink is unsupported: {source}")
    source = source.resolve(strict=True)
    target = new_output(args.destination, "Destination")
    proof_path = new_output(args.proof, "Proof")
    require(proof_path != target and target not in proof_path.parents,
            "Proof must be outside the restored directory")
    require(args.max_json_bytes > 0, "max-json-bytes must be positive")
    expected_sha = digest_value(args.expected_sha256, "expected-sha256") if args.expected_sha256 else None
    stream, source_info = prepare_source(source, args.input_format, args.max_json_bytes, expected_sha)
    created = False
    root_fd = None
    succeeded = False
    try:
        target.mkdir(mode=0o700)
        created = True
        root_fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        actual = {}
        manifest_bytes = None
        with gzip.GzipFile(fileobj=stream, mode="rb") as compressed:
            # ignore_zeros also scans members after tar terminator blocks, so a
            # concatenated hidden member is subject to the same checks.
            with tarfile.open(fileobj=compressed, mode="r|", ignore_zeros=True) as archive:
                for member in archive:
                    parts = member_parts(member.name)
                    require(member.type in (tarfile.REGTYPE, tarfile.AREGTYPE)
                            and not member.issparse(),
                            f"Non-regular or sparse member: {member.name!r}")
                    require(member.name not in actual, f"Duplicate archive member: {member.name!r}")
                    require(type(member.size) is int and member.size >= 0,
                            f"Invalid tar member byte count: {member.name!r}")
                    if member.name == ROOT_MANIFEST:
                        require(member.size <= args.max_json_bytes, "Root checksum manifest exceeds JSON limit")
                    member_hash = hashlib.sha256()
                    member_size = 0
                    manifest_chunks = [] if member.name == ROOT_MANIFEST else None
                    payload = archive.extractfile(member)
                    require(payload is not None, f"Unreadable member: {member.name!r}")
                    with payload, open_destination(root_fd, parts) as output:
                        while chunk := payload.read(CHUNK):
                            output.write(chunk)
                            member_hash.update(chunk)
                            member_size += len(chunk)
                            if manifest_chunks is not None:
                                manifest_chunks.append(chunk)
                    require(member_size == member.size, f"Truncated member: {member.name!r}")
                    actual[member.name] = {"bytes": member_size, "sha256": member_hash.hexdigest()}
                    if manifest_chunks is not None:
                        manifest_bytes = b"".join(manifest_chunks)
            # Read through gzip trailers/CRC and every concatenated gzip stream.
            while compressed.read(CHUNK):
                pass
        stream.finish()
        require(manifest_bytes is not None, "Missing root file_checksums.json")
        comparisons = verify_manifest(manifest_bytes, actual)
        require(set(actual) == set(comparisons) | {ROOT_MANIFEST}, "Final member set mismatch")
        root_record = actual[ROOT_MANIFEST]
        source_info.update({
            "archive_bytes": stream.total_bytes, "archive_sha256": stream.total_hash.hexdigest(),
            "archive_size_verified": True,
            "archive_sha256_verified_against_expected": stream.expected_sha256 is not None,
            "parts": stream.part_results if source_info["kind"] == "evidence_parts_index" else [],
        })
        proof = {
            "schema": "cup.complete_evidence_restore.v1", "ok": True,
            "synthetic_tool_test": args.tool_test,
            "verification_scope": "Streaming byte-integrity and complete archive restoration",
            "started_utc": started, "completed_utc": utc_now(),
            "source": source_info, "restored_directory": str(target),
            "archive_members": len(actual), "files_in_root_manifest": len(comparisons),
            "restored_bytes": sum(row["bytes"] for row in actual.values()),
            "all_members_regular_files": True,
            "all_paths_unique_relative_without_dot_components": True,
            "exact_member_set_matches_root_manifest": True,
            "all_registered_file_sizes_and_sha256_match": True,
            "gzip_stream_read_to_eof_and_crc_checked": True,
            "root_manifest": {"path": ROOT_MANIFEST, **root_record,
                              "self_hash_excluded_by_archive_format": True},
            "nested_same_name_checksum_manifests_retained": sorted(
                name for name in comparisons if name.split("/")[-1] == ROOT_MANIFEST),
            "files": comparisons,
        }
        write_new_proof(proof_path, proof)
        succeeded = True
        return {"ok": True, "archive_bytes": stream.total_bytes,
                "archive_sha256": stream.total_hash.hexdigest(), "archive_members": len(actual),
                "restored_directory": str(target), "proof": str(proof_path),
                "synthetic_tool_test": args.tool_test}
    finally:
        if root_fd is not None:
            os.close(root_fd)
        stream.close()
        if created and not succeeded:
            shutil.rmtree(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Complete .tar.gz or evidence_parts .parts.json")
    parser.add_argument("destination", type=Path, help="Fresh extraction directory; parent must exist")
    parser.add_argument("proof", type=Path, help="Fresh proof JSON outside destination; parent must exist")
    parser.add_argument("--input-format", choices=("auto", "archive", "parts"), default="auto")
    parser.add_argument("--expected-sha256", help="Optional independent expected compressed archive SHA256")
    parser.add_argument("--max-json-bytes", type=int, default=DEFAULT_JSON_LIMIT)
    parser.add_argument("--tool-test", action="store_true", help="Mark a synthetic tool test in the proof")
    args = parser.parse_args()
    try:
        result = restore(args)
    except (OSError, EOFError, tarfile.TarError, VerificationError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
