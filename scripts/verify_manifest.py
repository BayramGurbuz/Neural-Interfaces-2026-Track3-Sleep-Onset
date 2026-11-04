"""Verify a local NEMAR dataset copy against its official manifest.json (size + checksum).

Usage:
    curl -sO https://data.nemar.org/nm000287/v1.0.0/manifest.json
    python scripts/verify_manifest.py manifest.json ~/neuralbench/data/Interaxon2026Muse/download/nm000287
"""
import hashlib
import json
import sys
from pathlib import Path

manifest_path, root = Path(sys.argv[1]), Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text())


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


missing, bad_size, bad_hash, ok = [], [], [], 0
for e in manifest:
    p = root / e["path"]
    if not p.is_file():
        missing.append(e["path"])
        continue
    data = p.read_bytes()
    if len(data) != e["size"]:
        bad_size.append(e["path"])
        continue
    algo = e["checksum_algorithm"]
    digest = git_blob_sha1(data) if algo == "git" else hashlib.new(algo, data).hexdigest()
    if digest != e["checksum"]:
        bad_hash.append(e["path"])
    else:
        ok += 1

listed = {e["path"] for e in manifest}
extra = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and str(p.relative_to(root)) not in listed)

print(f"manifest entries: {len(manifest)} | verified OK: {ok}")
print(f"missing: {len(missing)} {missing[:10]}")
print(f"size mismatch: {len(bad_size)} {bad_size[:10]}")
print(f"checksum mismatch: {len(bad_hash)} {bad_hash[:10]}")
print(f"extra local files not in manifest: {len(extra)} {extra[:10]}")
