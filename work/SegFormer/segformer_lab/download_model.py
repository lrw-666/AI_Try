"""Stage 5: pin and download the nvidia/mit-b0 encoder snapshot.

Why a dedicated module instead of a one-off command
---------------------------------------------------
A pretrained encoder is part of the experiment identity. If the snapshot can be
replaced silently, two runs labelled "the same" stop being comparable. So the
download does four things every time, in this order:

1. use the pinned commit (or an explicitly requested revision for a new directory);
2. fetch only the files the experiment needs, at that SHA;
3. reject Git LFS pointer files (tensor parsing belongs to the loading check);
4. write ``source.json`` with model id, revision, per-file size and SHA-256.

Re-running is safe: when ``source.json`` matches the requested revision and all
recorded hashes still verify, nothing is downloaded again.

Offline note: this module touches the network only via ``--revision``
resolution and ``snapshot_download``. Use ``--verify-only`` to check an
existing snapshot without any network access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

MODEL_ID = "nvidia/mit-b0"
PINNED_REVISION = "80983a413c30d36a39c20203974ae7807835e2b4"
SCHEMA = "segformer-lab/model-source/1"

# Exactly what the experiment needs. tf_model.h5 and README are deliberately
# excluded: unused weight formats only add bytes and ambiguity.
ALLOW_PATTERNS = ["config.json", "preprocessor_config.json", "pytorch_model.bin", "model.safetensors"]
WEIGHT_CANDIDATES = ("model.safetensors", "pytorch_model.bin")
METADATA_FILES = ("config.json", "preprocessor_config.json")

CHUNK = 1024 * 1024
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def sha256_file(path: Path, chunk: int = CHUNK) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def is_lfs_pointer(path: Path) -> bool:
    """A Git LFS pointer is a tiny text file; real weights are megabytes."""
    if path.stat().st_size > 1024:
        return False
    with path.open("rb") as handle:
        return handle.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX


def apply_proxy(proxy: str | None) -> None:
    """Force the HTTP(S) proxy inside this process.

    The parent shell on this machine carries both ``HTTP_PROXY`` and
    ``http_proxy`` with different values, and ``conda run`` forwards the stale
    one to the child. Setting the explicit value here (before any Hub session is
    created) is the only reliable way to make the download use the working port.
    """
    if not proxy:
        return
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ[name] = proxy
    for name in ("ALL_PROXY", "all_proxy"):
        os.environ.pop(name, None)
    bypass = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or "127.0.0.1,localhost"
    for name in ("NO_PROXY", "no_proxy"):
        os.environ[name] = bypass


def _snapshot_files(root: Path) -> list[Path]:
    """Files that belong to the snapshot (skip huggingface_hub's bookkeeping)."""
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in {".cache", ".git"} for part in path.relative_to(root).parts):
            continue
        files.append(path)
    return files


def validate_source(source: Any) -> list[str]:
    """Validate schema/1 before trusting paths, file hashes or snapshot identity."""
    problems: list[str] = []
    if not isinstance(source, dict):
        return ["source must be an object"]
    if source.get("schema") != SCHEMA:
        problems.append("invalid source schema")
    if not isinstance(source.get("model_id"), str) or not source["model_id"].strip():
        problems.append("missing model_id")
    if not isinstance(source.get("revision"), str) or not re.fullmatch(r"[0-9a-f]{40}", source["revision"]):
        problems.append("revision must be a lowercase 40-character commit SHA")
    items = source.get("files")
    if not isinstance(items, list) or not items:
        return problems + ["files must be a nonempty list"]
    names: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            problems.append("invalid file entry")
            continue
        name = item.get("path")
        if (not isinstance(name, str) or not name or "\\" in name or ":" in name
                or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                or PurePosixPath(name).as_posix() != name or name not in ALLOW_PATTERNS):
            problems.append("unsafe or unsupported snapshot path")
            continue
        if name in names:
            problems.append(f"duplicate file: {name}")
        names.add(name)
        if type(item.get("size_bytes")) is not int or item["size_bytes"] <= 0:
            problems.append(f"invalid size: {name}")
        if not isinstance(item.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            problems.append(f"invalid sha256: {name}")
    for name in METADATA_FILES:
        if name not in names:
            problems.append(f"metadata not recorded: {name}")
    weight = source.get("weights")
    if weight not in WEIGHT_CANDIDATES or weight not in names:
        problems.append("selected weights must be recorded")
    # Both formats are allowed only when the recorded selection matches loader priority.
    selected = next((name for name in WEIGHT_CANDIDATES if name in names), None)
    if weight != selected:
        problems.append("selected weights disagree with loader priority")
    if "total_bytes" in source and not problems:
        if source["total_bytes"] != sum(item["size_bytes"] for item in items):
            problems.append("total_bytes mismatch")
    return problems


def verify_snapshot(root: Path, source: dict[str, Any]) -> list[str]:
    """Return problems, including unrecorded files and invalid source identity."""
    problems = validate_source(source)
    if problems:
        return problems
    recorded = {item["path"]: item for item in source["files"]}
    for path in _snapshot_files(root):
        name = path.relative_to(root).as_posix()
        if name != "source.json" and name not in recorded:
            problems.append(f"unrecorded file: {name}")
    for name, item in recorded.items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            problems.append(f"unsafe file target: {name}")
            continue
        if not path.is_file():
            problems.append(f"missing file: {name}")
            continue
        size = path.stat().st_size
        if size != item["size_bytes"]:
            problems.append(f"size changed: {name} ({size} != {item['size_bytes']})")
            continue
        if sha256_file(path) != item["sha256"]:
            problems.append(f"sha256 changed: {name}")
        if is_lfs_pointer(path):
            problems.append(f"LFS pointer: {name}")
    if not any((root / name).is_file() for name in WEIGHT_CANDIDATES):
        problems.append("no weight file present")
    for name in METADATA_FILES:
        if not (root / name).is_file():
            problems.append(f"missing metadata: {name}")
    return problems


def resolve_revision(model_id: str, revision: str, token: str | None = None) -> tuple[str, str | None]:
    """Return (commit_sha, last_modified) for *revision* (branch, tag or sha)."""
    from huggingface_hub import HfApi

    try:
        info = HfApi(token=token).model_info(model_id, revision=revision)
    except Exception as error:
        # Exception strings may embed credential-bearing proxy URLs.
        raise SystemExit(f"Hub revision lookup failed ({type(error).__name__}); check connectivity/proxy") from None
    return info.sha, (info.lastModified.isoformat() if info.lastModified else None)


def download(
    model_id: str,
    out_dir: Path,
    revision: str | None = None,
    verify_only: bool = False,
    token: str | None = None,
    source_record: Path | None = None,
) -> dict[str, Any]:
    source_path = out_dir / "source.json"
    def read_record(path: Path) -> dict[str, Any]:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise SystemExit("cannot read source record; refusing to replace snapshot") from None
        if isinstance(record, dict) and "source" in record:
            record = record["source"]
        errors = validate_source(record)
        if errors:
            raise SystemExit(f"invalid source record: {errors}")
        return record

    expected = read_record(Path(source_record)) if source_record is not None else None
    if expected and (expected["model_id"] != model_id or
                     revision not in (None, expected["revision"])):
        raise SystemExit("requested identity differs from source record; choose a new directory and source")
    existing = read_record(source_path) if source_path.is_file() else None
    if existing is not None:
        if existing["model_id"] != model_id:
            raise SystemExit("existing model_id differs; use a new snapshot directory")
        # A recorded branch is an alias for the existing identity, never an update request.
        explicit_sha = isinstance(revision, str) and bool(re.fullmatch(r"[0-9a-f]{40}", revision))
        if ((explicit_sha and revision != existing["revision"])
                or revision not in (None, existing["revision"], existing.get("requested_revision"))):
            raise SystemExit("existing revision differs; use a new snapshot directory")
        problems = verify_snapshot(out_dir, existing)
        if expected:
            if (existing["revision"], existing["weights"]) != (expected["revision"], expected["weights"]):
                problems.append("existing identity differs from source record")
            problems.extend(verify_snapshot(out_dir, expected))
            actual_files = {i["path"]: (i["size_bytes"], i["sha256"]) for i in existing["files"]}
            expected_files = {i["path"]: (i["size_bytes"], i["sha256"]) for i in expected["files"]}
            if actual_files != expected_files:
                problems.append("existing manifest differs from source record")
        if problems and not verify_only:
            raise SystemExit(f"snapshot verification failed; refusing network or overwrite: {problems}")
        return {"mode": "verify-only" if verify_only else "reused", "revision": existing["revision"],
                "problems": problems, "ok": not problems, "source": existing}
    if verify_only:
        raise SystemExit(f"no source.json in {out_dir}; run the download first")
    if _snapshot_files(out_dir):
        raise SystemExit("destination contains untracked snapshot files; use an empty directory")

    revision = expected["revision"] if expected else (revision or PINNED_REVISION)
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        commit, last_modified = revision, expected.get("revision_last_modified") if expected else None
    else:
        commit, last_modified = resolve_revision(model_id, revision, token)
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise SystemExit("Hub did not return a commit SHA")
    print(f"  resolved {model_id}@{revision} -> {commit}")

    from huggingface_hub import snapshot_download
    from huggingface_hub import __version__ as hub_version

    started = time.perf_counter()
    try:
        snapshot_download(
            repo_id=model_id,
            revision=commit,
            local_dir=str(out_dir),
            allow_patterns=[i["path"] for i in expected["files"]] if expected else ALLOW_PATTERNS,
            token=token,
        )
    except Exception as error:
        raise SystemExit(f"Hub snapshot download failed ({type(error).__name__}); check connectivity/proxy") from None
    elapsed = time.perf_counter() - started

    files = []
    for path in _snapshot_files(out_dir):
        if path.name == "source.json":
            continue
        if is_lfs_pointer(path):
            raise SystemExit(f"refusing LFS pointer, weights were not fetched: {path}")
        files.append(
            {
                "path": path.relative_to(out_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise SystemExit(f"snapshot is empty: {out_dir}")

    weights = [name for name in WEIGHT_CANDIDATES if any(i["path"] == name for i in files)]
    if not weights:
        raise SystemExit(f"no weight file among {[i['path'] for i in files]}")

    source = {
        "schema": SCHEMA,
        "model_id": model_id,
        "requested_revision": revision,
        "revision": commit,
        "revision_last_modified": last_modified,
        "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "local_dir": str(out_dir),
        "allow_patterns": ALLOW_PATTERNS,
        "huggingface_hub_version": hub_version,
        "download_seconds": round(elapsed, 2),
        "files": files,
        "weights": weights[0],
        "total_bytes": sum(item["size_bytes"] for item in files),
    }
    problems = verify_snapshot(out_dir, source)
    if expected:
        problems.extend(verify_snapshot(out_dir, expected))
    if problems:
        raise SystemExit(f"download verification failed; source.json not written: {problems}")
    tmp = source_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(source, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(source_path)
    return {"mode": "downloaded", "revision": commit, "problems": [], "ok": True, "source": source}


def main(argv: list[str] | None = None) -> int:
    from .paths import MODELS_DIR
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--out", default=str(MODELS_DIR / "mit-b0"))
    parser.add_argument("--revision", default=None, help="new directory: explicit branch/tag/SHA; default pinned commit")
    parser.add_argument("--source-record", type=Path, help="trusted schema/1 source JSON or download report for exact replay")
    parser.add_argument("--verify-only", action="store_true", help="re-check an existing snapshot offline")
    parser.add_argument("--proxy", default="", help="explicit HTTP(S) proxy for this process only")
    parser.add_argument("--report", default="", help="optional JSON report path")
    args = parser.parse_args(argv)

    apply_proxy(args.proxy)

    out_dir = Path(args.out).expanduser().resolve()
    if not args.verify_only:
        out_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None

    print("SegFormer stage 5: model snapshot")
    print(f"  model : {args.model_id}")
    print(f"  out   : {out_dir}")
    print("  proxy : explicit process override" if args.proxy else "  proxy : inherited process settings")

    result = download(args.model_id, out_dir, revision=args.revision, verify_only=args.verify_only,
                      token=token, source_record=args.source_record)

    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  mode  : {result['mode']}")
    print(f"  commit: {result['revision']}")
    if result["problems"]:
        print(f"  RESULT: FAILED - {result['problems']}", file=sys.stderr)
        return 1
    source = result.get("source", {})
    for item in source.get("files", []):
        print(f"    {item['path']:<26} {item['size_bytes']:>10} bytes  {item['sha256'][:16]}…")
    print(f"  total : {source.get('total_bytes')} bytes")
    print("  RESULT: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
