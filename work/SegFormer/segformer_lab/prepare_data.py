"""Stage 3: import VisA PCB1 and write the fixed split manifests.

What this module does
---------------------
1. Reads the official ``2cls_highshot.csv`` straight out of the code ZIP,
   verifies its SHA-256, and keeps a byte-identical copy in ``manifests/``.
2. Extracts only ``pcb1/`` from the dataset TAR into the project resource root,
   preserving the official ``pcb1/Data/...`` relative paths. Every member path
   is validated before writing: absolute paths, drive letters, ``..``, links
   and alternate data streams are rejected, and the resolved target must stay
   inside the data root. Existing files are verified by SHA-256 instead of
   being overwritten.
3. Splits only the official ``train`` rows into train/val with a fixed quota
   (120 normal + 12 anomaly) and one ``numpy.random.default_rng(42)``; the
   official ``test`` rows are copied unchanged.
4. Writes ``train.csv`` / ``val.csv`` / ``test.csv`` (columns
   ``object,split,label,image,mask``), ``split_meta.json`` and
   ``<data-root>/extraction_manifest.json``.

Everything is processed row by row and member by member: no image is decoded
into memory, and no archive is loaded whole. Only image headers are opened to
confirm that a file is a readable image of the expected size.

Label convention handled here (storage only, no conversion)
-----------------------------------------------------------
``label`` is ``normal`` or ``anomaly``. Normal rows carry an empty ``mask``
field on purpose: the dataset layer must build an all-zero mask from the image
size, never join an empty string into a path. Anomaly rows point at a mask PNG
whose stored values may be 1..4; converting those to the training label 1 is a
later stage's job, and this module never edits the mask files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import tarfile
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import __version__, paths

SCHEMA = "segformer-lab/split-meta/1"
GROUP_SCHEMA = "segformer-lab/reviewed-groups/1"
COLUMNS = ("object", "split", "label", "image", "mask")
OFFICIAL_CSV_MEMBER = "spot-diff-main/split_csv/2cls_highshot.csv"
# Recorded in the end-to-end guide after reading the local ZIP on 2026-10-31.
EXPECTED_CSV_SHA256 = "c3331eede15f2da8a75b380d4fbfb24f75ee036ec9a25147eca898da6e972f09"
CHUNK = 1024 * 1024


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    """Publish via an exclusively created sibling; never truncate a link."""
    if path.is_symlink():
        raise SystemExit(f"refusing a symbolic-link output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".part", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


# --------------------------------------------------------------------- utils ----
def sha256_file(path: Path, chunk: int = CHUNK) -> tuple[str, int]:
    """Return (hex digest, byte count) while reading the file in chunks."""
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
            total += len(block)
    return digest.hexdigest(), total


def _sha256_stream(handle: Any, chunk: int = CHUNK) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    while True:
        block = handle.read(chunk)
        if not block:
            break
        digest.update(block)
        total += len(block)
    return digest.hexdigest(), total


# ------------------------------------------------------------------- official ----
@dataclass(frozen=True)
class OfficialRow:
    object: str
    split: str
    label: str
    image: str
    mask: str


def read_official_csv(zip_path: Path, expect_sha256: str | None) -> tuple[bytes, str, list[OfficialRow]]:
    """Load the official CSV from the code ZIP without unpacking the ZIP."""
    with zipfile.ZipFile(zip_path) as archive:
        names = [info.filename for info in archive.infolist() if info.filename.endswith("2cls_highshot.csv")]
        if OFFICIAL_CSV_MEMBER not in names:
            raise SystemExit(f"official CSV not found in {zip_path}; members ending with 2cls_highshot.csv: {names}")
        raw = archive.read(OFFICIAL_CSV_MEMBER)

    digest = hashlib.sha256(raw).hexdigest()
    if expect_sha256 and digest != expect_sha256:
        raise SystemExit(
            "official CSV checksum mismatch.\n"
            f"  expected {expect_sha256}\n  actual   {digest}\n"
            "The upstream split may have changed; inspect the file before continuing."
        )

    text = raw.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != COLUMNS:
        raise SystemExit(f"unexpected CSV header: {reader.fieldnames}")

    rows: list[OfficialRow] = []
    for line_no, item in enumerate(reader, start=2):
        try:
            row = OfficialRow(
                object=item["object"].strip(),
                split=item["split"].strip(),
                label=item["label"].strip(),
                image=item["image"].strip(),
                mask=(item["mask"] or "").strip(),
            )
        except (KeyError, AttributeError) as exc:
            raise SystemExit(f"malformed CSV line {line_no}: {item!r} ({exc})") from exc
        rows.append(row)
    return raw, digest, rows


# ----------------------------------------------------------------- extraction ----
_DRIVE = re.compile(r"^[A-Za-z]:")


def safe_member_target(root: Path, member_name: str) -> Path:
    """Map an archive member name to a path inside *root*, or raise.

    Rejects absolute paths, drive letters, parent traversal, backslash tricks,
    NTFS alternate data streams and anything that resolves outside the root.
    """
    name = member_name.replace("\\", "/")
    if name.startswith("/") or _DRIVE.match(name):
        raise ValueError(f"absolute member path: {member_name!r}")
    parts = [part for part in name.split("/") if part not in ("", ".")]
    if not parts:
        raise ValueError(f"empty member path: {member_name!r}")
    if any(part == ".." for part in parts):
        raise ValueError(f"parent traversal in member path: {member_name!r}")
    if any(":" in part for part in parts):
        raise ValueError(f"alternate data stream in member path: {member_name!r}")

    root_resolved = root.resolve()
    target = (root_resolved / Path(*parts)).resolve()
    try:
        target.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"member escapes data root: {member_name!r} -> {target}") from exc
    return target


def _member_hash(tar: tarfile.TarFile, member: tarfile.TarInfo) -> tuple[str, int]:
    handle = tar.extractfile(member)
    if handle is None:
        raise SystemExit(f"cannot read member: {member.name}")
    with handle:
        return _sha256_stream(handle)


def extract_category(
    tar_path: Path,
    category: str,
    data_root: Path,
    force_overwrite: bool = False,
    rewrite_identical: bool = False,
) -> dict[str, Any]:
    """Extract every regular file under ``<category>/`` into *data_root*.

    Returns a summary with the per-file list (path, size, sha256) used as
    extraction evidence. Existing files are hashed and compared with the
    archive member; a mismatch aborts instead of overwriting, unless
    *force_overwrite* is set. Byte-identical files are normally skipped;
    *rewrite_identical* only rewrites equal content; repairing a different copy
    still requires the explicit *force_overwrite* flag.
    """
    prefix = f"{category}/"
    files: list[dict[str, Any]] = []
    directories = 0
    skipped = 0
    rejected: list[str] = []
    written_bytes = 0
    started = time.perf_counter()

    with tarfile.open(tar_path, mode="r:") as tar:
        members = [m for m in tar.getmembers() if m.name.startswith(prefix)]
        if not members:
            raise SystemExit(f"no members under {prefix!r} in {tar_path}")

        for index, member in enumerate(members, start=1):
            if member.isdir():
                try:
                    safe_member_target(data_root, member.name).mkdir(parents=True, exist_ok=True)
                    directories += 1
                except ValueError as exc:
                    rejected.append(f"{member.name}: {exc}")
                continue
            if not member.isreg():
                # symlink / hardlink / device: refuse rather than follow.
                rejected.append(f"{member.name}: non-regular member ({member.type!r})")
                continue

            try:
                target = safe_member_target(data_root, member.name)
            except ValueError as exc:
                rejected.append(f"{member.name}: {exc}")
                continue

            if target.exists():
                if target.is_dir():
                    raise SystemExit(f"archive member maps onto an existing directory: {target}")
                existing_sha, existing_size = sha256_file(target)
                member_sha, member_size = _member_hash(tar, member)
                identical = existing_size == member_size and existing_sha == member_sha
                if identical and not rewrite_identical:
                    skipped += 1
                    files.append(
                        {"path": member.name, "size": member_size, "sha256": member_sha, "state": "verified-existing"}
                    )
                    continue
                if not identical and not force_overwrite:
                    raise SystemExit(
                        "existing file differs from the archive and was NOT overwritten:\n"
                        f"  {target}\n"
                        f"  existing size={existing_size} sha256={existing_sha}\n"
                        f"  archive  size={member_size} sha256={member_sha}\n"
                        "Re-run with --force-overwrite only after deciding which copy is correct."
                    )

            target.parent.mkdir(parents=True, exist_ok=True)
            part = None
            digest = hashlib.sha256()
            size = 0
            handle = tar.extractfile(member)
            if handle is None:
                raise SystemExit(f"cannot read member: {member.name}")
            try:
                with handle, tempfile.NamedTemporaryFile(mode="wb", dir=target.parent, prefix=f".{target.name}.", suffix=".part", delete=False) as out:
                    part = Path(out.name)
                    while True:
                        block = handle.read(CHUNK)
                        if not block:
                            break
                        digest.update(block)
                        size += len(block)
                        out.write(block)
                if size != member.size:
                    raise SystemExit(f"short read for {member.name}: {size} != {member.size}")
                part.replace(target)
            finally:
                if part is not None:
                    part.unlink(missing_ok=True)
            written_bytes += size
            files.append({"path": member.name, "size": size, "sha256": digest.hexdigest(), "state": "extracted"})

            if index % 200 == 0:
                print(f"  ... {index}/{len(members)} members processed", flush=True)

    if rejected:
        raise SystemExit("refused archive members:\n  " + "\n  ".join(rejected))

    elapsed = time.perf_counter() - started
    summary = {
        "schema": "segformer-lab/extraction-manifest/1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "archive": {"path": str(tar_path), "size_bytes": tar_path.stat().st_size},
        "category": category,
        "data_root": str(data_root),
        "members_matched": len(members),
        "files_written": sum(1 for f in files if f["state"] == "extracted"),
        "files_verified_existing": skipped,
        "directories": directories,
        "bytes_written": written_bytes,
        "elapsed_seconds": round(elapsed, 2),
        "files": files,
    }
    return summary


# ---------------------------------------------------------------------- split ----
def _sort_key(row: OfficialRow) -> str:
    return row.image


def _split_inputs(
    rows: list[OfficialRow], category: str, val_normal: int, val_anomaly: int,
) -> tuple[list[OfficialRow], list[OfficialRow]]:
    """Validate inputs without drawing randomness for either split version."""
    category_rows = [row for row in rows if row.object == category]
    if not category_rows:
        raise SystemExit(f"no rows for object == {category!r}")

    unknown_label = {row.label for row in category_rows} - {"normal", "anomaly"}
    if unknown_label:
        raise SystemExit(f"unexpected label values: {sorted(unknown_label)}")
    unknown_split = {row.split for row in category_rows} - {"train", "test"}
    if unknown_split:
        raise SystemExit(f"unexpected split values: {sorted(unknown_split)}")

    official_train = [row for row in category_rows if row.split == "train"]
    official_test = [row for row in category_rows if row.split == "test"]

    normal_count = sum(r.label == "normal" for r in official_train)
    anomaly_count = sum(r.label == "anomaly" for r in official_train)
    if val_normal < 0 or val_anomaly < 0:
        raise SystemExit("validation quotas must be non-negative")
    if val_normal > normal_count or val_anomaly > anomaly_count:
        raise SystemExit(
            f"quota exceeds available rows: normal {val_normal}/{normal_count}, "
            f"anomaly {val_anomaly}/{anomaly_count}"
        )
    return official_train, official_test


def build_split(
    rows: list[OfficialRow],
    category: str,
    seed: int,
    val_normal: int,
    val_anomaly: int,
) -> tuple[dict[str, list[OfficialRow]], dict[str, Any]]:
    """Split official train rows into train/val; keep official test as-is."""
    official_train, official_test = _split_inputs(rows, category, val_normal, val_anomaly)
    train_normal = sorted([r for r in official_train if r.label == "normal"], key=_sort_key)
    train_anomaly = sorted([r for r in official_train if r.label == "anomaly"], key=_sort_key)

    rng = np.random.default_rng(seed)
    # Order matters: normal first, then anomaly, from the same generator.
    perm_normal = rng.permutation(len(train_normal))
    perm_anomaly = rng.permutation(len(train_anomaly))

    val_normal_rows = [train_normal[i] for i in perm_normal[:val_normal]]
    val_anomaly_rows = [train_anomaly[i] for i in perm_anomaly[:val_anomaly]]
    val_images = {row.image for row in val_normal_rows} | {row.image for row in val_anomaly_rows}

    train_rows = [row for row in official_train if row.image not in val_images]

    derived = {
        "train": sorted(train_rows, key=_sort_key),
        "val": sorted(val_normal_rows + val_anomaly_rows, key=_sort_key),
        "test": sorted(official_test, key=_sort_key),
    }
    algorithm = {
        "seed": seed,
        "numpy_version": np.__version__,
        "generator": "numpy.random.default_rng(seed)",
        "procedure": (
            "official train rows grouped by label; each group sorted by image path; "
            "one default_rng(seed) draws permutation(group_normal) first, then "
            "permutation(group_anomaly); the first val_normal / val_anomaly indices "
            "become validation rows; remaining official train rows stay train; "
            "official test rows are copied unchanged"
        ),
        "quota": {"val_normal": val_normal, "val_anomaly": val_anomaly},
    }
    return derived, algorithm


def _validate_groups(groups: Any, rows: list[OfficialRow], category: str) -> list[dict[str, Any]]:
    if not isinstance(groups, list):
        raise SystemExit("reviewed groups must be a JSON list")
    category_rows = [row for row in rows if row.object == category]
    by_image = {row.image: row for row in category_rows}
    if len(by_image) != len(category_rows):
        raise SystemExit("official category rows contain duplicate image paths")
    seen_members: set[str] = set()
    seen_ids: set[str] = set()
    validated = []
    for group in groups:
        if not isinstance(group, dict):
            raise SystemExit("each reviewed group must be a JSON object")
        group_id = group.get("id")
        label = group.get("label")
        members = group.get("members")
        if not isinstance(group_id, str) or not group_id.strip() or group_id in seen_ids:
            raise SystemExit(f"reviewed group id must be non-empty and unique: {group_id!r}")
        if label not in ("normal", "anomaly"):
            raise SystemExit(f"unexpected reviewed group label: {label!r}")
        if not isinstance(members, list) or len(members) < 2 or not all(isinstance(m, str) for m in members):
            raise SystemExit(f"group {group_id!r} needs at least two image-path strings")
        if len(set(members)) != len(members) or seen_members.intersection(members):
            raise SystemExit(f"reviewed group members must be unique within and across groups: {group_id!r}")
        for member in members:
            row = by_image.get(member)
            if row is None:
                raise SystemExit(f"unknown reviewed group member for {category}: {member}")
            if row.split != "train":
                raise SystemExit(f"reviewed groups may contain only official-training images: {member}")
            if row.label != label:
                raise SystemExit(f"reviewed group members must all have label {label!r}: {member}")
        seen_ids.add(group_id)
        seen_members.update(members)
        validated.append({**group, "members": sorted(members)})
    return sorted(validated, key=lambda group: group["members"][0])


def read_groups(path: Path, official_rows: list[OfficialRow], category: str = "pcb1") -> list[dict[str, Any]]:
    """Read reviewed same-object risk groups; reject test/unknown/mixed members."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read reviewed groups {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema") != GROUP_SCHEMA:
        raise SystemExit(f"reviewed groups must use schema {GROUP_SCHEMA!r}")
    return _validate_groups(document.get("groups"), official_rows, category)


def check_group_splits(derived: dict[str, list[OfficialRow]], groups: list[dict[str, Any]]) -> dict[str, Any]:
    """Check saved memberships, independently of the assignment algorithm."""
    locations: dict[str, set[str]] = {}
    for split, members in derived.items():
        for row in members:
            locations.setdefault(row.image, set()).add(split)
    missing = []
    crossed = []
    for group in groups:
        splits: set[str] = set()
        for member in group["members"]:
            if member not in locations:
                missing.append({"group_id": group["id"], "image": member})
            splits.update(locations.get(member, set()))
        if len(splits) > 1:
            crossed.append({"group_id": group["id"], "splits": sorted(splits), "members": group["members"]})
    return {"groups_checked": len(groups), "cross_split_groups": crossed,
            "missing_members": missing, "ok": not crossed and not missing}


def build_group_split(
    rows: list[OfficialRow], category: str, seed: int, val_normal: int, val_anomaly: int,
    groups: list[dict[str, Any]],
) -> tuple[dict[str, list[OfficialRow]], dict[str, Any]]:
    """V2: permute indivisible groups, then greedily fill the exact val quotas.

    Ungrouped images are singleton groups. Groups are sorted by their minimum
    image path; one RNG permutes normal groups first, then anomaly groups.
    A group is taken only if it fits the remaining quota. Failure to reach the
    exact quota aborts; no member is peeled off and no test image is moved.
    """
    official_train, official_test = _split_inputs(rows, category, val_normal, val_anomaly)
    validated = _validate_groups(groups, rows, category)
    by_image = {row.image: row for row in official_train}
    grouped_images = {member for group in validated for member in group["members"]}
    rng = np.random.default_rng(seed)
    val_rows: list[OfficialRow] = []
    for label, quota in (("normal", val_normal), ("anomaly", val_anomaly)):
        buckets = [[by_image[member] for member in group["members"]]
                   for group in validated if group["label"] == label]
        buckets += [[row] for row in official_train if row.label == label and row.image not in grouped_images]
        buckets.sort(key=lambda bucket: min(row.image for row in bucket))
        remaining = quota
        for index in rng.permutation(len(buckets)):
            bucket = buckets[index]
            if len(bucket) <= remaining:
                val_rows.extend(bucket)
                remaining -= len(bucket)
        if remaining:
            raise SystemExit(f"indivisible reviewed groups cannot fill exact validation quota for {label}: "
                             f"requested {quota}, short by {remaining}")
    val_images = {row.image for row in val_rows}
    derived = {"train": sorted([r for r in official_train if r.image not in val_images], key=_sort_key),
               "val": sorted(val_rows, key=_sort_key), "test": sorted(official_test, key=_sort_key)}
    algorithm = {
        "protocol": {"name": "pcb1-highshot", "version": 2, "grouped": True},
        "seed": seed, "numpy_version": np.__version__, "generator": "numpy.random.default_rng(seed)",
        "procedure": (
            "official train only; reviewed same-label groups are indivisible and all other images "
            "are singleton groups; sort groups by minimum image path; one default_rng(seed) "
            "permutes normal groups first, then anomaly groups; greedily take a group if it fits "
            "the remaining exact per-label validation quota; fail if a quota cannot be filled; "
            "remaining official train rows stay train; official test rows are copied unchanged"
        ),
        "quota": {"val_normal": val_normal, "val_anomaly": val_anomaly},
        "reviewed_group_count": len(validated),
    }
    return derived, algorithm


# -------------------------------------------------------------------- outputs ----
def _write_manifest(path: Path, rows: list[OfficialRow], split_name: str) -> dict[str, Any]:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(COLUMNS)
    for row in rows:
        writer.writerow([row.object, split_name, row.label, row.image, row.mask])
    content = buffer.getvalue().encode("utf-8")
    if path.is_symlink():
        raise SystemExit(f"refusing a symbolic-link manifest: {path}")
    if path.exists():
        if path.read_bytes() != content:
            raise SystemExit(f"existing manifest differs; not overwriting a fixed split: {path}")
    else:
        _atomic_write_bytes(path, content)
    digest, size = sha256_file(path)
    return {"path": str(path), "sha256": digest, "size_bytes": size, "rows": len(rows)}


def _counts(rows: list[OfficialRow]) -> dict[str, int]:
    normal = sum(1 for r in rows if r.label == "normal")
    return {"normal": normal, "anomaly": len(rows) - normal, "total": len(rows)}


# ---------------------------------------------------------------------- audit ----
def audit(
    derived: dict[str, list[OfficialRow]],
    official_rows: list[OfficialRow],
    category: str,
    data_root: Path,
    expected_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Path, mask and leakage checks over the derived manifests."""
    report: dict[str, Any] = {}
    sets = {name: {row.image for row in rows} for name, rows in derived.items()}

    report["counts"] = {name: _counts(rows) for name, rows in derived.items()}
    report["counts_match_expected"] = report["counts"] == expected_counts

    pairwise = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[a] & sets[b]
        pairwise[f"{a}&{b}"] = len(overlap)
    report["pairwise_image_overlap"] = pairwise
    report["pairwise_disjoint"] = all(value == 0 for value in pairwise.values())

    official_test = {row.image for row in official_rows if row.object == category and row.split == "test"}
    report["test_equals_official_test_set"] = sets["test"] == official_test
    report["test_missing_vs_official"] = len(official_test - sets["test"])
    report["test_extra_vs_official"] = len(sets["test"] - official_test)

    all_images = [row.image for rows in derived.values() for row in rows]
    report["duplicate_image_paths"] = len(all_images) - len(set(all_images))

    mask_paths = [row.mask for rows in derived.values() for row in rows if row.label == "anomaly"]
    report["duplicate_mask_paths"] = len(mask_paths) - len(set(mask_paths))

    missing_images: list[str] = []
    unreadable_images: list[str] = []
    image_sizes: dict[str, int] = {}
    for row in (r for rows in derived.values() for r in rows):
        target = data_root / row.image
        if not target.is_file():
            missing_images.append(row.image)
            continue
        try:
            with Image.open(target) as image:
                image_sizes[row.image] = f"{image.size[0]}x{image.size[1]}"
        except Exception as exc:  # noqa: BLE001 - report, do not abort
            unreadable_images.append(f"{row.image}: {type(exc).__name__}: {exc}")
    report["images_checked"] = len(all_images)
    report["images_missing"] = missing_images[:20]
    report["images_missing_count"] = len(missing_images)
    report["images_unreadable"] = unreadable_images[:20]
    report["images_unreadable_count"] = len(unreadable_images)
    report["image_size_distribution"] = {size: sum(1 for v in image_sizes.values() if v == size)
                                         for size in sorted(set(image_sizes.values()))}

    mask_missing: list[str] = []
    mask_unreadable: list[str] = []
    mask_size_mismatch: list[str] = []
    mask_modes: dict[str, int] = {}
    anomaly_rows = [row for rows in derived.values() for row in rows if row.label == "anomaly"]
    for row in anomaly_rows:
        if not row.mask:
            mask_missing.append(f"{row.image}: empty mask field")
            continue
        target = data_root / row.mask
        if not target.is_file():
            mask_missing.append(row.mask)
            continue
        try:
            with Image.open(target) as mask:
                mask_modes[mask.mode] = mask_modes.get(mask.mode, 0) + 1
                expected = image_sizes.get(row.image)
                if expected and f"{mask.size[0]}x{mask.size[1]}" != expected:
                    mask_size_mismatch.append(f"{row.mask}: {mask.size[0]}x{mask.size[1]} vs image {expected}")
        except Exception as exc:  # noqa: BLE001
            mask_unreadable.append(f"{row.mask}: {type(exc).__name__}: {exc}")
    report["anomaly_rows"] = len(anomaly_rows)
    report["anomaly_masks_checked"] = len(anomaly_rows)
    report["mask_missing"] = mask_missing[:20]
    report["mask_missing_count"] = len(mask_missing)
    report["mask_unreadable"] = mask_unreadable[:20]
    report["mask_unreadable_count"] = len(mask_unreadable)
    report["mask_size_mismatch"] = mask_size_mismatch[:20]
    report["mask_size_mismatch_count"] = len(mask_size_mismatch)
    report["mask_modes"] = mask_modes

    normal_rows = [row for rows in derived.values() for row in rows if row.label == "normal"]
    report["normal_rows"] = len(normal_rows)
    report["normal_rows_with_mask_field"] = sum(1 for row in normal_rows if row.mask)
    report["normal_rows_zero_mask_constructible"] = report["normal_rows_with_mask_field"] == 0

    report["ok"] = all(
        [
            report["counts_match_expected"],
            report["pairwise_disjoint"],
            report["test_equals_official_test_set"],
            report["duplicate_image_paths"] == 0,
            report["images_missing_count"] == 0,
            report["images_unreadable_count"] == 0,
            report["mask_missing_count"] == 0,
            report["mask_unreadable_count"] == 0,
            report["mask_size_mismatch_count"] == 0,
            report["normal_rows_zero_mask_constructible"],
        ]
    )
    return report


# ------------------------------------------------------------------------ main ----
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", required=True, help="VisA TAR path (kept in place, read only)")
    parser.add_argument("--source-zip", required=True, help="code ZIP that contains split_csv/2cls_highshot.csv")
    parser.add_argument("--data-root", default=str(paths.DATA_DIR / "visa"), help="extraction root")
    parser.add_argument("--out", default=str(paths.MANIFESTS_DIR), help="manifest output directory")
    parser.add_argument("--category", default="pcb1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-normal", type=int, default=120)
    parser.add_argument("--val-anomaly", type=int, default=12)
    parser.add_argument("--reviewed-groups", default="", help="explicit reviewed group JSON opts into the version-2 grouped split")
    parser.add_argument("--expect-csv-sha256", default=EXPECTED_CSV_SHA256)
    parser.add_argument("--force-overwrite", action="store_true", help="overwrite files that differ from the archive")
    parser.add_argument(
        "--rewrite-extraction",
        action="store_true",
        help="rewrite byte-identical files; different existing content still requires --force-overwrite",
    )
    parser.add_argument("--skip-extraction", action="store_true", help="only rebuild manifests (audit still checks files)")
    args = parser.parse_args(argv)
    if (args.category, args.seed, args.val_normal, args.val_anomaly) != ("pcb1", 42, 120, 12):
        parser.error("this entry point implements the fixed PCB1 protocol: category=pcb1, seed=42, val-normal=120, val-anomaly=12; use a separately reviewed protocol for alternatives")

    archive = Path(args.archive).expanduser().resolve()
    source_zip = Path(args.source_zip).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    for path, label in ((archive, "archive"), (source_zip, "source zip")):
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}")

    started = time.perf_counter()
    print("SegFormer stage 3: VisA PCB1 preparation")
    print(f"  archive   : {archive} ({archive.stat().st_size / 1024 ** 3:.2f} GiB)")
    print(f"  source zip: {source_zip}")
    print(f"  data root : {data_root}")
    print(f"  manifests : {out_dir}")

    # 1) official CSV, byte-identical copy kept as evidence.
    raw_csv, csv_sha, official_rows = read_official_csv(source_zip, args.expect_csv_sha256)
    category_rows = [row for row in official_rows if row.object == args.category]
    print(f"  official CSV: rows={len(official_rows)} sha256={csv_sha}")
    print(f"  category {args.category}: rows={len(category_rows)}")
    group_path = paths.resolve(args.reviewed_groups) if args.reviewed_groups else None
    reviewed_groups = read_groups(group_path, official_rows, args.category) if group_path else []
    group_source = None
    if group_path:
        group_sha, group_size = sha256_file(group_path)
        recorded_group_path = group_path.relative_to(out_dir) if group_path.is_relative_to(out_dir) else group_path
        group_source = {"path": recorded_group_path.as_posix(), "sha256": group_sha, "size_bytes": group_size,
                        "schema": GROUP_SCHEMA}

    out_dir.mkdir(parents=True, exist_ok=True)
    official_copy = out_dir / "official_2cls_highshot.csv"
    if official_copy.is_symlink():
        raise SystemExit(f"refusing a symbolic-link official CSV: {official_copy}")
    if official_copy.exists():
        existing_sha, _ = sha256_file(official_copy)
        if existing_sha != csv_sha:
            raise SystemExit(f"existing {official_copy} differs from the ZIP copy; not overwriting")
        print(f"  official CSV copy: verified existing ({official_copy.name})")
    else:
        _atomic_write_bytes(official_copy, raw_csv)
        print(f"  official CSV copy: written ({official_copy.name})")

    # 2) extraction.
    if args.skip_extraction:
        print("  extraction : skipped by flag")
        extraction = {"skipped": True}
    else:
        print("  extraction : starting")
        extraction = extract_category(
            archive, args.category, data_root, args.force_overwrite, args.rewrite_extraction
        )
        data_root.mkdir(parents=True, exist_ok=True)
        extraction_path = data_root / "extraction_manifest.json"

        # Keep the record of the run that actually wrote the bytes. A later
        # verification-only run would otherwise replace "1205 written" with
        # "1205 verified", losing the provenance of the extracted copy.
        extraction_baseline: dict[str, Any] | None = None
        if extraction_path.exists():
            try:
                previous = json.loads(extraction_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                previous = {}
            candidates = [
                previous.get("extraction_baseline"),
                {
                    "generated_at": previous.get("generated_at"),
                    "files_written": previous.get("files_written", 0),
                    "bytes_written": previous.get("bytes_written", 0),
                    "elapsed_seconds": previous.get("elapsed_seconds"),
                },
            ]
            for candidate in candidates:
                if candidate and candidate.get("files_written", 0) > 0:
                    extraction_baseline = candidate
                    break
        if extraction_baseline is None:
            extraction_baseline = {
                "generated_at": extraction["generated_at"],
                "files_written": extraction["files_written"],
                "bytes_written": extraction["bytes_written"],
                "elapsed_seconds": extraction["elapsed_seconds"],
            }
        extraction["extraction_baseline"] = extraction_baseline

        _atomic_write_bytes(extraction_path, json.dumps(extraction, indent=2, ensure_ascii=False).encode("utf-8"))
        extraction_sha, _ = sha256_file(extraction_path)
        print(
            f"  extraction : files={extraction['files_written']} "
            f"verified={extraction['files_verified_existing']} dirs={extraction['directories']} "
            f"bytes={extraction['bytes_written']} elapsed={extraction['elapsed_seconds']}s"
        )
        print(
            f"  baseline write: files={extraction_baseline['files_written']} bytes={extraction_baseline['bytes_written']} "
            f"at {extraction_baseline['generated_at']}"
        )
        extraction["manifest_path"] = str(extraction_path)

    # 3) split.
    if group_path:
        derived, algorithm = build_group_split(category_rows, args.category, args.seed, args.val_normal,
                                               args.val_anomaly, reviewed_groups)
    else:
        derived, algorithm = build_split(category_rows, args.category, args.seed, args.val_normal, args.val_anomaly)

    # 4) audit before writing, so a failed check is visible in the report too.
    #    Expected counts are derived from the official CSV, so a different
    #    --val-* quota stays consistent; the guide's fixed protocol numbers are
    #    checked separately below.
    official_train_rows = [row for row in category_rows if row.split == "train"]
    official_test_rows = [row for row in category_rows if row.split == "test"]
    train_normal_total = sum(1 for row in official_train_rows if row.label == "normal")
    train_anomaly_total = sum(1 for row in official_train_rows if row.label == "anomaly")
    test_normal_total = sum(1 for row in official_test_rows if row.label == "normal")
    test_anomaly_total = sum(1 for row in official_test_rows if row.label == "anomaly")
    expected_counts = {
        "train": {
            "normal": train_normal_total - args.val_normal,
            "anomaly": train_anomaly_total - args.val_anomaly,
            "total": train_normal_total + train_anomaly_total - args.val_normal - args.val_anomaly,
        },
        "val": {
            "normal": args.val_normal,
            "anomaly": args.val_anomaly,
            "total": args.val_normal + args.val_anomaly,
        },
        "test": {"normal": test_normal_total, "anomaly": test_anomaly_total,
                 "total": test_normal_total + test_anomaly_total},
    }
    checks = audit(derived, official_rows, args.category, data_root, expected_counts)
    group_check = check_group_splits(derived, reviewed_groups)
    if group_path:
        checks["reviewed_groups_stay_in_one_split"] = group_check["ok"]
        checks["ok"] = checks["ok"] and group_check["ok"]

    # The documented protocol for pcb1: 530 / 132 / 442.
    protocol_expected = {"train": 530, "val": 132, "test": 442}
    protocol_actual = {name: checks["counts"][name]["total"] for name in ("train", "val", "test")}
    protocol_by_label = {
        "train": {"normal": 482, "anomaly": 48, "total": 530},
        "val": {"normal": 120, "anomaly": 12, "total": 132},
        "test": {"normal": 402, "anomaly": 40, "total": 442},
    }
    protocol_check = {
        "expected": protocol_expected,
        "actual": protocol_actual,
        "match": protocol_actual == protocol_expected and checks["counts"] == protocol_by_label,
        "expected_by_label": protocol_by_label,
        "note": "fixed protocol numbers for pcb1 high-shot with val 120 normal + 12 anomaly",
    }

    if not checks["ok"] or not protocol_check["match"]:
        failure_path = out_dir / "preparation-failure.json"
        _atomic_write_bytes(failure_path, json.dumps({"checks": checks, "protocol_counts": protocol_check}, indent=2, ensure_ascii=False).encode("utf-8"))
        print(f"  RESULT: FAILED - fixed manifests unchanged; see {failure_path}", file=sys.stderr)
        return 1

    # 5) Publish only successful fixed manifests, preserving different existing copies.
    manifests = {name: _write_manifest(out_dir / f"{name}.csv", rows, name) for name, rows in derived.items()}
    official_copy_sha, official_copy_size = sha256_file(official_copy)

    meta = {
        "schema": "segformer-lab/split-meta/2" if group_path else SCHEMA,
        "package_version": __version__,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "category": args.category,
        "algorithm": algorithm,
        "counts": checks["counts"],
        "sources": {
            "archive": {
                "path": str(archive),
                "size_bytes": archive.stat().st_size,
                "sha256": sha256_file(archive)[0],
            },
            "source_zip": {"path": str(source_zip), "size_bytes": source_zip.stat().st_size,
                           "sha256": sha256_file(source_zip)[0]},
            "official_csv": {
                "member": OFFICIAL_CSV_MEMBER,
                "copy": str(official_copy),
                "size_bytes": official_copy_size,
                "sha256": official_copy_sha,
            },
        },
        "extraction": {
            "skipped": bool(extraction.get("skipped", False)),
            "data_root": str(data_root),
            "manifest": extraction.get("manifest_path"),
            "manifest_sha256": sha256_file(Path(extraction["manifest_path"]))[0] if extraction.get("manifest_path") else None,
            "files_written": extraction.get("files_written"),
            "files_verified_existing": extraction.get("files_verified_existing"),
            "bytes_written": extraction.get("bytes_written"),
            "extraction_baseline": extraction.get("extraction_baseline"),
        },
        "manifests": manifests,
        "checks": checks,
        "protocol_counts": protocol_check,
        "elapsed_seconds": None,
    }
    if group_path:
        meta["protocol"] = algorithm["protocol"]
        meta["sources"]["reviewed_groups"] = group_source
        meta["reviewed_group_check"] = group_check
    meta["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    meta_path = out_dir / "split_meta.json"
    _atomic_write_bytes(meta_path, json.dumps(meta, indent=2, ensure_ascii=False).encode("utf-8"))

    # 6) console summary.
    print("")
    print("  split counts")
    for name in ("train", "val", "test"):
        c = checks["counts"][name]
        print(f"    {name:<5}: normal={c['normal']:<4} anomaly={c['anomaly']:<3} total={c['total']}")
    print(f"  disjoint      : {checks['pairwise_disjoint']} {checks['pairwise_image_overlap']}")
    print(f"  test == official: {checks['test_equals_official_test_set']}")
    print(f"  images        : checked={checks['images_checked']} missing={checks['images_missing_count']} "
          f"unreadable={checks['images_unreadable_count']} sizes={checks['image_size_distribution']}")
    print(f"  masks         : checked={checks['anomaly_masks_checked']} missing={checks['mask_missing_count']} "
          f"unreadable={checks['mask_unreadable_count']} size_mismatch={checks['mask_size_mismatch_count']} "
          f"modes={checks['mask_modes']}")
    print(f"  normal zero mask constructible: {checks['normal_rows_zero_mask_constructible']}")
    print(f"  protocol 530/132/442: {protocol_check['match']} (actual {protocol_actual})")
    print(f"  checks ok     : {checks['ok']}")
    for name, info in manifests.items():
        print(f"  {name}.csv sha256: {info['sha256']}")
    print(f"  meta          : {meta_path}")
    print(f"  elapsed       : {meta['elapsed_seconds']}s")

    if not checks["ok"] or not protocol_check["match"]:
        print("  RESULT: FAILED - see checks in split_meta.json", file=sys.stderr)
        return 1
    print("  RESULT: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
