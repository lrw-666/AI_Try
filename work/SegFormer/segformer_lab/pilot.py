"""Stage 10: three full epochs over the whole 530/132 split (a pilot, not a baseline).

Why a separate entry point instead of a flag on ``train.py``: stage 09's entry
point is *defined* by an eight-image subset and ``flip: none``. Relaxing those
guards to fit a pilot would quietly remove the boundary that makes stage 09's
result meaningful. This module therefore owns only the *control flow* that stage
09 does not have -- epochs, augmentation epochs, a warmup+cosine schedule and
per-epoch full-validation bookkeeping -- and calls the same
``training.train_step`` / ``training.step_optimizer`` / ``training.WarmupCosineLR``
and the same ``data`` / ``losses`` / ``metrics`` / ``evaluation`` code.

    python -m segformer_lab.pilot --config configs/pilot.yaml --out <run>
    python -m segformer_lab.pilot --verify-run <run> --config configs/pilot.yaml

Accounting this module insists on:

* one *attempt* per batch; an attempt becomes a *successful update* only when
  AdamW really ran (``step_optimizer`` proves it);
* the scheduler advances only on a real update, so an AMP skip cannot push the
  learning rate forward;
* every epoch must cover all 530 training images exactly once (``drop_last`` is
  false and the coverage is asserted, not assumed);
* per-epoch loss is aggregated with the right weights (CE by valid pixels, Dice
  by positive-GT images), never as a mean of batch means;
* validation is the full 132-image split, B1, IEEE FP32, original space;
* no 132-image probability cache is held in RAM: evaluation streams one sample
  at a time and only the six fixed preview images keep their maps.

No resume support: checkpoints carry optimizer/scaler/scheduler/epoch/RNG state,
but a restart has not been exercised, so this module does not pretend it works.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
import random
import sys
import traceback
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import data, paths
from .config import load_config, write_resolved_config
from .losses import loss_definition
from .metrics import restore_defect_probability, score_definition
from .modeling import load_all
from .smoke import _eval_summary, _labels_of, _render_samples
from .training import (
    WarmupCosineLR,
    build_optimizer,
    evaluate_dataset,
    gpu_memory,
    parameter_digest,
    set_seed,
    sha256_file,
    train_step,
    trainable_parameter_summary,
    _memory_status,
)

SCHEMA = "segformer-lab/pilot/1"
CHECKPOINT_SCHEMA = "segformer-lab/pilot-checkpoint/1"
NOT_A_BASELINE = True


# ------------------------------------------------------------------- helpers ----
def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _jsonl_append(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def _git_identity() -> dict[str, Any]:
    import subprocess

    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=str(paths.PROJECT_ROOT),
                                  capture_output=True, text=True, timeout=30).stdout
        except Exception as exc:  # pragma: no cover - diagnostics only
            return f"<error: {exc}>"

    diff = run("diff")
    return {
        "head": run("rev-parse", "HEAD").strip(),
        "status_porcelain": run("status", "--porcelain").strip(),
        "diff_sha256": sha256_file_text(diff),
        "diff_text": diff,
        "diff_bytes": len(diff.encode("utf-8")),
    }


def sha256_file_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _code_identity() -> dict[str, str]:
    files = [
        "segformer_lab/training.py", "segformer_lab/pilot.py", "segformer_lab/train.py",
        "segformer_lab/smoke.py", "segformer_lab/subset.py", "segformer_lab/losses.py",
        "segformer_lab/metrics.py", "segformer_lab/evaluation.py", "segformer_lab/data.py",
        "segformer_lab/modeling.py", "segformer_lab/config.py", "segformer_lab/geometry.py",
        "segformer_lab/run_check.py",
    ]
    out: dict[str, str] = {}
    for name in files:
        path = paths.PROJECT_ROOT / name
        out[name] = sha256_file(path) if path.is_file() else ""
    return out


def freeze_identity(config: dict[str, Any], *, label: str) -> dict[str, Any]:
    """Snapshot everything that would invalidate the run if it changed."""
    payload = {
        "label": label,
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {"path": str(Path(config["path"]).resolve()),
                   "sha256": sha256_file(Path(config["path"]))},
        "manifests": {split: {"path": str(config["paths"][f"data.{split}_manifest"]),
                              "sha256": sha256_file(Path(config["paths"][f"data.{split}_manifest"]))}
                      for split in ("train", "val", "test")},
        "split_meta_sha256": sha256_file(Path(config["paths"]["data.split_meta"])),
        "preview_ids_sha256": sha256_file(Path(config["paths"]["logging.fixed_preview_manifest"])),
        "train_ids_file": None,
        "code": _code_identity(),
        "git": _git_identity(),
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "cuda_build": torch.version.cuda},
    }
    return payload


def _manifest_of(run_dir: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            files[path.relative_to(run_dir).as_posix()] = {
                "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {"run_dir": str(run_dir.resolve()), "files": files, "file_count": len(files)}


def config_failures(config: dict[str, Any]) -> list[str]:
    """Everything that would make a pilot run meaningless if it were wrong."""
    raw = config["raw"]
    checks = {
        "train.flip must be 'random'": raw["train"].get("flip") == "random",
        "eval.precision must be 'ieee_fp32'": raw["eval"].get("precision") == "ieee_fp32",
        "train.amp must be true": bool(raw["train"].get("amp")),
        "data.train_ids_file must be null (full split)": config["paths"].get("data.train_ids_file") is None,
        "gradient_accumulation_steps must be 1": int(raw["train"]["gradient_accumulation_steps"]) == 1,
        "data.workers must be 0": int(raw["data"]["workers"]) == 0,
        "epochs must be a positive integer": int(raw["train"]["epochs"]) >= 1,
        "grad_clip_norm must be finite and positive": math.isfinite(float(raw["train"]["grad_clip_norm"]))
        and float(raw["train"]["grad_clip_norm"]) > 0,
    }
    return [name for name, ok in checks.items() if not ok]


def advance_schedule(schedule: WarmupCosineLR, evidence: dict[str, Any],
                     optimizer: torch.optim.Optimizer) -> bool:
    """Advance the LR schedule only for a real optimizer update.

    A GradScaler skip leaves the weights untouched; letting the scheduler step
    anyway would move the learning rate ahead of the model it controls.
    """
    if evidence["skipped"]:
        return False
    schedule.step(optimizer)
    return True


class _LossTally:
    """Weighted aggregation: CE by valid pixels, Dice by positive-GT images."""

    def __init__(self) -> None:
        self.ce_sum = 0.0
        self.ce_pixels = 0
        self.dice_sum = 0.0
        self.dice_images = 0
        self.attempts = 0

    def add(self, ce: float, dice: float, valid_pixels: int, positive_images: int) -> None:
        self.ce_sum += ce * valid_pixels
        self.ce_pixels += int(valid_pixels)
        self.dice_sum += dice * positive_images
        self.dice_images += int(positive_images)
        self.attempts += 1

    def as_dict(self, *, ce_weight: float, dice_weight: float) -> dict[str, Any]:
        ce = self.ce_sum / self.ce_pixels if self.ce_pixels else None
        dice = self.dice_sum / self.dice_images if self.dice_images else None
        total = None
        if ce is not None:
            total = ce_weight * ce + dice_weight * (dice or 0.0)
        return {
            "aggregation": "ce = sum(ce_batch*valid_pixels)/sum(valid_pixels); "
                           "dice = sum(dice_batch*positive_images)/sum(positive_images)",
            "batches": self.attempts,
            "ce": ce,
            "dice": dice,
            "total": total,
            "valid_pixels": self.ce_pixels,
            "positive_images": self.dice_images,
            "mean_of_batch_means_is_not_used": True,
        }


def _save_checkpoint(
    run_dir: Path,
    name: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    schedule: WarmupCosineLR,
    epoch: int,
    successful: int,
    attempts: int,
    skipped: int,
    best: dict[str, Any],
    metrics: dict[str, Any],
    identity: dict[str, Any],
) -> Path:
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "kind": name,
        "epoch_completed": int(epoch),
        "global_step": int(successful),
        "attempts": int(attempts),
        "skipped": int(skipped),
        "model": state,
        "model_config": model.config.to_dict(),
        "optimizer": optimizer.state_dict(),
        "grad_scaler": scaler.state_dict(),
        "scheduler": schedule.state_dict(),
        "lr": float(optimizer.param_groups[0]["lr"]),
        "best": dict(best),
        "metrics": metrics,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        "identity": {key: value for key, value in identity.items() if key != "git"},
        "git_head": identity.get("git", {}).get("head"),
        "git_diff_sha256": identity.get("git", {}).get("diff_sha256"),
        "resume_supported": False,
    }
    target = run_dir / "checkpoints" / f"{name}.pt"
    _atomic_torch_save(payload, target)
    return target


def _val_summary(result: dict[str, Any]) -> dict[str, Any]:
    summary = _eval_summary(result)
    return {
        "images": summary["images"],
        "confusion": summary["confusion"],
        "global": summary["global"],
        "na_reasons": summary["global"].get("na_reasons", {}),
        "normal_images": summary["normal_images"],
        "anomaly_images": {"count": summary["anomaly_images"]["count"],
                           "macro": summary["anomaly_images"]["macro"]},
        "per_image": summary["per_image"],
    }


def _preview_npz(preview: dict[str, Any]) -> dict[str, np.ndarray]:
    payload = {}
    for sample in preview["samples"]:
        key = sample["image_id"].replace("/", "_")
        payload[key] = restore_defect_probability(sample["logits"], sample["geometry"]).numpy()
    return payload


# --------------------------------------------------------------------- train ----
def do_train(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    config = load_config(args.config)
    out_dir = Path(args.out).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite an existing run: {out_dir}")

    raw = config["raw"]
    failures = config_failures(config)
    if failures:
        raise SystemExit(f"pilot config guard failed: {failures}")

    expected_train = int(raw.get("pilot", {}).get("expected_train_images", 530))
    expected_val = int(raw.get("pilot", {}).get("expected_val_images", 132))
    epochs = int(raw["train"]["epochs"])
    batch_size = int(raw["train"]["batch_size"])
    grad_clip = float(raw["train"]["grad_clip_norm"])
    amp = bool(raw["train"]["amp"])
    base_lr = float(raw["train"]["learning_rate"])
    min_lr = float(raw["train"]["min_learning_rate"])
    warmup_ratio = float(raw["train"]["warmup_ratio"])
    weight_decay = float(raw["train"]["weight_decay"])
    threshold = float(raw["eval"]["threshold"])
    ce_weight = float(raw["loss"]["ce_weight"])
    dice_weight = float(raw["loss"]["foreground_dice_weight"])
    seed = int(raw["experiment"]["seed"])

    device = args.device
    rng_record = set_seed(seed)
    init_start = time.perf_counter()

    train_ds = data.PCB1Dataset(config, "train", flip="random", epoch=0, return_original_mask=False)
    val_ds = data.PCB1Dataset(config, "val", flip="none", return_original_mask=True)
    preview_ds = data.PCB1Dataset(config, "val", flip="none", return_original_mask=True,
                                  ids_file=Path(config["paths"]["logging.fixed_preview_manifest"]))
    if len(train_ds) != expected_train:
        raise SystemExit(f"train split must hold {expected_train} images, found {len(train_ds)}")
    if len(val_ds) != expected_val:
        raise SystemExit(f"val split must hold {expected_val} images, found {len(val_ds)}")

    train_ids = train_ds.image_ids()
    train_labels = [row["label"] for row in train_ds.rows]
    val_labels = [row["label"] for row in val_ds.rows]
    label_of_val = _labels_of(val_ds)
    loader = data.build_dataloader(train_ds, batch_size=batch_size, shuffle=True,
                                   num_workers=0, drop_last=False)
    batches_per_epoch = math.ceil(len(train_ds) / batch_size)

    loaded = load_all(Path(config["paths"]["model.pretrained_dir"]), num_labels=2, ignore_index=255)
    if not loaded.encoder_check.get("ok"):
        raise SystemExit("encoder weights failed the checkpoint comparison")
    model = loaded.model.to(device)
    params = trainable_parameter_summary(model)
    init_digest = parameter_digest(model)

    optimizer, optimizer_info = build_optimizer(model, base_lr, weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    total_steps = epochs * batches_per_epoch
    warmup_steps = max(1, int(round(warmup_ratio * total_steps)))
    schedule = WarmupCosineLR(base_lr=base_lr, min_lr=min_lr,
                              total_steps=total_steps, warmup_steps=warmup_steps)
    schedule.apply(optimizer)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    loaded.processor.save_pretrained(str(out_dir / "processor"))
    (out_dir / "model_config.json").write_text(json.dumps(model.config.to_dict(), indent=2),
                                               encoding="utf-8")
    write_resolved_config(config, out_dir / "resolved_config.yaml")
    identity_start = freeze_identity(config, label="start")
    _atomic_write_text(out_dir / "code_identity_start.json",
                       json.dumps(identity_start, indent=2, ensure_ascii=False))
    init_seconds = time.perf_counter() - init_start

    torch.cuda.reset_peak_memory_stats()
    print(f"  run     : {out_dir}")
    print(f"  data    : train {len(train_ds)} ({train_labels.count('normal')}N+{train_labels.count('anomaly')}A) "
          f"/ val {len(val_ds)} ({val_labels.count('normal')}N+{val_labels.count('anomaly')}A)")
    print(f"  epoch   : {epochs} x {batches_per_epoch} batches ({batch_size}/batch) = {total_steps} planned updates")
    print(f"  schedule: warmup {warmup_steps} + cosine to {min_lr}, first lr {schedule.lr_at(0):.3e}")

    epochs_log: list[dict[str, Any]] = []
    successful = 0
    attempts = 0
    skipped_total = 0
    best = {"iou": -1.0, "epoch": None, "iou_at_best": None}
    previous_flip_signature: str | None = None
    consecutive_skips = 0
    epoch_count = 0
    phase = "init"

    updates_path = out_dir / "updates.jsonl"
    epochs_path = out_dir / "epochs.jsonl"
    try:
        with updates_path.open("w", encoding="utf-8") as updates_handle, \
                epochs_path.open("w", encoding="utf-8") as epochs_handle:
            for epoch in range(1, epochs + 1):
                phase = f"train epoch {epoch}"
                train_ds.set_epoch(epoch)
                model.train()
                tally = _LossTally()
                seen_ids: list[str] = []
                flips: list[str] = []
                step_times: list[float] = []
                epoch_attempts = epoch_success = epoch_skips = 0
                lr_first = lr_last = float(optimizer.param_groups[0]["lr"])
                train_start = time.perf_counter()

                for batch in loader:
                    attempts += 1
                    epoch_attempts += 1
                    lr_used = float(optimizer.param_groups[0]["lr"])
                    if epoch_attempts == 1:
                        lr_first = lr_used
                    lr_last = lr_used
                    terms, evidence, step_ms = train_step(
                        model, batch, config, device, optimizer, scaler,
                        grad_clip=grad_clip, amp=amp, attempt=attempts,
                    )
                    step_times.append(step_ms)
                    skipped = evidence["skipped"]
                    tally.add(float(terms.ce.detach()), float(terms.dice.detach()),
                              terms.valid_pixels, terms.positive_images)
                    seen_ids.extend(batch["image_id"])
                    flips.extend([image_id for image_id, flipped in
                                  zip(batch["image_id"], batch["flipped"]) if flipped])
                    _jsonl_append(updates_handle, {
                        "event": "skipped" if skipped else "update",
                        "epoch": epoch,
                        "attempt": attempts,
                        "success_index": None if skipped else successful + 1,
                        "ce": round(float(terms.ce.detach()), 8),
                        "dice": round(float(terms.dice.detach()), 8),
                        "total": round(float(terms.total.detach()), 8),
                        "valid_pixels": terms.valid_pixels,
                        "ignored_pixels": terms.ignored_pixels,
                        "positive_images": terms.positive_images,
                        "lr": lr_used,
                        "scheduler_step_completed": schedule.completed,
                        "grad_norm": rounded(evidence["grad_norm"]),
                        "scale_before": evidence["scale_before"],
                        "scale_after": evidence["scale_after"],
                        "skipped": skipped,
                        "skip_reason": evidence["skip_reason"],
                        "optimizer_step_executed": evidence["optimizer_step_executed"],
                        "parameters_unchanged": evidence["parameters_unchanged"],
                        "optimizer_steps_unchanged": evidence["optimizer_steps_unchanged"],
                        "step_ms": round(step_ms, 2),
                        "batch_ids": [str(item).split("/")[-1] for item in batch["image_id"]],
                    })

                    if skipped:
                        skipped_total += 1
                        epoch_skips += 1
                        consecutive_skips += 1
                        if consecutive_skips >= 3:
                            raise RuntimeError(
                                f"three consecutive AMP skips at attempt {attempts} (epoch {epoch}); "
                                "stop and diagnose instead of making up steps")
                        continue

                    consecutive_skips = 0
                    successful += 1
                    epoch_success += 1
                    advance_schedule(schedule, evidence, optimizer)

                train_seconds = time.perf_counter() - train_start

                # -- coverage / augmentation audit ---------------------------
                expected_ids = set(train_ids)
                seen_set = set(seen_ids)
                coverage = {
                    "images_seen": len(seen_ids),
                    "unique_images_seen": len(seen_set),
                    "missing_ids": sorted(expected_ids - seen_set)[:5],
                    "unexpected_ids": sorted(seen_set - expected_ids)[:5],
                    "duplicate_visits": len(seen_ids) - len(seen_set),
                    "expected_images": expected_train,
                    "batches": epoch_attempts,
                    "batch_size_last": len(seen_ids) - batch_size * (epoch_attempts - 1),
                    "drop_last": False,
                    "dataset_epoch": train_ds.epoch,
                    "flipped_images": len(flips),
                    "flip_fraction": round(len(flips) / max(1, len(seen_ids)), 6),
                    "flip_signature_sha256": sha256_file_text("\n".join(sorted(flips))),
                    "flip_signature_changed_vs_previous_epoch":
                        previous_flip_signature is None
                        or sha256_file_text("\n".join(sorted(flips))) != previous_flip_signature,
                }
                previous_flip_signature = coverage["flip_signature_sha256"]
                if coverage["missing_ids"] or coverage["unexpected_ids"] or coverage["duplicate_visits"]:
                    raise RuntimeError(f"epoch {epoch} did not traverse the train split exactly once: {coverage}")
                if coverage["unique_images_seen"] != expected_train:
                    raise RuntimeError(f"epoch {epoch} covered {coverage['unique_images_seen']} images")

                # -- full validation ----------------------------------------
                phase = f"validate epoch {epoch}"
                val_start = time.perf_counter()
                val_result = evaluate_dataset(model, val_ds, config, device,
                                              label_of=label_of_val, keep_probabilities=False)
                val_seconds = time.perf_counter() - val_start
                val_summary = _val_summary(val_result)
                iou = val_summary["global"]["foreground_iou"]
                macro = val_summary["anomaly_images"]["macro"]["iou"]["mean"]
                is_best = iou is not None and float(iou) > float(best["iou"])
                if is_best:
                    best = {"iou": float(iou), "epoch": epoch, "iou_at_best": float(iou),
                            "preview_dir": f"val_previews/epoch_{epoch:02d}"}

                phase = f"preview epoch {epoch}"
                preview_start = time.perf_counter()
                preview_result = evaluate_dataset(model, preview_ds, config, device,
                                                  label_of=_labels_of(preview_ds),
                                                  keep_probabilities=True)
                preview_seconds = time.perf_counter() - preview_start
                preview_tag = f"pilot | epoch {epoch}"
                preview_dir = out_dir / "val_previews" / f"epoch_{epoch:02d}"
                preview_files = _render_samples(preview_ds, preview_result, preview_dir,
                                                threshold, preview_tag)

                phase = f"save epoch {epoch}"
                save_start = time.perf_counter()
                metrics_block = {"epoch": epoch, "val": val_summary, "iou": iou, "macro_iou": macro}
                last_path = _save_checkpoint(
                    out_dir, "last", model=model, optimizer=optimizer, scaler=scaler,
                    schedule=schedule, epoch=epoch, successful=successful, attempts=attempts,
                    skipped=skipped_total, best=best, metrics=metrics_block,
                    identity=identity_start)
                if is_best:
                    _save_checkpoint(
                        out_dir, "best", model=model, optimizer=optimizer, scaler=scaler,
                        schedule=schedule, epoch=epoch, successful=successful, attempts=attempts,
                        skipped=skipped_total, best=best, metrics=metrics_block,
                        identity=identity_start)
                    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                               out_dir / "checkpoints" / "model_state.pt")
                    (out_dir / "model_config.json").write_text(
                        json.dumps(model.config.to_dict(), indent=2), encoding="utf-8")
                    np.savez_compressed(out_dir / "pre_save_probabilities.npz",
                                        **_preview_npz(preview_result))
                save_seconds = time.perf_counter() - save_start

                entry = {
                    "epoch": epoch,
                    "coverage": coverage,
                    "attempts_in_epoch": epoch_attempts,
                    "successful_in_epoch": epoch_success,
                    "skipped_in_epoch": epoch_skips,
                    "attempts_total": attempts,
                    "successful_total": successful,
                    "skipped_total": skipped_total,
                    "scheduler_completed": schedule.completed,
                    "lr_first": lr_first,
                    "lr_last": lr_last,
                    "loss": tally.as_dict(ce_weight=ce_weight, dice_weight=dice_weight),
                    "train_seconds": round(train_seconds, 3),
                    "val_seconds": round(val_seconds, 3),
                    "preview_seconds": round(preview_seconds, 3),
                    "save_seconds": round(save_seconds, 3),
                    "median_step_ms": round(float(np.median(step_times)), 2) if step_times else None,
                    "val": val_summary,
                    "is_best": bool(is_best),
                    "best_so_far": dict(best),
                    "preview_files": preview_files,
                    "gpu": gpu_memory(),
                    "system_ram": _memory_status(),
                }
                epochs_log.append(entry)
                _jsonl_append(epochs_handle, entry)
                epoch_count = epoch
                best_text = "best" if is_best else "not best"
                print(f"  epoch {epoch}: {epoch_success}/{epoch_attempts} upd "
                      f"(skip {epoch_skips}), lr {lr_first:.3e}->{lr_last:.3e}, "
                      f"loss {entry['loss']['total']:.4f} | val IoU {iou} macro {macro} [{best_text}] "
                      f"| {train_seconds:.1f}s train + {val_seconds:.1f}s val")

        phase = "finalise"
        identity_end = freeze_identity(config, label="end")
        _atomic_write_text(out_dir / "code_identity_end.json",
                           json.dumps(identity_end, indent=2, ensure_ascii=False))
        source_unchanged = identity_unchanged(identity_start, identity_end)
        if not source_unchanged:
            raise RuntimeError("source/config/manifests/split metadata/preview identity changed during run")

        artifact = _manifest_of(out_dir)
        marker = {
            "schema": SCHEMA,
            "stage": "10",
            "not_a_baseline": NOT_A_BASELINE,
            "statement": "3-epoch pilot over the full 530/132 split. It checks that full-data training, "
                         "augmentation, per-epoch validation, scheduling and saving work together; "
                         "it is not a baseline score.",
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "init_seconds": round(init_seconds, 3),
            "config": {"path": str(Path(config["path"]).resolve()),
                       "sha256": sha256_file(Path(config["path"]))},
            "seed": rng_record,
            "device": device,
            "data": {
                "train_images": len(train_ds),
                "train_labels": {"normal": train_labels.count("normal"),
                                 "anomaly": train_labels.count("anomaly")},
                "val_images": len(val_ds),
                "val_labels": {"normal": val_labels.count("normal"),
                               "anomaly": val_labels.count("anomaly")},
                "train_ids_file": None,
                "batches_per_epoch": batches_per_epoch,
                "batch_size": batch_size,
                "drop_last": False,
                "workers": 0,
                "train_manifests": identity_start["manifests"],
                "split_meta_sha256": identity_start["split_meta_sha256"],
                "test_images_read": False,
            },
            "parameters": params,
            "initial_parameter_digest": init_digest,
            "optimizer": optimizer_info,
            "scheduler": schedule.describe(),
            "loss": loss_definition(ce_weight=ce_weight, dice_weight=dice_weight,
                                    eps=float(raw["loss"]["dice_epsilon"])),
            "score_definition": score_definition(),
            "training": {
                "epochs_planned": epochs,
                "epochs_completed": epoch_count,
                "planned_updates": total_steps,
                "attempts": attempts,
                "successful_updates": successful,
                "skipped": skipped_total,
                "final_lr": float(optimizer.param_groups[0]["lr"]),
                "scheduler_completed": schedule.completed,
            },
            "epochs": epochs_log,
            "best": best,
            "selection": "highest val global foreground IoU; ties keep the earlier epoch",
            "timing": {
                "train_seconds_total": round(sum(e["train_seconds"] for e in epochs_log), 3),
                "val_seconds_total": round(sum(e["val_seconds"] for e in epochs_log), 3),
                "preview_seconds_total": round(sum(e["preview_seconds"] for e in epochs_log), 3),
                "save_seconds_total": round(sum(e["save_seconds"] for e in epochs_log), 3),
            },
            "memory": {"gpu": gpu_memory(), "system_ram": _memory_status()},
            "identity": {"start": {k: v for k, v in identity_start.items() if k != "git"} | {
                "git_head": identity_start["git"]["head"],
                "git_status": identity_start["git"]["status_porcelain"],
                "git_diff_sha256": identity_start["git"]["diff_sha256"]},
                "end_git_head": identity_end["git"]["head"],
                "end_git_diff_sha256": identity_end["git"]["diff_sha256"],
                "source_and_config_unchanged": source_unchanged},
            "artifacts": artifact,
            "resume": {"implemented": False,
                       "note": "checkpoints store optimizer/scaler/scheduler/epoch/RNG, "
                               "but no restart at an epoch boundary was exercised, so resume is "
                               "not claimed"},
            "limitations": [
                "3 epochs on 530 images is a wiring/short-run check, not a baseline.",
                "workers=0 only; image decoding blocks the GPU between steps.",
                "The schedule is built for the 3-epoch pilot budget; the 50-epoch baseline "
                "must build its own schedule and must not be described as continuing this one.",
                "No test image, mask or prediction was used in this stage.",
            ],
        }
        _atomic_write_text(out_dir / "summary.json",
                           json.dumps(marker, indent=2, ensure_ascii=False))
        _atomic_write_text(out_dir / "artifact_manifest.json",
                           json.dumps(_manifest_of(out_dir), indent=2))
        print(f"  best    : epoch {best['epoch']} val IoU {best['iou']}")
        print(f"  totals  : {successful}/{attempts} updates, skipped {skipped_total}, "
              f"{marker['elapsed_seconds']} s, peak reserved "
              f"{marker['memory']['gpu'].get('peak_reserved_mib')} MiB")
        print(f"  source  : unchanged during run = {source_unchanged}")
        return 0
    except Exception as exc:  # keep the scene, then let the caller see the failure
        failure = {
            "schema": SCHEMA,
            "stage": "10",
            "phase": phase,
            "epoch": epoch_count,
            "attempts": attempts,
            "successful_updates": successful,
            "skipped": skipped_total,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "note": "run directory preserved as-is; no checkpoint from a failed epoch is claimed as valid",
        }
        _atomic_write_text(out_dir / "failure.json", json.dumps(failure, indent=2, ensure_ascii=False))
        raise


def rounded(value: float) -> float | None:
    return round(float(value), 6) if math.isfinite(float(value)) else None


# -------------------------------------------------------------------- verify ----
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def identity_unchanged(start: dict[str, Any], end: dict[str, Any]) -> bool:
    fields = ("code", "manifests", "split_meta_sha256", "preview_ids_sha256")
    return (all(start[key] == end[key] for key in fields)
            and start["config"]["sha256"] == end["config"]["sha256"]
            and all(start["git"][key] == end["git"][key] for key in ("head", "diff_sha256")))


def verify_inventory(run_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    """Check both frozen inventories before deserializing local checkpoints."""
    inventory = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    files = inventory["files"]
    earlier = summary["artifacts"]["files"]
    _require(set(files) - {"summary.json"} == set(earlier), "artifact inventory membership differs")
    for name, expected in files.items():
        path = (run_dir / name).resolve()
        _require(path.is_relative_to(run_dir.resolve()), f"artifact escapes run: {name}")
        _require(path.is_file(), f"artifact missing: {name}")
        _require(path.stat().st_size == expected["bytes"], f"artifact bytes differ: {name}")
        _require(sha256_file(path) == expected["sha256"], f"artifact SHA256 differs: {name}")
        if name in earlier:
            _require(expected == earlier[name], f"summary artifact identity differs: {name}")
    return {"ok": True, "files_checked": len(files), "summary_inventory_files": len(earlier),
            "manifest_sha256": sha256_file(run_dir / "artifact_manifest.json")}


def compare_metrics(saved: dict[str, Any], fresh: dict[str, Any]) -> dict[str, Any]:
    """All deterministic summaries, including image identities and NA reasons."""
    fields = sorted(set(saved) | set(fresh))
    checks = {key: key in saved and key in fresh and saved[key] == fresh[key] for key in fields}
    return {"ok": all(checks.values()), "fields_identical": checks}


def compare_probabilities(stored: Any, fresh: dict[str, np.ndarray], *,
                          threshold: float, atol: float = 1e-5) -> dict[str, Any]:
    _require(bool(fresh) and set(stored) == set(fresh), "preview probability keys differ or empty")
    comparisons = []
    for key, current in fresh.items():
        old = np.asarray(stored[key])
        current = np.asarray(current)
        _require(old.shape == current.shape and current.ndim == 2 and current.size > 0,
                 f"preview probability shape differs: {key}")
        _require(np.issubdtype(old.dtype, np.number) and np.issubdtype(current.dtype, np.number),
                 f"preview probability is not numeric: {key}")
        _require(bool(np.isfinite(old).all() and np.isfinite(current).all()),
                 f"preview probability must be finite: {key}")
        _require(bool(((old >= 0) & (old <= 1)).all() and ((current >= 0) & (current <= 1)).all()),
                 f"preview probability outside [0,1]: {key}")
        diff = float(np.abs(old.astype(np.float64) - current.astype(np.float64)).max())
        xor = int(((old >= threshold) != (current >= threshold)).sum())
        comparisons.append({"image_id": key, "shape": list(current.shape),
                            "max_abs_probability_diff": diff, "xor_pixels": xor,
                            "total_pixels": int(current.size), "ok": diff <= atol and xor == 0})
    return {"ok": all(item["ok"] for item in comparisons), "tolerance_atol": atol,
            "max_abs_probability_diff": max(item["max_abs_probability_diff"] for item in comparisons),
            "xor_pixels_total": sum(item["xor_pixels"] for item in comparisons),
            "total_pixels": sum(item["total_pixels"] for item in comparisons),
            "within_tolerance": all(item["max_abs_probability_diff"] <= atol for item in comparisons),
            "per_image": comparisons}


def _verify_identity(run_dir: Path, config: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    start = json.loads((run_dir / "code_identity_start.json").read_text(encoding="utf-8"))
    end = json.loads((run_dir / "code_identity_end.json").read_text(encoding="utf-8"))
    _require(identity_unchanged(start, end), "run start/end frozen identity differs")
    for key in ("code", "config", "manifests", "split_meta_sha256", "preview_ids_sha256"):
        _require(summary["identity"]["start"][key] == start[key], f"summary start identity differs: {key}")
    _require(summary["config"]["sha256"] == start["config"]["sha256"], "summary config identity differs")
    actual = {"config": sha256_file(Path(config["path"])),
              "split_meta": sha256_file(Path(config["paths"]["data.split_meta"])),
              "preview_ids": sha256_file(Path(config["paths"]["logging.fixed_preview_manifest"]))}
    _require(actual["config"] == start["config"]["sha256"], "verify config differs from run")
    _require(actual["split_meta"] == start["split_meta_sha256"], "verify split metadata differs")
    _require(actual["preview_ids"] == start["preview_ids_sha256"], "verify preview ids differ")
    for split in ("train", "val", "test"):
        actual[split] = sha256_file(Path(config["paths"][f"data.{split}_manifest"]))
        _require(actual[split] == start["manifests"][split]["sha256"], f"verify {split} manifest differs")
    current_code = _code_identity()
    return {"ok": True, "start_end_unchanged": True, "current_data_hashes": actual,
            "training_code_sha256": start["code"], "verifier_code_sha256": current_code,
            "code_changed_since_training": {key: {"trained": value, "current": current_code.get(key)}
                                            for key, value in start["code"].items()
                                            if value != current_code.get(key)},
            "test_images_read": False}


def verify_checkpoint(payload: dict[str, Any], kind: str, summary: dict[str, Any],
                      identity: dict[str, Any], *, parameter_count: int,
                      parameter_shapes: dict[int, tuple[int, ...]] | None = None) -> dict[str, Any]:
    """Validate saved state schema/counters; this is not a resume exercise."""
    epoch = summary["best"]["epoch"] if kind == "best" else summary["training"]["epochs_completed"]
    entry = next(item for item in summary["epochs"] if item["epoch"] == epoch)
    _require(payload["schema"] == CHECKPOINT_SCHEMA and payload["kind"] == kind, f"{kind} checkpoint schema/kind")
    for key, wanted in {"epoch_completed": epoch, "global_step": entry["successful_total"],
                        "attempts": entry["attempts_total"], "skipped": entry["skipped_total"]}.items():
        _require(payload[key] == wanted, f"{kind} checkpoint {key} differs")
    _require(payload["attempts"] == payload["global_step"] + payload["skipped"], f"{kind} attempt accounting")
    _require(payload["best"] == entry["best_so_far"], f"{kind} best selection differs")
    expected_metrics = {"epoch": epoch, "val": entry["val"], "iou": entry["val"]["global"]["foreground_iou"],
                        "macro_iou": entry["val"]["anomaly_images"]["macro"]["iou"]["mean"]}
    _require(payload["metrics"] == expected_metrics, f"{kind} checkpoint metrics differ")
    _require(payload["identity"] == {key: value for key, value in identity.items() if key != "git"},
             f"{kind} checkpoint run identity differs")
    _require(payload["git_head"] == identity["git"]["head"]
             and payload["git_diff_sha256"] == identity["git"]["diff_sha256"], f"{kind} Git identity differs")
    schedule = WarmupCosineLR(**payload["scheduler"])
    _require(schedule.completed == payload["global_step"] == entry["scheduler_completed"], f"{kind} scheduler step differs")
    for key in ("base_lr", "min_lr", "total_steps", "warmup_steps"):
        _require(payload["scheduler"][key] == summary["scheduler"][key], f"{kind} scheduler {key} differs")
    expected_lr = schedule.lr_at(schedule.completed)
    _require(math.isclose(payload["lr"], expected_lr, rel_tol=1e-12, abs_tol=1e-15), f"{kind} LR differs")
    for key, tensor in payload["model"].items():
        _require(isinstance(tensor, torch.Tensor) and bool(torch.isfinite(tensor).all()), f"{kind} non-finite model: {key}")
    optimizer = payload["optimizer"]
    ids = [index for group in optimizer["param_groups"] for index in group["params"]]
    _require(len(ids) == len(set(ids)) == parameter_count and set(ids) == set(optimizer["state"]),
             f"{kind} optimizer parameter state incomplete")
    recipe = summary["optimizer"]
    expected_groups = [(recipe["decay_tensors"], recipe["weight_decay"]), (recipe["no_decay_tensors"], 0.0)]
    _require(len(optimizer["param_groups"]) == len(expected_groups), f"{kind} optimizer group count differs")
    for group, (tensor_count, decay) in zip(optimizer["param_groups"], expected_groups):
        _require(math.isclose(group["lr"], expected_lr, rel_tol=1e-12, abs_tol=1e-15), f"{kind} optimizer LR differs")
        _require(len(group["params"]) == tensor_count and group["weight_decay"] == decay
                 and list(group["betas"]) == recipe["betas"] and group["eps"] == recipe["eps"],
                 f"{kind} optimizer group recipe differs")
    for index, state in optimizer["state"].items():
        _require({"step", "exp_avg", "exp_avg_sq"} <= set(state), f"{kind} optimizer state missing: {index}")
        _require(float(state["step"]) == payload["global_step"], f"{kind} optimizer step differs: {index}")
        if parameter_shapes is not None:
            _require(all(tuple(state[key].shape) == parameter_shapes[index] for key in ("exp_avg", "exp_avg_sq")),
                     f"{kind} optimizer moment shape differs: {index}")
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                _require(bool(torch.isfinite(value).all()), f"{kind} non-finite optimizer: {index}/{key}")
    scaler = payload["grad_scaler"]
    _require(math.isfinite(float(scaler["scale"])) and scaler["scale"] > 0
             and scaler["growth_factor"] > 1 and 0 < scaler["backoff_factor"] < 1
             and 0 <= scaler["_growth_tracker"] < scaler["growth_interval"], f"{kind} invalid scaler state")
    rng = payload["rng"]
    _require(set(rng) == {"python", "numpy", "torch", "cuda"}, f"{kind} RNG schema incomplete")
    random.Random().setstate(rng["python"])
    np.random.RandomState().set_state(rng["numpy"])
    for value in [rng["torch"], *rng["cuda"]]:
        _require(isinstance(value, torch.Tensor) and value.dtype == torch.uint8
                 and value.ndim == 1 and value.numel() > 0, f"{kind} invalid RNG tensor")
    _require(isinstance(rng["cuda"], list) and (rng["cuda"] or not summary["device"].startswith("cuda")),
             f"{kind} CUDA RNG missing")
    return {"ok": True, "kind": kind, "epoch_completed": epoch, "global_step": payload["global_step"],
            "optimizer_parameter_states": len(ids), "all_optimizer_steps_match": True,
            "scheduler": payload["scheduler"], "rng_schema_valid": True, "resume_supported": False}


def _checkpoint_config(payload: dict[str, Any]) -> Any:
    from transformers import SegformerConfig

    # Transformers 4.57 from_dict adds attn_implementation to its input dict.
    # Preserve the saved payload for the later best/last/JSON identity checks.
    return SegformerConfig.from_dict(deepcopy(payload["model_config"]))


def _verify_run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = Path(args.verify_run).resolve()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    config = load_config(args.config)
    _require(summary["schema"] == SCHEMA and summary["not_a_baseline"] is True, "not a pilot summary")
    _require(not config_failures(config), "pilot config guards failed")
    inventory = verify_inventory(run_dir, summary)
    frozen_identity = _verify_identity(run_dir, config, summary)
    identity_start = json.loads((run_dir / "code_identity_start.json").read_text(encoding="utf-8"))
    files = summary["artifacts"]["files"]
    required = {"checkpoints/best.pt", "checkpoints/last.pt", "checkpoints/model_state.pt",
                "model_config.json", "processor/preprocessor_config.json", "resolved_config.yaml",
                "code_identity_start.json", "code_identity_end.json", "epochs.jsonl", "updates.jsonl",
                "pre_save_probabilities.npz"}
    _require(required <= set(files), "required artifact missing from frozen inventory")
    epochs = [json.loads(line) for line in (run_dir / "epochs.jsonl").read_text(encoding="utf-8").splitlines()]
    _require(epochs == summary["epochs"], "epoch log differs from summary")
    training = summary["training"]
    _require(training["epochs_completed"] == training["epochs_planned"] == int(config["raw"]["train"]["epochs"])
             and len(epochs) == training["epochs_completed"], "incomplete pilot epochs")
    final = epochs[-1]
    for key, epoch_key in (("attempts", "attempts_total"), ("successful_updates", "successful_total"),
                           ("skipped", "skipped_total"), ("scheduler_completed", "scheduler_completed")):
        _require(training[key] == final[epoch_key], f"summary total differs: {key}")
    selected = max(epochs, key=lambda item: item["val"]["global"]["foreground_iou"])
    _require(selected["epoch"] == summary["best"]["epoch"] and final["best_so_far"] == summary["best"],
             "best checkpoint selection differs from validation history")

    device = args.device
    best_path = run_dir / "checkpoints" / "best.pt"
    # Existing local artifacts only, SHA/size checked above. NumPy RNG state in
    # these bundles requires weights_only=False; never use this on remote files.
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    last = torch.load(run_dir / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    model_config = _checkpoint_config(best)
    model = SegformerForSemanticSegmentation(model_config)
    parameter_count = len(list(model.parameters()))
    expected_optimizer, _ = build_optimizer(model, float(config["raw"]["train"]["learning_rate"]),
                                            float(config["raw"]["train"]["weight_decay"]))
    parameter_shapes = {index: tuple(parameter.shape)
                        for group, serialized in zip(expected_optimizer.param_groups,
                                                     expected_optimizer.state_dict()["param_groups"])
                        for index, parameter in zip(serialized["params"], group["params"])}
    checkpoint_checks = {kind: verify_checkpoint(payload, kind, summary, identity_start,
                                                parameter_count=parameter_count, parameter_shapes=parameter_shapes)
                         for kind, payload in (("best", best), ("last", last))}
    updates = [json.loads(line) for line in (run_dir / "updates.jsonl").read_text(encoding="utf-8").splitlines()]
    _require(len(updates) == training["attempts"], "update log attempt count differs")
    successful = skipped = 0
    schedule = WarmupCosineLR(**best["scheduler"])
    schedule.completed = 0
    for attempt, record in enumerate(updates, 1):
        _require(record["attempt"] == attempt and record["scheduler_step_completed"] == successful,
                 f"update log counters differ at attempt {attempt}")
        _require(math.isclose(record["lr"], schedule.lr_at(successful), rel_tol=1e-12, abs_tol=1e-15),
                 f"update log LR differs at attempt {attempt}")
        if record["skipped"]:
            skipped += 1
            _require(record["success_index"] is None and not record["optimizer_step_executed"]
                     and record["parameters_unchanged"] is True and record["optimizer_steps_unchanged"] is True
                     and record["scale_after"] < record["scale_before"], f"unproven skip at attempt {attempt}")
        else:
            successful += 1
            _require(record["success_index"] == successful and record["optimizer_step_executed"] is True,
                     f"unproven update at attempt {attempt}")
    _require(successful == training["successful_updates"] and skipped == training["skipped"],
             "update log totals differ")
    for kind, payload in (("best", best), ("last", last)):
        _require(payload["grad_scaler"]["scale"] == updates[payload["attempts"] - 1]["scale_after"],
                 f"{kind} scaler differs from update log")
    # JSON normalizes integer id2label keys to strings, so compare both through JSON.
    normalize = lambda value: json.loads(json.dumps(value))
    saved_config = json.loads((run_dir / "model_config.json").read_text(encoding="utf-8"))
    _require(normalize(best["model_config"]) == normalize(last["model_config"]) == saved_config,
             "model configuration differs between best/last/bundle")
    model.load_state_dict(last["model"], strict=True)
    missing, unexpected = model.load_state_dict(best["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"strict load of best.pt failed: missing={missing} unexpected={unexpected}")
    mirrored = torch.load(run_dir / "checkpoints" / "model_state.pt", map_location="cpu", weights_only=True)
    identical = set(mirrored) == set(best["model"]) and all(
        torch.equal(mirrored[key], best["model"][key]) for key in mirrored)
    if not identical:
        raise RuntimeError("checkpoints/model_state.pt is not the same model as best.pt")

    processor = SegformerImageProcessor.from_pretrained(
        str(run_dir / "processor"), local_files_only=True)
    data._processor_statistics(processor)
    model = model.to(device)

    val_ds = data.PCB1Dataset(config, "val", processor=processor, flip="none",
                              return_original_mask=True)
    preview_ds = data.PCB1Dataset(config, "val", processor=processor, flip="none",
                                  return_original_mask=True,
                                  ids_file=Path(config["paths"]["logging.fixed_preview_manifest"]))
    _require(len(val_ds) == summary["data"]["val_images"] == int(config["raw"]["pilot"]["expected_val_images"]),
             "validation split size differs")
    val_result = evaluate_dataset(model, val_ds, config, device,
                                  label_of=_labels_of(val_ds), keep_probabilities=False)
    preview_result = evaluate_dataset(model, preview_ds, config, device,
                                      label_of=_labels_of(preview_ds), keep_probabilities=True)

    threshold = float(config["raw"]["eval"]["threshold"])
    with np.load(run_dir / "pre_save_probabilities.npz", allow_pickle=False) as stored:
        probabilities = compare_probabilities(stored, _preview_npz(preview_result), threshold=threshold)

    saved_epoch = summary["best"]["epoch"]
    saved_metrics = next(e["val"] for e in summary["epochs"] if e["epoch"] == saved_epoch)
    fresh_metrics = _val_summary(val_result)
    metrics_check = compare_metrics(saved_metrics, fresh_metrics)
    check = {
        "schema": "segformer-lab/pilot-reload/2",
        "ok": metrics_check["ok"] and probabilities["ok"],
        "not_a_baseline": True,
        "run_dir": str(run_dir),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "process": {"pid": os.getpid(), "python": sys.version.split()[0],
                    "torch": torch.__version__, "device": device},
        "best_checkpoint": {
            "path": str(best_path),
            "sha256": sha256_file(best_path),
            "epoch_completed": best["epoch_completed"],
            "global_step": best["global_step"],
            "scheduler": best["scheduler"],
            "strict": True,
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "mirror_matches_best": identical,
        },
        "checks": {"artifact_inventory": inventory, "frozen_identity": frozen_identity,
                   "checkpoints": checkpoint_checks, "complete_val_metrics": metrics_check,
                   "update_log": {"ok": True, "attempts": len(updates), "successful": successful, "skipped": skipped}},
        "verification_source": {"pilot_sha256": sha256_file(Path(__file__)),
                                "tests_sha256": sha256_file(paths.PROJECT_ROOT / "tests/test_pilot.py"),
                                "training_sha256": sha256_file(paths.PROJECT_ROOT / "segformer_lab/training.py")},
        "processor": {"path": str(run_dir / "processor"),
                       "sha256_of_config": sha256_file(run_dir / "processor" / "preprocessor_config.json"),
                       "loaded_without_flag_overrides": True,
                       "used_by_val_dataset": val_ds.processor is processor,
                       "used_by_preview_dataset": preview_ds.processor is processor},
        "evaluation": val_result["evaluation"],
        "val_metrics_saved": saved_metrics,
        "val_metrics_reloaded": fresh_metrics,
        "val_confusion_identical": saved_metrics["confusion"] == fresh_metrics["confusion"],
        "val_global_identical": saved_metrics["global"] == fresh_metrics["global"],
        "preview_probability_comparison": probabilities,
        "resume_supported": False,
    }
    print(f"  best.pt : epoch {best['epoch_completed']}, step {best['global_step']}, strict load ok")
    print(f"  val     : full summary identical={metrics_check['ok']} (IoU {fresh_metrics['global']['foreground_iou']})")
    print(f"  preview : max|dprob|={probabilities['max_abs_probability_diff']:.3e}, "
          f"XOR={probabilities['xor_pixels_total']}")
    return check


def do_verify(args: argparse.Namespace) -> int:
    run_dir = Path(args.verify_run).resolve()
    report = Path(args.report).resolve() if getattr(args, "report", "") else run_dir / "reload_check.json"
    # An explicit report outside the run makes historical verification read-only.
    _require(report != run_dir and not report.is_relative_to(run_dir / "checkpoints"), "unsafe verification report path")
    if report.is_relative_to(run_dir):
        _require(report == run_dir / "reload_check.json", "report may only write reload_check.json inside run")
    try:
        check = _verify_run(args)
    except Exception as exc:
        check = {"schema": "segformer-lab/pilot-reload/2", "ok": False, "not_a_baseline": True,
                 "run_dir": str(run_dir), "error": f"{type(exc).__name__}: {exc}",
                 "resume_supported": False}
        _atomic_write_text(report, json.dumps(check, indent=2, ensure_ascii=False, allow_nan=False))
        raise
    _atomic_write_text(report, json.dumps(check, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"  wrote   : {report}")
    return 0 if check["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="stage 10: full-data 3-epoch pilot")
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--out", default="")
    parser.add_argument("--verify-run", default="", help="existing pilot run to reload and re-score")
    parser.add_argument("--report", default="", help="verification JSON output (outside run preserves original artifacts)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.verify_run:
        return do_verify(args)
    if not args.out:
        raise SystemExit("--out is required when training")
    return do_train(args)


if __name__ == "__main__":
    raise SystemExit(main())
