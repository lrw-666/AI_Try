"""Stage 08: a 20-update CUDA AMP smoke test, then a fresh-process reload check.

Two entry points in one module on purpose -- the reload must happen in a
*different* process, and keeping the comparison code next to the training code
makes it obvious that both sides use the same B1 FP32 evaluation path.

    python -m segformer_lab.smoke --config configs/smoke.yaml --out <run>
    python -m segformer_lab.smoke --verify-run <run> --out <run>

The first call writes the model bundle and ``eval/pre_save_probabilities.npz``.
The second call rebuilds the model from that bundle with ``strict=True``, scores
the same images again, and writes ``reload_check.json``.

Nothing here is a quality result. A randomly initialised decode head trained for
20 steps is a plumbing test; the summary is tagged ``not_a_baseline``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from . import data, paths  # noqa: E402
from .config import load_config, write_resolved_config  # noqa: E402
from .losses import loss_definition  # noqa: E402
from .metrics import restore_defect_probability, score_definition  # noqa: E402
from .modeling import load_all  # noqa: E402
from .training import (  # noqa: E402
    evaluate_dataset,
    load_run_model,
    parameter_digest,
    run_training,
    save_run,
    set_seed,
    sha256_file,
    trainable_parameter_summary,
)

SCHEMA = "segformer-lab/smoke/1"
NOT_A_BASELINE = True


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=str(paths.PROJECT_ROOT),
                                  capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as exc:  # pragma: no cover - diagnostics only
            return f"<error: {exc}>"

    return {"head": run("rev-parse", "HEAD"), "status_porcelain": run("status", "--porcelain")}


def _code_identity() -> dict[str, str]:
    files = ["segformer_lab/training.py", "segformer_lab/smoke.py", "segformer_lab/losses.py",
             "segformer_lab/metrics.py", "segformer_lab/evaluation.py", "segformer_lab/data.py",
             "segformer_lab/modeling.py", "configs/smoke.yaml"]
    out = {}
    for name in files:
        path = paths.PROJECT_ROOT / name
        out[name] = sha256_file(path) if path.is_file() else ""
    return out


def provenance(config: dict[str, Any], extra_files: dict[str, str] | None = None) -> dict[str, Any]:
    model_dir = Path(config["paths"]["model.pretrained_dir"])
    source = json.loads((model_dir / "source.json").read_text(encoding="utf-8"))
    manifests = {}
    for split in ("train", "val", "test"):
        path = Path(config["paths"][f"data.{split}_manifest"])
        # hashing the test manifest is the only test-side access allowed here
        manifests[split] = {"path": str(path), "sha256": sha256_file(path)}
    files = {
        "config": {"path": str(Path(config["path"]).resolve()), "sha256": sha256_file(Path(config["path"]))},
        "ids": {"path": str(config["paths"]["data.train_ids_file"]),
                "sha256": sha256_file(Path(config["paths"]["data.train_ids_file"]))},
        "split_meta": {"path": str(config["paths"]["data.split_meta"]),
                       "sha256": sha256_file(Path(config["paths"]["data.split_meta"]))},
        "preview_ids": {"path": str(config["paths"]["logging.fixed_preview_manifest"]),
                        "sha256": sha256_file(Path(config["paths"]["logging.fixed_preview_manifest"]))},
    }
    if extra_files:
        files.update(extra_files)
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "manifest_identity": manifests,
        "files": files,
        "snapshot": {
            "model_id": source.get("model_id"),
            "revision": source.get("revision"),
            "weights_sha256": next(f["sha256"] for f in source["files"] if f["path"] == "pytorch_model.bin"),
        },
        "git": _git_state(),
        "code": _code_identity(),
    }


def _preview_dataset(config: dict[str, Any], ids_path: Path, split: str = "val") -> data.PCB1Dataset:
    return data.PCB1Dataset(config, split, flip="none", ids_file=ids_path, return_original_mask=True)


def _labels_of(dataset: data.PCB1Dataset) -> dict[str, str]:
    return {row["image"]: row["label"] for row in dataset.rows}


def _render_samples(dataset: data.PCB1Dataset, result: dict[str, Any], out_dir: Path,
                    threshold: float, tag: str) -> list[str]:
    """One three-column strip per image: source+GT, probability, thresholded overlay."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for sample in result["samples"]:
        image_id = sample["image_id"]
        geometry = sample["geometry"]
        probability = restore_defect_probability(sample["logits"], geometry).numpy()
        with Image.open(dataset.data_root / image_id) as handle:
            handle.load()
            rgb = np.asarray(handle.convert("RGB"))
        gt = np.asarray(sample["original_mask"])

        figure, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), dpi=100)
        axes[0].imshow(rgb)
        axes[0].set_title("source", fontsize=8)
        axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("original GT", fontsize=8)
        axes[2].imshow(rgb)
        overlay = np.ma.masked_where(probability < threshold, probability)
        axes[2].imshow(overlay, cmap="autumn", alpha=0.45, vmin=0.0, vmax=1.0)
        axes[2].set_title(f"pred >= {threshold}", fontsize=8)
        for axis in axes:
            axis.set_xticks([])
            axis.set_yticks([])
        figure.suptitle(f"{tag} | {image_id.split('/')[-1]} | {sample['label']}", fontsize=9)
        figure.tight_layout()
        name = f"{sample['label']}_{Path(image_id).stem}.png"
        figure.savefig(out_dir / name)
        plt.close(figure)
        written.append(name)
    return written


def _eval_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "images": result["images"],
        "confusion": {k: result["confusion"][k] for k in ("tp", "fp", "fn", "tn", "pixels")},
        "global": result["global"],
        "normal_images": result["normal_images"],
        "anomaly_images": {k: result["anomaly_images"][k] for k in ("count", "macro")},
        "evaluation": result["evaluation"],
        "per_image": result["per_image"],
    }


# --------------------------------------------------------------------- train ----
def do_train(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    config = load_config(args.config)
    out_dir = Path(args.out).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite an existing run: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_policy = config["raw"]["eval"].get("precision")
    if eval_policy != "ieee_fp32":
        raise SystemExit("eval.precision must be 'ieee_fp32'")
    amp = bool(config["raw"]["train"]["amp"])
    if not amp:
        raise SystemExit("stage 08 requires CUDA AMP; refusing to run without it")

    target = int(config["raw"]["smoke"]["target_updates"])
    attempt_cap = int(config["raw"]["smoke"]["max_attempts"])
    if target != 20:
        raise SystemExit(f"stage 08 fixes 20 successful updates, config says {target}")

    seed = int(config["raw"]["experiment"]["seed"])
    rng_record = set_seed(seed)
    device = args.device

    ids_path = Path(config["paths"]["data.train_ids_file"])
    train_ds = data.PCB1Dataset(config, "train", flip="none", return_original_mask=False)
    if len(train_ds) != 8:
        raise SystemExit(f"training subset must be 8 images, got {len(train_ds)}")
    train_labels = [row["label"] for row in train_ds.rows]
    if train_labels.count("normal") != 4 or train_labels.count("anomaly") != 4:
        raise SystemExit(f"subset composition is {train_labels}")
    loader = data.build_dataloader(train_ds, batch_size=int(config["raw"]["train"]["batch_size"]),
                                   shuffle=True, num_workers=0, drop_last=False)

    print(f"  run     : {out_dir}")
    print(f"  subset  : {len(train_ds)} images ({train_labels.count('normal')}N+{train_labels.count('anomaly')}A), flip=none")
    loaded = load_all(Path(config["paths"]["model.pretrained_dir"]), num_labels=2, ignore_index=255)
    if not loaded.encoder_check.get("ok"):
        raise SystemExit("encoder weights failed the checkpoint comparison")
    model = loaded.model.to(device)
    params = trainable_parameter_summary(model)

    updates_path = out_dir / "updates.jsonl"
    with updates_path.open("w", encoding="utf-8") as log_handle:
        outcome, optimizer, scaler, optimizer_info = run_training(
            model, loader, config, device,
            target_updates=target, max_attempts=attempt_cap,
            lr=float(config["raw"]["train"]["learning_rate"]),
            weight_decay=float(config["raw"]["train"]["weight_decay"]),
            grad_clip=float(config["raw"]["train"]["grad_clip_norm"]),
            amp=amp, accumulation=int(config["raw"]["train"]["gradient_accumulation_steps"]),
            log_handle=log_handle,
        )

    # -- B1 FP32 evaluation of the fixed 6-image val preview ------------------
    preview_ids = Path(config["paths"]["logging.fixed_preview_manifest"])
    val_ds = _preview_dataset(config, preview_ids)
    label_of = _labels_of(val_ds)
    eval_result = evaluate_dataset(model, val_ds, config, device,
                                   label_of=label_of, keep_probabilities=True)
    figures = _render_samples(val_ds, eval_result, out_dir / "eval_smoke",
                              float(config["raw"]["eval"]["threshold"]), "smoke")

    # probabilities at full resolution: 6 x 1070 x 1404 float32 ~ 36 MB,
    # kept in the resource tree so the reload process compares real numbers
    npz_path = out_dir / "pre_save_probabilities.npz"
    payload = {}
    for sample in eval_result["samples"]:
        key = sample["image_id"].replace("/", "_")
        payload[key] = restore_defect_probability(sample["logits"], sample["geometry"]).numpy()
    np.savez_compressed(npz_path, **payload)

    digest_before = parameter_digest(model)
    manifest = save_run(
        out_dir, model, loaded.processor, config,
        optimizer=optimizer, scaler=scaler,
        successful_updates=outcome.successful, attempts=outcome.attempts, skipped=outcome.skipped,
        extra={"stage": "08", "not_a_baseline": NOT_A_BASELINE},
    )
    write_resolved_config(config, out_dir / "resolved_config.yaml")

    summary = {
        "schema": SCHEMA,
        "stage": "08",
        "not_a_baseline": NOT_A_BASELINE,
        "statement": "20-update AMP plumbing test with a randomly initialised decode head; "
                     "the scores below only prove the path works",
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "config": {"path": str(Path(config["path"]).resolve()), "sha256": sha256_file(Path(config["path"]))},
        "seed": rng_record,
        "device": device,
        "train_subset": {
            "ids_file": str(ids_path),
            "ids_sha256": sha256_file(ids_path),
            "images": [row["image"] for row in train_ds.rows],
            "labels": train_labels,
        },
        "parameters": params,
        "optimizer": optimizer_info,
        "loss": loss_definition(ce_weight=float(config["raw"]["loss"]["ce_weight"]),
                                dice_weight=float(config["raw"]["loss"]["foreground_dice_weight"]),
                                eps=float(config["raw"]["loss"]["dice_epsilon"])),
        "score_definition": score_definition(),
        "training": {
            "successful_updates": outcome.successful,
            "attempts": outcome.attempts,
            "skipped": outcome.skipped,
            "stopped_reason": outcome.stopped_reason,
            "parameter_evidence": outcome.parameter_evidence,
            "timing": outcome.timing,
            "memory": outcome.memory,
        },
        "evaluation_before_save": _eval_summary(eval_result),
        "figures": figures,
        "parameter_digest_before_save": digest_before,
        "artifacts": manifest,
        "encoder_check": {"matched_exact": loaded.encoder_check.get("matched_exact"),
                          "checkpoint_tensors": loaded.encoder_check.get("checkpoint_tensor_count"),
                          "max_abs_diff": loaded.encoder_check.get("max_abs_diff")},
        "provenance": provenance(config),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  updates : {outcome.successful} successful / {outcome.attempts} attempts, "
          f"skipped={outcome.skipped}, {outcome.stopped_reason}")
    print(f"  evidence: encoder_delta={outcome.parameter_evidence.get('max_abs_parameter_delta', {})}")
    print(f"  timing  : median {outcome.timing['median_step_ms']} ms/update, "
          f"peak reserved {outcome.memory['gpu'].get('peak_reserved_mib')} MiB")
    print(f"  val     : IoU {eval_result['global']['foreground_iou']} (smoke only, not a baseline)")
    print(f"  saved   : {out_dir}")
    return 0


# -------------------------------------------------------------------- verify ----
def do_verify(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    run_dir = Path(args.verify_run).resolve()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    config = load_config(args.config)

    device = args.device
    model, processor, info = load_run_model(run_dir)
    model = model.to(device)

    preview_ids = Path(config["paths"]["logging.fixed_preview_manifest"])
    val_ds = _preview_dataset(config, preview_ids)
    label_of = _labels_of(val_ds)
    result = evaluate_dataset(model, val_ds, config, device, label_of=label_of, keep_probabilities=True)

    stored = np.load(run_dir / "pre_save_probabilities.npz")
    comparisons = []
    for sample in result["samples"]:
        key = sample["image_id"].replace("/", "_")
        fresh = restore_defect_probability(sample["logits"], sample["geometry"]).numpy()
        if key not in stored:
            raise SystemExit(f"stored probability missing for {key}")
        old = stored[key]
        threshold = float(config["raw"]["eval"]["threshold"])
        xor = int(((old >= threshold) != (fresh >= threshold)).sum())
        comparisons.append({
            "image_id": sample["image_id"],
            "max_abs_probability_diff": float(np.abs(old - fresh).max()),
            "xor_pixels": xor,
            "total_pixels": int(old.size),
        })

    before = summary["evaluation_before_save"]
    after = _eval_summary(result)
    confusion_equal = before["confusion"] == after["confusion"]
    metrics_equal = before["global"] == after["global"]
    max_prob = max(item["max_abs_probability_diff"] for item in comparisons)
    total_xor = sum(item["xor_pixels"] for item in comparisons)

    reload_check = {
        "schema": "segformer-lab/smoke-reload/1",
        "run_dir": str(run_dir),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "process": {"pid": os.getpid(), "python": sys.version.split()[0],
                    "torch": torch.__version__, "device": device},
        "load": info,
        "strict_loaded": info["strict"] and not info["missing_keys"] and not info["unexpected_keys"],
        "precision": after["evaluation"]["precision_record"],
        "comparison": {
            "max_abs_probability_diff": max_prob,
            "xor_pixels_total": total_xor,
            "total_pixels": int(sum(item["total_pixels"] for item in comparisons)),
            "tolerance_atol": 1e-5,
            "within_tolerance": max_prob <= 1e-5,
            "confusion_identical": confusion_equal,
            "global_metrics_identical": metrics_equal,
            "per_image": comparisons,
        },
        "metrics_after_reload": after,
        "metrics_before_save": before,
        "parameter_digest_before_save": summary.get("parameter_digest_before_save"),
        "parameter_digest_after_reload": parameter_digest(model),
        "checkpoint_sha256": info["model_state_sha256"],
        "steps_recorded": summary["training"]["successful_updates"],
    }
    (run_dir / "reload_check.json").write_text(json.dumps(reload_check, indent=2, ensure_ascii=False),
                                               encoding="utf-8")

    print(f"  strict load : missing={len(info['missing_keys'])} unexpected={len(info['unexpected_keys'])}")
    print(f"  probability : max|diff|={max_prob:.3e} (atol 1e-5), XOR={total_xor}")
    print(f"  metrics     : identical confusion={confusion_equal}, identical global={metrics_equal}")
    print(f"  wrote       : {run_dir / 'reload_check.json'}")
    return 0 if (reload_check["strict_loaded"] and reload_check["comparison"]["within_tolerance"]
                 and confusion_equal) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="stage 08 smoke test and reload verification")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--out", default="")
    parser.add_argument("--verify-run", default="", help="existing run directory to reload and re-score")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    if args.verify_run:
        return do_verify(args)
    if not args.out:
        raise SystemExit("--out is required when training")
    return do_train(args)


if __name__ == "__main__":
    raise SystemExit(main())
