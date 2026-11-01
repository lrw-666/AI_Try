"""Read-only checks of a frozen epoch-1 boundary and actual final last models.

Numerical tolerance is declared here, before comparisons, and cannot be widened
through CLI flags. A numerical diagnostic failure exits zero; structural/state
failures exit nonzero. Neither mode trains or saves model weights.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from . import data, train_full as tf
from .config import load_config
from .metrics import restore_defect_probability
from .training import evaluate_dataset, sha256_file

PROBABILITY_ATOL = 1e-5


def state_equal(left: Any, right: Any) -> bool:
    """Exact values, dtype, shape and keys, including optimizer groups and RNG."""
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (torch.is_tensor(left) and torch.is_tensor(right) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
                and left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(state_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and type(left) is type(right):
        return len(left) == len(right) and all(state_equal(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and bool(left == right)


def validate_checkpoint(payload: dict, *, boundary: bool, path: Path | None = None,
                        final_epoch: int = 3) -> None:
    expected = 1 if boundary else final_epoch
    if (payload.get("kind") != "last" or type(payload.get("epoch_completed")) is not int
            or payload.get("epoch_completed") != expected
            or payload.get("next_epoch") != expected + 1):
        raise ValueError(f"requires kind=last, epoch_completed={expected}, next_epoch={expected + 1}")
    if not boundary and (path is None or path.name != "last.pt" or path.parent.name != "checkpoints"):
        raise ValueError("continuation comparison requires the actual checkpoints/last.pt")


def compare_probabilities(left: torch.Tensor, right: torch.Tensor) -> dict:
    if left.shape != right.shape or not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise ValueError("probabilities must have equal shapes and be finite")
    gap = float((left - right).abs().max())
    xor = int(((left >= 0.5) != (right >= 0.5)).sum())
    return {"max_abs_diff": gap, "xor_pixels": xor, "pixels": left.numel(),
            "within_tolerance": gap <= PROBABILITY_ATOL and xor == 0}


def protect_output(out: Path, runs: list[Path]) -> None:
    target = out.resolve()
    if target.exists() or any(target.is_relative_to(run.resolve()) for run in runs):
        raise ValueError("output must be a new file outside all input run directories")


def _processor(run: Path) -> Any:
    from transformers import SegformerImageProcessor

    if not (run / "processor/preprocessor_config.json").is_file():
        raise ValueError("saved run processor is required")
    return SegformerImageProcessor.from_pretrained(
        str(run / "processor"), local_files_only=True, do_resize=False,
        do_reduce_labels=False, do_rescale=True, do_normalize=True)


def _updates(run: Path, *, epoch: int | None = None, after: int = 0) -> list[dict]:
    with (run / "updates.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return [row for row in rows if row["epoch"] > after and (epoch is None or row["epoch"] == epoch)]


def _tensor_diff(left: Any, right: Any) -> dict:
    def tensors(value: Any, prefix: str = "") -> dict:
        if torch.is_tensor(value):
            return {prefix: value.detach().cpu()}
        items = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, (list, tuple)) else []
        return {key: tensor for name, part in items
                for key, tensor in tensors(part, f"{prefix}/{name}").items()}
    aa, bb = tensors(left), tensors(right)
    same_shapes = aa.keys() == bb.keys() and all(aa[k].shape == bb[k].shape for k in aa)
    diffs = {k: float((aa[k].double() - bb[k].double()).abs().max())
             for k in aa if k in bb and aa[k].shape == bb[k].shape}
    worst = max(diffs, key=diffs.get) if diffs else None
    return {"exact": state_equal(left, right), "tensor_keys_shapes_equal": same_shapes,
            "tensors": len(aa), "max_abs_diff": diffs[worst] if worst else None, "worst_tensor": worst}


def boundary_check(checkpoint: Path, run: Path, control: Path, config: dict, device: str) -> dict:
    digest = sha256_file(checkpoint)
    payload = tf.read_checkpoint(checkpoint)
    validate_checkpoint(payload, boundary=True)
    processor = _processor(run)
    dataset = data.PCB1Dataset(config, "train", processor=processor, flip="random",
                              ids_file=config["paths"].get("data.train_ids_file"), return_original_mask=False)
    raw = config["raw"]
    loader = data.build_dataloader(dataset, batch_size=int(raw["train"]["batch_size"]),
                                   shuffle=bool(raw["data"]["shuffle"]),
                                   num_workers=int(raw["data"]["workers"]), drop_last=bool(raw["data"]["drop_last"]))
    identities = tf._freeze_full_identity(config, run, label="resume-boundary-check")
    restored = tf.restore_checkpoint(payload, config, device, identities=identities,
                                     batches_per_epoch=len(loader))
    loader.generator.set_state(restored["loader_state"]["generator_state"])
    rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
           "cuda": torch.cuda.get_rng_state_all() if device.startswith("cuda") else []}
    # Capture every check BEFORE creating an iterator, which consumes loader RNG.
    flags = {"model": state_equal(payload["model"], restored["model"].state_dict()),
             "optimizer_including_param_groups": state_equal(payload["optimizer"], restored["optimizer"].state_dict()),
             "scaler": state_equal(payload["grad_scaler"], restored["scaler"].state_dict()),
             "scheduler": state_equal(payload["scheduler"], restored["schedule"].state_dict()),
             "all_group_lr": all(group["lr"] == payload["lr"] for group in restored["optimizer"].param_groups),
             "step": payload["step"] == {"successful": restored["successful"], "attempts": restored["attempts"], "skipped": restored["skipped"]},
             "best": state_equal(payload["best"], restored["best"]),
             "next_epoch": restored["epoch_start"] == payload["next_epoch"],
             "model_train_mode": restored["model"].training,
             "loader_generator": state_equal(payload["loader"]["generator_state"], loader.generator.get_state()),
             "loader_settings": all(payload["loader"][key] == value for key, value in {
                 "batch_size": loader.batch_size, "num_workers": loader.num_workers,
                 "drop_last": loader.drop_last, "shuffle": isinstance(loader.sampler, torch.utils.data.RandomSampler)}.items()),
             "saved_processor_is_actual_dataset_processor": dataset.processor is processor}
    flags.update({f"rng_{key}": state_equal(payload["rng"][key], value) for key, value in rng.items()})
    dataset.set_epoch(2)
    actual = [{"batch_ids": [Path(name).name for name in batch["image_id"]],
               "flipped_ids": [Path(name).name for name, flipped in zip(batch["image_id"], batch["flipped"]) if flipped]}
              for batch in loader]
    reference = [{key: row[key] for key in ("batch_ids", "flipped_ids")} for row in _updates(control, epoch=2)]
    order_ok = actual == reference and len(actual) == len(loader) == 6
    immutable = digest == sha256_file(checkpoint)
    return {"mode": "frozen_epoch1_boundary", "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": digest, "checkpoint_epoch": payload["epoch_completed"],
            "step": payload["step"], "scheduler": payload["scheduler"], "checks_before_loader_iteration": flags,
            "state_at_load_ok": all(flags.values()), "checkpoint_unchanged": immutable,
            "processor": {"path": str((run / "processor").resolve()), "class": type(processor).__name__,
                          "sha256": sha256_file(run / "processor/preprocessor_config.json"), "instance_used_for_decoding": dataset.processor is processor},
            "next_epoch": dataset.epoch, "actual_batches": actual, "control_batches": reference,
            "next_epoch_order_and_flip_ok": order_ok,
            "structural_ok": all(flags.values()) and order_ok and immutable,
            "scope": "Exact frozen epoch-1 load state before iteration, then six CPU loader batches; no model forward or update."}


def continuation_check(left: Path, right: Path, config: dict, device: str) -> dict:
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    paths = [left, right]
    digests = [sha256_file(path) for path in paths]
    payloads = [tf.read_checkpoint(path) for path in paths]
    results = []
    processors = []
    for path, payload in zip(paths, payloads):
        validate_checkpoint(payload, boundary=False, path=path, final_epoch=int(config["raw"]["train"]["epochs"]))
        run = path.parent.parent
        tf._validate_resume_identity(payload["identity"], tf._freeze_full_identity(config, run, label="last-comparison"))
        tf._validated_resume_logs(run, payload)
        processor = _processor(run)
        dataset = data.PCB1Dataset(config, "val", processor=processor, flip="none",
                                  ids_file=Path(config["paths"]["logging.fixed_preview_manifest"]), return_original_mask=True)
        if len(dataset) != 6:
            raise ValueError("continuation check requires exactly six fixed validation images")
        processors.append({"path": str((run / "processor").resolve()),
                           "sha256": sha256_file(run / "processor/preprocessor_config.json"),
                           "instance_used_for_decoding": dataset.processor is processor})
        model = SegformerForSemanticSegmentation(SegformerConfig.from_dict(deepcopy(payload["model_config"])))
        model.load_state_dict(payload["model"], strict=True)
        result = evaluate_dataset(model.to(device), dataset, config, device, keep_probabilities=True)
        results.append(result)
        del model
    a, b = payloads
    rows = [_updates(path.parent.parent, after=1) for path in paths]
    trajectory = {key: len(rows[0]) == len(rows[1]) > 0 and all(x[key] == y[key] for x, y in zip(*rows))
                  for key in ("batch_ids", "flipped_ids", "lr", "attempt", "success_index", "scheduler_completed", "skipped")}
    prefix = [_updates(path.parent.parent, epoch=1) for path in paths]
    probs = []
    for aa, bb in zip(results[0]["samples"], results[1]["samples"]):
        if aa["image_id"] != bb["image_id"]:
            raise ValueError("validation image order differs")
        probs.append({"image_id": aa["image_id"], **compare_probabilities(
            restore_defect_probability(aa["logits"], aa["geometry"]),
            restore_defect_probability(bb["logits"], bb["geometry"]))})
    counts_ok = all(state_equal(a[key], b[key]) for key in ("step", "scheduler", "lr"))
    expected_rows = (int(config["raw"]["train"]["epochs"]) - 1) * len(_updates(paths[0].parent.parent, epoch=1))
    complete_rows = all(len(row) == expected_rows for row in rows)
    immutable = all(sha256_file(path) == digest for path, digest in zip(paths, digests))
    for result in results:
        result.pop("samples")
    metrics_equal = state_equal(results[0]["per_image"], results[1]["per_image"])
    return {"mode": "actual_final_last_continuation", "checkpoints": [
                {"path": str(path.resolve()), "sha256": digest, "epoch": value["epoch_completed"],
                 "step": value["step"], "lr": value["lr"], "processor": processor}
                for path, value, digest, processor in zip(paths, payloads, digests, processors)],
            "declared_tolerance": {"probability_atol": PROBABILITY_ATOL, "threshold": 0.5, "required_xor": 0,
                                   "per_image_metrics": "exact with identical GT and threshold masks",
                                   "model_optimizer": "differences reported; no acceptance bound claimed"},
            "model": _tensor_diff(a["model"], b["model"]), "optimizer": _tensor_diff(a["optimizer"], b["optimizer"]),
            "counts_scheduler_lr_equal": counts_ok, "post_boundary_update_rows": [len(row) for row in rows],
            "post_boundary_expected_rows": expected_rows, "post_boundary_rows_complete": complete_rows,
            "checkpoints_unchanged": immutable,
            "trajectory_equal": trajectory, "epoch1_logged_losses_equal": len(prefix[0]) == len(prefix[1]) > 0
                and all(x["total"] == y["total"] for x, y in zip(*prefix)),
            "epoch1_max_logged_loss_abs_diff": max((abs(x["total"] - y["total"]) for x, y in zip(*prefix)), default=None),
            "probabilities": probs, "probability_max_abs_diff": max(row["max_abs_diff"] for row in probs),
            "xor_pixels": sum(row["xor_pixels"] for row in probs), "val_left": results[0], "val_right": results[1],
            "per_image_metrics_identical": metrics_equal,
            "val_confusion_identical": state_equal(results[0]["confusion"], results[1]["confusion"]),
            "numeric_continuation_within_tolerance": len(probs) == 6 and metrics_equal
                and all(row["within_tolerance"] for row in probs),
            "structural_ok": counts_ok and complete_rows and immutable and all(trajectory.values()),
            "scope": "Actual final last models, six B1 IEEE FP32 validation forwards per model with each saved run processor; no training. Independent prefixes can already differ. Observed divergence is not attributed to a particular kernel or exclusively to resume."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "run", "control", "left", "right"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args(argv)
    boundary = args.checkpoint is not None
    if boundary and (not args.run or not args.control or args.left or args.right):
        parser.error("boundary mode requires --checkpoint --run --control only")
    if not boundary and (not args.left or not args.right or args.run or args.control):
        parser.error("continuation mode requires --left --right only")
    runs = [args.run, args.control] if boundary else [args.left.parent.parent, args.right.parent.parent]
    protect_output(args.out, runs)
    config = load_config(args.config)
    if config["raw"].get("stage", {}).get("kind") != "fixture":
        parser.error("this helper is restricted to explicit fixture configs")
    report = (boundary_check(args.checkpoint, args.run, args.control, config, args.device) if boundary
              else continuation_check(args.left, args.right, config, args.device))
    report.update({"schema": "segformer-lab/resume-check/1", "helper_sha256": sha256_file(Path(__file__)),
                   "no_training": True, "exit_policy": "structural failure nonzero; numerical tolerance failure remains diagnostic"})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"out": str(args.out), "structural_ok": report["structural_ok"],
                      "numeric_continuation_within_tolerance": report.get("numeric_continuation_within_tolerance")}))
    return 0 if report["structural_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
