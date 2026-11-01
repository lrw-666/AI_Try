"""Read-only acceptance of an existing formal stage-11 run (never trains).

Only hash-verified local checkpoint bundles are deserialized. The saved processor
and best weights run full validation and six fixed previews in observed CUDA B1
IEEE FP32. This command neither resumes training nor evaluates the test split.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from . import data, paths, pilot
from .config import load_config
from .training import WarmupCosineLR, build_optimizer, evaluate_dataset, sha256_file

SCHEMA = "segformer-lab/full-reload/1"
CHECKPOINT_SCHEMA = "segformer-lab/full-checkpoint/1"
require = pilot._require


def guard_report_path(run: Path, out: Path, *, protected: list[Path]) -> None:
    """Permit only an independent JSON report, never an input or source tree."""
    run, out = run.resolve(), out.resolve()
    require(out.suffix.lower() == ".json", "report must have .json suffix")
    for source in [run, *protected]:
        source = source.resolve()
        require(out != source and not out.is_relative_to(source), f"report would overwrite protected input: {source}")
    if out.exists():
        require(out.is_file() and json.loads(out.read_text(encoding="utf-8")).get("schema") == SCHEMA,
                "existing output is not a full-reload report")


def verify_checkpoint(payload: dict[str, Any], kind: str, summary: dict[str, Any],
                      identity: dict[str, Any], *,
                      parameter_shapes: dict[int, tuple[int, ...]]) -> dict[str, Any]:
    """Full-schema boundary checks, then reuse the established optimizer/RNG gates."""
    require(payload["schema"] == CHECKPOINT_SCHEMA and payload["stage"] == "11"
            and payload["kind"] == kind, f"{kind} full checkpoint schema/kind")
    epoch = summary["best"]["epoch"] if kind == "best" else summary["training"]["epochs_completed"]
    entry = next(row for row in summary["epochs"] if row["epoch"] == epoch)
    require(payload["epoch_completed"] == epoch and payload["next_epoch"] == epoch + 1,
            f"{kind} next epoch boundary differs")
    counts = payload["step"]
    require(set(counts) == {"successful", "attempts", "skipped"}
            and all(type(v) is int and v >= 0 for v in counts.values()), f"{kind} invalid step counters")
    require(counts["attempts"] == epoch * summary["data"]["batches_per_epoch"],
            f"{kind} attempts do not cover complete epochs")
    require(payload["metrics"] == {"epoch": epoch, "val": entry["val"], "diagnostics": entry["diagnostics"]},
            f"{kind} full checkpoint metrics differ")
    require(payload["resume_supported"] is True and payload["resume_scope"] == "complete_epoch_boundary",
            f"{kind} saved resume contract differs")
    require(payload["scheduler_plan"] == {k: summary["scheduler"][k] for k in ("total_steps", "warmup_steps")},
            f"{kind} scheduler plan differs")
    loader = payload["loader"]
    expected_loader = {"batch_size": summary["data"]["batch_size"], "num_workers": summary["data"]["workers"],
                       "shuffle": summary["data"]["shuffle"], "drop_last": summary["data"]["drop_last"],
                       "dataset_epoch": epoch}
    require(all(loader.get(k) == v for k, v in expected_loader.items()), f"{kind} loader boundary differs")
    generator = loader["generator_state"]
    require(isinstance(generator, torch.Tensor) and generator.dtype == torch.uint8
            and generator.ndim == 1 and generator.numel() > 0, f"{kind} invalid loader generator")
    torch.Generator().set_state(generator)
    for name, tensor in payload["model"].items():
        require(isinstance(tensor, torch.Tensor) and (not tensor.is_floating_point() or tensor.dtype == torch.float32),
                f"{kind} model must be FP32: {name}")
    for index, state in payload["optimizer"]["state"].items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                require(value.dtype == torch.float32, f"{kind} optimizer must be FP32: {index}/{key}")
    expected_groups = [{"weight_decay": g["weight_decay"], "lr": g["lr"], "tensors": len(g["params"])}
                       for g in payload["optimizer"]["param_groups"]]
    require(payload["optimizer_groups"] == expected_groups, f"{kind} optimizer_groups differ")
    # Translate only the schema differences. Never mutate the original payload:
    # its full metrics and identity were checked above and remain report evidence.
    adapted = dict(payload, schema=pilot.CHECKPOINT_SCHEMA,
                   global_step=counts["successful"], attempts=counts["attempts"], skipped=counts["skipped"],
                   git_head=payload["git"]["head"], git_diff_sha256=payload["git"]["diff_sha256"],
                   metrics={"epoch": epoch, "val": entry["val"],
                            "iou": entry["val"]["global"]["foreground_iou"],
                            "macro_iou": entry["val"]["anomaly_images"]["macro"]["iou"]["mean"]})
    result = pilot.verify_checkpoint(adapted, kind, summary, identity,
                                     parameter_count=len(parameter_shapes), parameter_shapes=parameter_shapes)
    result.update(step=dict(counts), next_epoch=payload["next_epoch"], loader={**expected_loader,
                  "generator_state_bytes": generator.numel(), "generator_state_valid": True},
                  saved_resume_supported=True, resume_exercised=False, model_and_optimizer_fp32=True)
    result.pop("resume_supported", None)
    return result


def verify_identity(run: Path, config: dict[str, Any], summary: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    start = json.loads((run / "code_identity_start.json").read_text(encoding="utf-8"))
    end = json.loads((run / "code_identity_end.json").read_text(encoding="utf-8"))
    stable = lambda item: {k: v for k, v in item.items() if k not in ("at", "label", "git")}
    require(stable(start) == stable(end), "historical start/end identity differs")
    require(all(start["git"][k] == end["git"][k] for k in ("head", "diff_sha256", "status_porcelain")),
            "historical start/end Git identity differs")
    embedded = {k: v for k, v in start.items() if k != "git"} | {
        "git_head": start["git"]["head"], "git_status": start["git"]["status_porcelain"],
        "git_diff_sha256": start["git"]["diff_sha256"]}
    require(summary["identity"]["start"] == embedded, "embedded start identity differs")
    require(summary["identity"]["source_and_config_unchanged"] is True
            and summary["identity"]["end_git_head"] == end["git"]["head"]
            and summary["identity"]["end_git_diff_sha256"] == end["git"]["diff_sha256"],
            "embedded end identity differs")
    actual = {"config": sha256_file(Path(config["path"])),
              "split_meta": sha256_file(config["paths"]["data.split_meta"]),
              "preview_ids": sha256_file(config["paths"]["logging.fixed_preview_manifest"])}
    require(summary["config"] == start["config"] and actual["config"] == start["config"]["sha256"],
            "actual config differs from training")
    require(actual["split_meta"] == start["split_meta_sha256"]
            and actual["preview_ids"] == start["preview_ids_sha256"], "actual metadata/preview identity differs")
    for split in ("train", "val", "test"):
        actual[split] = sha256_file(config["paths"][f"data.{split}_manifest"])
        require(actual[split] == start["manifests"][split]["sha256"], f"actual {split} CSV differs")
    require(start["train_ids_file"] is None and start.get("eval_ids_file") is None, "formal run used a subset")
    resolved = yaml.safe_load((run / "resolved_config.yaml").read_text(encoding="utf-8"))
    require(resolved["raw"] == config["raw"] and resolved["paths"] == {
        k: str(v) if v is not None else None for k, v in config["paths"].items()}, "resolved config differs")
    from .download_model import MODEL_ID, PINNED_REVISION, verify_snapshot
    model_dir = Path(config["paths"]["model.pretrained_dir"])
    source = json.loads((model_dir / "source.json").read_text(encoding="utf-8"))
    require(source["model_id"] == MODEL_ID == start["snapshot"]["model_id"]
            and source["revision"] == PINNED_REVISION == start["snapshot"]["revision"], "snapshot identity differs")
    require(not verify_snapshot(model_dir, source), "actual snapshot differs from source.json")
    snapshot_files = {item["path"]: sha256_file(model_dir / item["path"]) for item in source["files"]}
    require(snapshot_files["pytorch_model.bin"] == start["snapshot"]["weights_sha256"], "pretrained weight hash differs")
    for key, value in (("snapshot_files", snapshot_files), ("snapshot_source_sha256", sha256_file(model_dir / "source.json"))):
        if key in start:
            require(start[key] == value, f"historical {key} differs")
    processor_files = {p.relative_to(run / "processor").as_posix(): sha256_file(p)
                       for p in (run / "processor").rglob("*") if p.is_file()}
    if "saved_processor" in start:
        require(start["saved_processor"] == processor_files, "historical saved processor differs")
    current = {name: sha256_file(paths.PROJECT_ROOT / name) for name in start["code"]}
    drift = {k: {"trained": v, "current": current[k]} for k, v in start["code"].items() if v != current[k]}
    return start, {"ok": True, "start_end_unchanged": True, "current_data_sha256": actual,
                   "historical_code_sha256": start["code"], "current_code_sha256": current,
                   "code_changed_since_training": drift, "code_equality_required": False,
                   "snapshot_files_current_sha256": snapshot_files,
                   "snapshot_source_current_sha256": sha256_file(model_dir / "source.json"),
                   "historical_snapshot_files_recorded": "snapshot_files" in start,
                   "saved_processor_sha256": processor_files, "test_images_read": False}


class ForwardObserver:
    """Fail on observed CPU, half precision, training, gradients, autocast or TF32."""
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.count = 0
        self.devices: set[str] = set()
        self.handles = [model.register_forward_pre_hook(self.before, with_kwargs=True),
                        model.register_forward_hook(self.after), model.register_full_backward_hook(self.backward)]

    def before(self, model: Any, args: Any, kwargs: Any) -> None:
        pixels = kwargs.get("pixel_values", args[0] if args else None)
        require(isinstance(pixels, torch.Tensor) and pixels.is_cuda, "actual forward input must be CUDA")
        require(pixels.dtype == torch.float32 and pixels.shape[0] == 1, "actual forward must be B1 FP32")
        require(not model.training and not torch.is_grad_enabled() and torch.is_inference_mode_enabled(),
                "actual forward must use eval/inference_mode without gradients")
        require(not torch.is_autocast_enabled("cuda") and not torch.backends.cuda.matmul.allow_tf32
                and not torch.backends.cudnn.allow_tf32, "actual forward must disable autocast and TF32")
        require(all(t.is_cuda and t.device == pixels.device and (not t.is_floating_point() or t.dtype == torch.float32)
                    for t in [*model.parameters(), *model.buffers()]), "actual model must be CUDA FP32")
        self.devices.add(str(pixels.device))

    def after(self, model: Any, args: Any, output: Any) -> None:
        logits = output.logits
        require(logits.is_cuda and logits.dtype == torch.float32 and logits.shape[0] == 1
                and bool(torch.isfinite(logits).all()), "actual logits must be finite CUDA B1 FP32")
        self.count += 1

    def backward(self, *args: Any) -> None:
        raise ValueError("backward is forbidden during read-only acceptance")

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def verify_history(run: Path, summary: dict[str, Any]) -> dict[str, Any]:
    epochs = [json.loads(line) for line in (run / "epochs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    epochs = [row for row in epochs if row.get("event") != "resume"]
    require(epochs == summary["epochs"] and [row["epoch"] for row in epochs] == list(range(1, 51)),
            "epoch history differs or does not contain exactly 50 epochs")
    training = summary["training"]
    require(training["epochs_planned"] == training["epochs_completed"] == 50
            and training["planned_updates"] == training["attempts"] == 13250
            and training["stopped_early"] is False, "formal run incomplete")
    for key, field in (("attempts", "attempts_total"), ("successful_updates", "successful_total"),
                       ("skipped", "skipped_total"), ("scheduler_completed", "scheduler_completed")):
        require(training[key] == epochs[-1][field], f"training total differs: {key}")
    require(max(epochs, key=lambda e: e["val"]["global"]["foreground_iou"])["epoch"] == summary["best"]["epoch"]
            and epochs[-1]["best_so_far"] == summary["best"], "best selection differs")
    updates = [json.loads(line) for line in (run / "updates.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(updates) == training["attempts"], "update log length differs")
    successful = skipped = 0
    schedule = WarmupCosineLR(**{k: summary["scheduler"][k] for k in ("base_lr", "min_lr", "total_steps", "warmup_steps")})
    for attempt, row in enumerate(updates, 1):
        require(row["attempt"] == attempt and row["epoch"] == (attempt - 1) // 265 + 1
                and row["scheduler_completed"] == successful, f"update boundary differs at {attempt}")
        require(math.isclose(row["lr"], schedule.lr_at(successful), rel_tol=1e-12, abs_tol=1e-15),
                f"update LR differs at {attempt}")
        if row["skipped"]:
            skipped += 1
            require(row["success_index"] is None and row["optimizer_step_executed"] is False
                    and row["skip_reason"] == "nonfinite_grad"
                    and row["scale_after"] < row["scale_before"], f"unproven historical skip at {attempt}")
        else:
            successful += 1
            require(row["success_index"] == successful and row["optimizer_step_executed"] is True,
                    f"historical update count differs at {attempt}")
        if attempt % 265 == 0:
            entry = epochs[attempt // 265 - 1]
            require((entry["attempts_total"], entry["successful_total"], entry["skipped_total"], entry["scheduler_completed"])
                    == (attempt, successful, skipped, successful), "epoch counters differ from update history")
    require((successful, skipped) == (training["successful_updates"], training["skipped"]), "update totals differ")
    require(math.isclose(training["final_lr"], schedule.lr_at(successful), rel_tol=1e-12, abs_tol=1e-15),
            "training final LR differs")
    return {"ok": True, "epochs": 50, "attempts": len(updates), "successful": successful, "skipped": skipped,
            "historical_skip_evidence": "optimizer execution flag, nonfinite_grad reason, scale decrease, counters and scheduler",
            "historical_skip_tensor_identity_recorded": False}


def verify_run(run: Path, config: dict[str, Any], device: str) -> dict[str, Any]:
    started = time.perf_counter()
    # This is the only mode: actual CUDA validation of a completed formal run.
    require(torch.device(device).type == "cuda" and torch.cuda.is_available(), "full acceptance requires CUDA")
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    from .train_full import formal_failures
    require(summary["schema"] == "segformer-lab/full-training/1" and summary["stage"] == "11"
            and summary["kind"] == "formal" and summary["not_a_baseline"] is False
            and summary["resume_fixture"] is False and not formal_failures(config), "formal schema/config guards failed")
    inventory = pilot.verify_inventory(run, summary)  # MUST precede torch.load.
    required = {"checkpoints/best.pt", "checkpoints/last.pt", "checkpoints/model_state.pt", "model_config.json",
                "processor/preprocessor_config.json", "resolved_config.yaml", "code_identity_start.json",
                "code_identity_end.json", "epochs.jsonl", "updates.jsonl", "pre_save_probabilities.npz"}
    require(required <= set(summary["artifacts"]["files"]), "required full artifacts missing")
    before = pilot._manifest_of(run)
    try:
        identity, identity_check = verify_identity(run, config, summary)
        history = verify_history(run, summary)
        # NumPy RNG objects require weights_only=False. All are existing local
        # project bundles whose actual SHA256/size have just been checked.
        best = torch.load(run / "checkpoints/best.pt", map_location="cpu", weights_only=False)
        last = torch.load(run / "checkpoints/last.pt", map_location="cpu", weights_only=False)
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        model = SegformerForSemanticSegmentation(pilot._checkpoint_config(best))
        optimizer, _ = build_optimizer(model, float(config["raw"]["train"]["learning_rate"]),
                                       float(config["raw"]["train"]["weight_decay"]))
        shapes = {index: tuple(parameter.shape) for group, saved in zip(optimizer.param_groups,
                   optimizer.state_dict()["param_groups"]) for index, parameter in zip(saved["params"], group["params"])}
        checks = {kind: verify_checkpoint(payload, kind, summary, identity, parameter_shapes=shapes)
                  for kind, payload in (("best", best), ("last", last))}
        normalize = lambda value: json.loads(json.dumps(value))
        require(normalize(best["model_config"]) == normalize(last["model_config"])
                == json.loads((run / "model_config.json").read_text(encoding="utf-8")), "saved model configs differ")
        model.load_state_dict(last["model"], strict=True)
        model.load_state_dict(best["model"], strict=True)
        mirror = torch.load(run / "checkpoints/model_state.pt", map_location="cpu", weights_only=True)
        require(set(mirror) == set(best["model"]) and all(torch.equal(mirror[k], best["model"][k]) for k in mirror),
                "inference mirror differs from best")
        model.load_state_dict(mirror, strict=True)
        processor = SegformerImageProcessor.from_pretrained(str(run / "processor"), local_files_only=True)
        data._processor_statistics(processor)
        val_ds = data.PCB1Dataset(config, "val", processor=processor, flip="none", return_original_mask=True)
        preview_ds = data.PCB1Dataset(config, "val", processor=processor, flip="none", return_original_mask=True,
                                     ids_file=config["paths"]["logging.fixed_preview_manifest"])
        require(len(val_ds) == 132 and len(preview_ds) == 6, "requires full132 val and fixed6 preview")
        require(val_ds.processor is processor and preview_ds.processor is processor, "saved processor not bound")
        del optimizer, last, mirror
        model.eval().requires_grad_(False).to(device)
        observer = ForwardObserver(model)
        try:
            val = evaluate_dataset(model, val_ds, config, device)
            preview = evaluate_dataset(model, preview_ds, config, device, keep_probabilities=True)
        finally:
            observer.close()
        fresh = pilot._val_summary(val)
        saved = best["metrics"]["val"]
        metric_check = pilot.compare_metrics(saved, fresh)
        require(fresh["images"] == {"total": 132, "normal": 120, "anomaly": 12}, "val membership/count differs")
        require(observer.count == 138 and len(observer.devices) == 1, "observed forward coverage incomplete")
        require(all(torch.equal(value.detach().cpu(), best["model"][key]) for key, value in model.state_dict().items()),
                "model state changed during read-only inference")
        with np.load(run / "pre_save_probabilities.npz", allow_pickle=False) as stored:
            probability_check = pilot.compare_probabilities(stored, pilot._preview_npz(preview), threshold=.5, atol=1e-5)
        require(metric_check["ok"] and probability_check["ok"], "reloaded validation/probabilities differ")
        preview_metric_check = pilot.compare_metrics(summary["best_preview"]["preview_metrics"], pilot._val_summary(preview))
        require(preview_metric_check["ok"], "fixed preview metrics differ")
        result = {"schema": SCHEMA, "ok": True, "run_dir": str(run), "not_test_evaluation": True,
                  "test_images_read": False, "optimizer_updates": 0, "backward_calls": 0,
                  "resume_exercised": False, "elapsed_seconds": round(time.perf_counter() - started, 3),
                  "checks": {"artifact_inventory": inventory, "frozen_identity": identity_check,
                             "history": history, "checkpoints": checks, "complete_val_metrics": metric_check,
                             "preview_metrics": preview_metric_check},
                  "checkpoint_loading": {"strict_best": True, "strict_last": True, "strict_mirror": True,
                                         "mirror_matches_best": True, "model_configs_identical": True,
                                         "all_model_tensors_unchanged_after_inference": True},
                  "processor": {"path": str(run / "processor"), "loaded_without_flag_overrides": True,
                                "used_by_val_dataset": True, "used_by_preview_dataset": True},
                  "observed_forward": {"count": observer.count, "val_count": 132, "preview_count": 6,
                                       "devices": sorted(observer.devices), "batch_size": 1,
                                       "model_input_logits_float32": True, "finite_logits": True,
                                       "eval_mode": True, "inference_mode": True, "grad_enabled": False,
                                       "autocast_enabled": False, "tf32_enabled": False},
                  "evaluation": val["evaluation"], "val_metrics_saved": saved, "val_metrics_reloaded": fresh,
                  "preview_probability_comparison": probability_check,
                  "process": {"executable": sys.executable, "python": sys.version, "torch": torch.__version__,
                              "numpy": np.__version__, "cuda_build": torch.version.cuda,
                              "cuda_device_name": torch.cuda.get_device_name(torch.device(device)),
                              "environment": {k: os.environ.get(k) for k in ("CONDA_PREFIX", "SEGFORMER_RESOURCE",
                                  "HF_HOME", "HF_HUB_CACHE", "TORCH_HOME", "XDG_CACHE_HOME", "TEMP", "TMP", "PYTHONPATH")}},
                  "verification_source_sha256": {name: sha256_file(paths.PROJECT_ROOT / name) for name in
                      ("segformer_lab/full_check.py", "tests/test_full_check.py", "segformer_lab/pilot.py",
                       "segformer_lab/training.py", "segformer_lab/data.py", "segformer_lab/evaluation.py")}}
    finally:
        after = pilot._manifest_of(run)
        require(before == after, "original run bytes changed during read-only acceptance")
    result["raw_run_unchanged"] = {"ok": True, "files_checked": before["file_count"],
        "artifact_manifest_sha256_before": before["files"]["artifact_manifest.json"]["sha256"],
        "artifact_manifest_sha256_after": after["files"]["artifact_manifest.json"]["sha256"],
        "all_actual_file_sizes_and_sha256_identical": True}
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    run, out = paths.resolve(args.run), paths.resolve(args.out)
    config = load_config(args.config)
    protected = [paths.PROJECT_ROOT / name for name in ("segformer_lab", "tests", "configs", "manifests", ".git")]
    protected += [Path(config["path"]), *[Path(p) for p in config["paths"].values() if p is not None]]
    # output_root contains runs but is not an input; an independent report may
    # live beside them. All actual data/model/config/run trees stay protected.
    protected = [p for p in protected if p != config["paths"]["experiment.output_root"]]
    require(out.is_relative_to(paths.REPORTS_DIR.resolve()) or out.parent == run.parent,
            "report must be in project reports or beside the original run")
    guard_report_path(run, out, protected=protected)
    try:
        result = verify_run(run, config, args.device)
    except Exception as exc:
        result = {"schema": SCHEMA, "ok": False, "run_dir": str(run), "not_test_evaluation": True,
                  "optimizer_updates": 0, "backward_calls": 0, "error": f"{type(exc).__name__}: {exc}"}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=out.parent, delete=False, suffix=".tmp") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        temporary = Path(handle.name)
    os.replace(temporary, out)
    print(f"full acceptance ok={result['ok']}; wrote {out}")
    if not result["ok"]:
        print(result["error"], file=sys.stderr)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
