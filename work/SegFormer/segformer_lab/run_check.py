"""Read-only stage 08/09 artifact, provenance and fresh-process reload check.

No optimiser is constructed and no training/resume path exists. The report and
optional figures must be outside the immutable source run. CPU helper tests do
not import torch; a full acceptance can pass only after actual CUDA forwards.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def run_file(run: Path, name: str) -> Path:
    path = (run / name).resolve()
    require(not Path(name).is_absolute() and path != run.resolve()
            and path.is_relative_to(run.resolve()), f"artifact escapes run boundary: {name}")
    return path


def validate_outputs(run: Path, out: Path, figures: Path | None, protected: tuple[Path, ...] = ()) -> None:
    require(run.is_dir(), f"source run does not exist: {run}")
    require(out.suffix == ".json", "report must have .json extension")
    for path in (out, figures):
        if path is not None:
            require(not path.resolve().is_relative_to(run.resolve()), "output cannot overwrite source run")
            require(all(not path.resolve().is_relative_to(p.resolve()) for p in protected),
                    "output overlaps protected input/source")
    if figures is not None:
        require(not figures.exists() or (figures.is_dir() and not any(figures.iterdir())),
                "figure directory must be new or empty")


def verify_artifacts(run: Path, summary: dict[str, Any]) -> dict[str, Any]:
    manifest = read_json(run / "artifact_manifest.json")
    require(manifest == summary["artifacts"], "summary.artifacts differs from artifact_manifest")
    require(Path(manifest["run_dir"]).resolve() == run.resolve(), "manifest source run path mismatch")
    actual = {name: identity(run_file(run, name)) for name in manifest["files"]}
    for name, entry in manifest["files"].items():
        require(all(actual[name][key] == entry[key] for key in ("bytes", "sha256")),
                f"artifact identity mismatch: {name}")
    require(actual["checkpoints/model_state.pt"]["sha256"] == manifest["model_state_sha256"],
            "checkpoint SHA disagrees with manifest")
    return {"files_checked": len(actual), "files": actual, "manifest": identity(run / "artifact_manifest.json")}


def compare_evaluations(old: dict[str, Any], fresh: dict[str, Any], split: str) -> dict[str, Any]:
    """Exact comparison of every stored metric/count/NA field; timing excluded."""
    left, right = copy.deepcopy(old), copy.deepcopy(fresh)
    corrections = []
    for row in right["per_image"]:
        require(row["split"] == split, f"fresh per-image split must be {split}")
    require([r["image_id"] for r in left["per_image"]] == [r["image_id"] for r in right["per_image"]],
            "stored/fresh image order differs")
    for row in left["per_image"]:
        if row["split"] != split:
            require(split == "train" and row["split"] == "val", "unrecognised historical split mismatch")
            corrections.append({"image_id": row["image_id"], "stored_split": row["split"], "actual_split": split})
            row["split"] = split
    for value in (left, right):
        for key in ("forward_ms", "median_forward_ms"):
            value["evaluation"].pop(key, None)
    differences = [key for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)]
    return {"identical": not differences, "different_sections": differences,
            "metadata_correction": corrections, "excluded_fields": ["evaluation.forward_ms", "evaluation.median_forward_ms"],
            "scope": "all stored global/per-image counts, metrics, NA reasons and evaluation policy"}


def verify_provenance(config: dict[str, Any], summary: dict[str, Any], run: Path) -> dict[str, Any]:
    from .config import resolved_yaml
    import yaml

    provenance, actual = summary["provenance"], {}
    def anchor(name: str, path: Path, expected: dict[str, Any]) -> None:
        actual[name] = identity(path)
        require(Path(expected["path"]).resolve() == path.resolve()
                and expected["sha256"] == actual[name]["sha256"], f"provenance anchor mismatch: {name}")
    anchor("config", config["path"], summary["config"])
    fields = {"config": config["path"], "ids": config["paths"]["data.train_ids_file"],
              "split_meta": config["paths"]["data.split_meta"],
              "preview_ids": config["paths"]["logging.fixed_preview_manifest"]}
    for name, path in fields.items():
        anchor(name, Path(path), provenance["files"][name])
    for split in ("train", "val", "test"):
        anchor(split, Path(config["paths"][f"data.{split}_manifest"]), provenance["manifest_identity"][split])
    meta = read_json(Path(fields["split_meta"]))
    for split in ("train", "val", "test"):
        require(meta["manifests"][split]["sha256"] == actual[split]["sha256"], "split metadata manifest mismatch")
    snapshot = Path(config["paths"]["model.pretrained_dir"])
    source = read_json(snapshot / "source.json")
    for entry in source["files"]:
        item = identity(run_file(snapshot, entry["path"]))
        require(item["sha256"] == entry["sha256"] and item["bytes"] == entry["size_bytes"], "snapshot identity mismatch")
        actual[f"snapshot/{entry['path']}"] = item
    actual["source_metadata"] = identity(snapshot / "source.json")
    require(provenance["snapshot"] == {"model_id": source["model_id"], "revision": source["revision"],
            "weights_sha256": actual["snapshot/pytorch_model.bin"]["sha256"]}, "snapshot anchor mismatch")
    require(yaml.safe_load((run / "resolved_config.yaml").read_text(encoding="utf-8"))
            == yaml.safe_load(resolved_yaml(config)), "saved resolved config differs from actual configuration")
    from .paths import PROJECT_ROOT
    code = {}
    names = set(provenance["code"]) | {p.relative_to(PROJECT_ROOT).as_posix()
            for folder in ("segformer_lab", "tests") for p in (PROJECT_ROOT / folder).rglob("*.py")}
    for name in sorted(names):
        code[name] = {**identity(run_file(PROJECT_ROOT, name)), "stored_sha256": provenance["code"].get(name)}
        code[name]["matches_stored"] = code[name]["sha256"] == code[name]["stored_sha256"]
        if name.startswith("configs/"):
            require(code[name]["matches_stored"], f"recorded configuration anchor changed: {name}")
    return {"anchors": actual, "current_code": code, "stored_code": provenance["code"],
            "code_note": "source drift is recorded; artifact and data/config anchors must match", "test_access": "manifest hash only; no test images/masks"}


def selected_rows(config: dict[str, Any], summary: dict[str, Any]) -> tuple[str, Path, list[dict[str, str]]]:
    def select(split: str, path: Path) -> list[dict[str, str]]:
        ids = [s.strip() for s in path.read_text(encoding="utf-8").splitlines() if s.strip() and not s.lstrip().startswith("#")]
        with Path(config["paths"][f"data.{split}_manifest"]).open(encoding="utf-8", newline="") as handle:
            by_id = {row["image"]: row for row in csv.DictReader(handle)}
        require(len(ids) == len(set(ids)) and all(i in by_id for i in ids), "IDs must be unique members of their actual split")
        rows = [by_id[i] for i in ids]
        require(all(r["split"] == split for r in rows), "manifest split column mismatch")
        return rows
    train_ids = Path(config["paths"]["data.train_ids_file"])
    train = select("train", train_ids)
    require(len(train) == 8 and [r["label"] for r in train].count("normal") == 4, "train subset must be fixed 4N/4A")
    require(summary["train_subset"]["images"] == [r["image"] for r in train]
            and summary["train_subset"]["labels"] == [r["label"] for r in train]
            and summary["train_subset"]["ids_sha256"] == identity(train_ids)["sha256"], "stored training subset mismatch")
    split = "train" if summary["stage"] == "09" else "val"
    ids = train_ids if split == "train" else Path(config["paths"]["logging.fixed_preview_manifest"])
    rows = train if split == "train" else select(split, ids)
    require(len(rows) == (8 if split == "train" else 6)
            and [r["label"] for r in rows].count("anomaly") == (4 if split == "train" else 3), "evaluation subset composition mismatch")
    return split, ids, rows


def verify_model_config(state_config: dict[str, Any], saved_json: dict[str, Any]) -> None:
    # JSON object keys are strings; torch.save preserves id2label's integer keys.
    require(json.loads(json.dumps(state_config)) == saved_json, "training model configuration differs")


def verify_state(model: Any, state: dict[str, Any], summary: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    import torch

    saved, loaded = state["model"], model.state_dict()
    require(saved.keys() == loaded.keys(), "training/model state key mismatch")
    require(all(torch.isfinite(t).all().item() and torch.equal(t, loaded[k].detach().cpu()) for k, t in saved.items()),
            "training/model tensors must be finite and exactly equal for all keys")
    require(all(t.dtype == torch.float32 for t in loaded.values() if t.is_floating_point()), "model must be FP32")
    counters = {k: state[k] for k in ("successful_updates", "attempts", "skipped")}
    require(all(type(v) is int and v >= 0 for v in counters.values()) and counters["successful_updates"] > 0
            and counters["attempts"] == counters["successful_updates"] + counters["skipped"], "invalid update counters")
    require(all(counters[k] == summary["training"][k] == manifest[k] for k in counters), "stored counters disagree")
    optimizer = state["optimizer"]
    ids = [i for group in optimizer["param_groups"] for i in group["params"]]
    require(len(ids) == len(set(ids)) == sum(1 for p in model.parameters() if p.requires_grad)
            and set(ids) == set(optimizer["state"]), "optimizer must cover every parameter exactly once")
    steps = [float(optimizer["state"][i]["step"]) for i in ids]
    require(all(step == counters["successful_updates"] for step in steps), "optimizer per-parameter step differs from successful count")
    require(all(torch.isfinite(v).all().item() for item in optimizer["state"].values() for v in item.values() if torch.is_tensor(v)), "nonfinite optimizer tensor")
    scaler = state["grad_scaler"]
    require(all(math.isfinite(float(v)) for v in scaler.values()) and scaler["scale"] > 0
            and scaler["growth_factor"] > 1 and 0 < scaler["backoff_factor"] < 1
            and 0 <= scaler["_growth_tracker"] < scaler["growth_interval"], "invalid saved GradScaler state")
    return {"all_model_keys_exact_and_finite": True, "tensor_count": len(saved), "counters": counters,
            "optimizer_parameters_checked": len(ids), "optimizer_step_min": min(steps), "optimizer_step_max": max(steps), "grad_scaler": scaler}


def compare_probabilities(path: Path, samples: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "not_available", "scope": "stored counts/metrics only; old probability maps were not saved",
                "max_abs_probability_diff": None, "xor_pixels_total": None}
    import numpy as np
    from .metrics import restore_defect_probability

    comparisons = []
    with np.load(path, allow_pickle=False) as stored:
        require(set(stored.files) == {s["image_id"].replace("/", "_") for s in samples}, "probability IDs differ")
        for sample in samples:
            old = stored[sample["image_id"].replace("/", "_")]
            fresh = restore_defect_probability(sample["logits"], sample["geometry"]).numpy()
            require(old.shape == fresh.shape and np.isfinite(old).all() and np.isfinite(fresh).all(), "invalid probability map")
            comparisons.append({"image_id": sample["image_id"], "max_abs_probability_diff": float(np.abs(old - fresh).max()),
                                "xor_pixels": int(((old >= threshold) != (fresh >= threshold)).sum())})
    maximum, xor = max(c["max_abs_probability_diff"] for c in comparisons), sum(c["xor_pixels"] for c in comparisons)
    return {"status": "compared", "max_abs_probability_diff": maximum, "xor_pixels_total": xor,
            "atol": 1e-5, "ok": maximum <= 1e-5 and xor == 0, "per_image": comparisons}


def check_run(args: argparse.Namespace, report: dict[str, Any]) -> None:
    import torch
    from . import data
    from .config import load_config
    from .smoke import _eval_summary
    from .training import evaluate_dataset, load_run_model, set_seed

    run = Path(args.run).resolve()
    summary = read_json(run / "summary.json")
    require(summary["stage"] in ("08", "09") and summary["not_a_baseline"] is True, "unsupported run identity")
    config = load_config(args.config)
    require(config["raw"]["eval"]["precision"] == "ieee_fp32" and config["raw"]["eval"]["batch_size"] == 1
            and config["raw"]["eval"]["postprocess"] == "none", "requires B1 IEEE FP32 original-space evaluation")
    require(summary["seed"]["seed"] == config["raw"]["experiment"]["seed"], "stored seed differs from configuration")
    report["source"] = {"run_dir": str(run), "summary": identity(run / "summary.json"),
                        "checkpoint": identity(run / "checkpoints/model_state.pt"), "stage": summary["stage"]}
    report["artifacts"] = verify_artifacts(run, summary)
    report["provenance"] = verify_provenance(config, summary, run)
    split, ids, rows = selected_rows(config, summary)
    device = torch.device(args.device)
    report["process"].update({"torch": torch.__version__, "cuda_build": torch.version.cuda, "device": str(device),
                              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None})
    set_seed(int(summary["seed"]["seed"]))
    model, processor, load_info = load_run_model(run)
    report["load"] = load_info
    require(load_info["strict"] and not load_info["missing_keys"] and not load_info["unexpected_keys"]
            and load_info["model_state_sha256"] == report["source"]["checkpoint"]["sha256"], "strict reload/checkpoint mismatch")
    state = torch.load(run / "checkpoints/training_state.pt", map_location="cpu", weights_only=True)
    verify_model_config(state["model_config"], read_json(run / "model_config.json"))
    report["training_state"] = verify_state(model, state, summary, summary["artifacts"])
    dataset = data.PCB1Dataset(config, split, processor=processor, flip="none", ids_file=ids, return_original_mask=True)
    require(dataset.processor is processor and dataset.rows == rows, "saved processor/subset not actually used")
    report["evaluation_subset"] = {"split": split, "ids_file": identity(ids), "rows": rows, "saved_processor_used": True, "flip": "none"}
    model = model.float().to(device)
    observed = []
    def guard(module: Any, positional: Any, kwargs: Any) -> None:
        pixels = kwargs["pixel_values"]
        require(pixels.dtype == torch.float32 and pixels.shape[0] == 1 and pixels.device.type == device.type
                and not module.training and torch.is_inference_mode_enabled()
                and not torch.is_grad_enabled() and not torch.is_autocast_enabled(device.type), "forward protocol violated")
        observed.append(str(pixels.device))
    hook = model.register_forward_pre_hook(guard, with_kwargs=True)
    probability_file = run / "pre_save_probabilities.npz"
    try:
        result = evaluate_dataset(model, dataset, config, str(device), label_of={r["image"]: r["label"] for r in rows},
                                  keep_probabilities=bool(args.figure_dir) or probability_file.is_file())
    finally:
        hook.remove()
    fresh = _eval_summary(result)
    old = summary["final" if summary["stage"] == "09" else "evaluation_before_save"]
    require([r["image_id"] for r in fresh["per_image"]] == [r["image"] for r in rows], "evaluation IDs differ from fixed subset")
    report["comparison"] = compare_evaluations(old, fresh, split)
    report["probabilities"] = compare_probabilities(probability_file, result["samples"], float(config["raw"]["eval"]["threshold"]))
    report["metrics_before"], report["metrics_after"] = old, fresh
    verify_state(model, state, summary, summary["artifacts"])
    require(verify_artifacts(run, summary) == report["artifacts"], "source artifacts changed during evaluation")
    report["read_only"] = {"optimizer_created": False, "backward_calls": 0, "updates": 0, "all_tensors_unchanged": True, "source_artifacts_unchanged": True}
    if args.figure_dir:
        from .train import _render_sheet, _render_zoom
        folder, threshold = Path(args.figure_dir).resolve(), float(config["raw"]["eval"]["threshold"])
        report["figures"] = [_render_sheet(result["samples"], dataset.data_root, folder / "reload_sheet.png", threshold, "saved-run reload; not a baseline")]
        for sample in result["samples"]:
            if sample["label"] == "anomaly" and Path(sample["image_id"]).stem in ("065", "031"):
                report["figures"].append(_render_zoom(sample, dataset.data_root, folder / f"reload_zoom_{Path(sample['image_id']).stem}.png", threshold, "saved-run reload"))
    report["checks"] = {"artifact_and_provenance_anchors": True, "strict_saved_bundle_loaded": True,
                        "saved_training_state_consistent": True,
                        "all_stored_counts_metrics_equal": report["comparison"]["identical"],
                        "available_probability_maps_equal": report["probabilities"].get("ok"),
                        "read_only": True, "real_cuda_fp32_b1_forwards": device.type == "cuda" and len(observed) == len(rows)}
    report["optional_checks"] = ["available_probability_maps_equal"]
    report["skipped_checks"] = ["available_probability_maps_equal"] if report["probabilities"]["status"] == "not_available" else []
    report["observed_forward_devices"] = observed
    report["ok"] = all(value is True for name, value in report["checks"].items() if name not in report["skipped_checks"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", default="configs/overfit.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--figure-dir")
    args = parser.parse_args(argv)
    run, out = Path(args.run).resolve(), Path(args.out).resolve()
    from .config import load_config
    from .paths import PROJECT_ROOT
    config = load_config(args.config)
    protected = (Path(config["data_root"]), Path(config["paths"]["model.pretrained_dir"]),
                 Path(config["path"]), *(Path(v) for k, v in config["paths"].items() if v is not None and k.startswith(("data.", "logging."))),
                 *(PROJECT_ROOT / name for name in ("configs", "manifests", "segformer_lab", "tests", "scripts", ".git")))
    # Validate before the exception-to-report path: even a failed check must not overwrite its inputs.
    validate_outputs(run, out, Path(args.figure_dir).resolve() if args.figure_dir else None, protected)
    report = {"schema": "segformer-lab/saved-run-check/1", "not_a_baseline": True, "ok": False,
              "process": {"pid": os.getpid(), "python": sys.version.split()[0], "executable": sys.executable}}
    try:
        check_run(args, report)
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved-run check: ok={report['ok']} | {out}")
    if "error" in report:
        print(report["error"], file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
