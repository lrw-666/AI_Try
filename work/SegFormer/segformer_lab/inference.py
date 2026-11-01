"""Frozen-run inference channel for stages 12/13.

One place that answers "given this run's saved best, what is the defect
probability for this image?". ``evaluate.py``, ``predict.py`` and
``summarize.py`` all call it, so the resize order, the padding crop and the
scoring cannot drift apart between them.

The channel is deliberately read-only: no backward pass, no optimizer step, no
training entry point. It refuses to build a model from the pretrained snapshot;
the only accepted weights are the tensors stored in the run's ``best.pt``.

Restoration order (identical to training evaluation, stage 07/11):

    low-resolution logits
    -> bilinear to 512x512, align_corners=False
    -> crop the recorded padding (left/top/new_w/new_h)
    -> bilinear to the original H/W
    -> softmax over channels, take the defect channel
    -> threshold at 0.5 (>= 0.5 is foreground)
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

from . import data, geometry, paths
from .evaluation import evaluation_precision
from .metrics import OriginalSpaceEvaluator, restore_defect_probability
from .training import sha256_file


class FrozenRunError(RuntimeError):
    """Raised when a run cannot be used as a frozen inference source."""


FROZEN_BEST_SHA256 = "192c827bef00607f4258e5fadf0f1d97970ca5cc727a0add749be0fba5997485"


def guard_output_directory(out: str | Path, run: str | Path,
                           config: dict[str, Any], *, inputs: tuple[Path, ...] = ()) -> Path:
    """Reject reuse and writes into any source tree before creating output."""
    destination = Path(out).resolve()
    protected = [Path(run), Path(config["path"]), *inputs]
    protected.extend(paths.PROJECT_ROOT / name for name in
                     ("segformer_lab", "tests", "configs", "manifests", "scripts", ".git"))
    protected.extend(Path(value) for key, value in config["paths"].items()
                     if value is not None and key != "experiment.output_root")
    for source in protected:
        source = source.resolve()
        if destination == source or destination.is_relative_to(source) or source.is_relative_to(destination):
            raise SystemExit(f"output would modify protected input: {source}")
    if destination.exists():
        raise SystemExit(f"refusing to reuse an existing output directory: {destination}")
    return destination


@dataclass
class FrozenRun:
    run_dir: Path
    config: dict[str, Any]
    model: Any
    processor: Any
    device: str
    best: dict[str, Any]
    identity: dict[str, Any]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_frozen_run(config: dict[str, Any], run_dir: str | Path, device: str = "cuda") -> FrozenRun:
    """Rebuild the model from the run's own structure and weights (strict)."""
    from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerImageProcessor

    run_dir = Path(run_dir).resolve()
    best_path = run_dir / "checkpoints" / "best.pt"
    for required in (best_path, run_dir / "processor" / "preprocessor_config.json",
                     run_dir / "model_config.json", run_dir / "summary.json",
                     run_dir / "resolved_config.yaml", run_dir / "pre_save_probabilities.npz",
                     run_dir / "artifact_manifest.json"):
        if not required.is_file():
            raise FrozenRunError(f"missing run artefact: {required}")

    # Verify the local bundle BEFORE deserializing optimizer/RNG pickle objects.
    from . import pilot
    summary = _read_json(run_dir / "summary.json")
    try:
        inventory = pilot.verify_inventory(run_dir, summary)
    except (ValueError, KeyError, OSError) as exc:
        raise FrozenRunError(f"run artifact integrity failed: {exc}") from exc
    best_sha = sha256_file(best_path)
    if best_sha != FROZEN_BEST_SHA256 or summary["best"]["epoch"] != 39:
        raise FrozenRunError("run is not the frozen stage-11 best39 checkpoint")
    source_files = pilot._manifest_of(run_dir)["files"]

    # Our own checkpoint carries optimizer/scheduler/RNG objects.
    payload = torch.load(best_path, map_location="cpu", weights_only=False)
    if payload.get("kind") != "best":
        raise FrozenRunError(f"checkpoint kind is {payload.get('kind')!r}, expected 'best'")
    if payload.get("schema") != "segformer-lab/full-checkpoint/1":
        raise FrozenRunError(f"unexpected checkpoint schema {payload.get('schema')!r}")

    run_config_sha = sha256_file(Path(config["path"]))
    if summary["config"]["sha256"] != run_config_sha:
        raise FrozenRunError(
            "the supplied config is not the one this run was trained with "
            f"({run_config_sha[:12]} != {summary['config']['sha256'][:12]})")
    for split in ("train", "val", "test"):
        path = Path(config["paths"][f"data.{split}_manifest"])
        if sha256_file(path) != summary["identity"]["start"]["manifests"][split]["sha256"]:
            raise FrozenRunError(f"{split} manifest differs from the run identity")
    if sha256_file(Path(config["paths"]["data.split_meta"])) != \
            summary["identity"]["start"]["split_meta_sha256"]:
        raise FrozenRunError("split_meta differs from the run identity")
    if sha256_file(config["paths"]["logging.fixed_preview_manifest"]) != \
            summary["identity"]["start"]["preview_ids_sha256"]:
        raise FrozenRunError("fixed preview IDs differ from the run identity")
    import yaml

    resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    if resolved["raw"] != config["raw"] or resolved["paths"] != {
            key: str(value) if value is not None else None for key, value in config["paths"].items()}:
        raise FrozenRunError("saved resolved config differs from the supplied config")

    if (int(payload["epoch_completed"]), int(payload["step"]["successful"])) != (39, 10330):
        raise FrozenRunError("frozen best epoch/step differs from best39/10330")
    entry = next(row for row in summary["epochs"] if row["epoch"] == 39)
    if payload["metrics"]["val"] != entry["val"]:
        raise FrozenRunError("checkpoint validation record differs from saved best epoch")

    model_config_file = _read_json(run_dir / "model_config.json")
    payload_config = payload["model_config"]
    # JSON serialisation normalises integer id2label keys, without dropping them.
    if model_config_file != json.loads(json.dumps(payload_config)):
        raise FrozenRunError("run model_config.json disagrees with the checkpoint's model_config")
    if model_config_file.get("id2label") != {"0": "background", "1": "defect"}:
        raise FrozenRunError("model labels must be 0=background, 1=defect")
    for name, tensor in payload["model"].items():
        if not isinstance(tensor, torch.Tensor) or (tensor.is_floating_point() and
                (tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()))):
            raise FrozenRunError(f"model tensor must be finite FP32: {name}")

    # Transformers pops fields from this argument in some versions.
    model = SegformerForSemanticSegmentation(SegformerConfig.from_dict(deepcopy(payload_config)))
    missing, unexpected = model.load_state_dict(payload["model"], strict=True)
    if missing or unexpected:
        raise FrozenRunError(f"strict load failed: missing={missing} unexpected={unexpected}")

    mirror = run_dir / "checkpoints" / "model_state.pt"
    if mirror.is_file():
        mirrored = torch.load(mirror, map_location="cpu", weights_only=True)
        same = set(mirrored) == set(payload["model"]) and all(
            torch.equal(mirrored[key], payload["model"][key]) for key in mirrored)
        if not same:
            raise FrozenRunError("model_state.pt is not the same model as best.pt")

    processor = SegformerImageProcessor.from_pretrained(
        str(run_dir / "processor"), local_files_only=True)
    data._processor_statistics(processor)
    if not np.isfinite(float(processor.rescale_factor)) or float(processor.rescale_factor) != 1 / 255:
        raise FrozenRunError("saved processor rescale_factor must be 1/255")

    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.requires_grad_(False)

    identity = {
        "run_dir": str(run_dir),
        "best_path": str(best_path),
        "best_sha256": best_sha,
        "best_epoch": int(payload["epoch_completed"]),
        "best_global_step": int(payload["step"]["successful"]),
        "best_lr": float(payload["lr"]),
        "model_config_sha256": sha256_file(run_dir / "model_config.json"),
        "processor_sha256": sha256_file(run_dir / "processor" / "preprocessor_config.json"),
        "processor_loaded_without_flag_overrides": True,
        "processor_parameters": processor.to_dict(),
        "source_files": source_files,
        "artifact_integrity": inventory,
        "preview_ids_sha256": sha256_file(config["paths"]["logging.fixed_preview_manifest"]),
        "model_state_sha256": sha256_file(mirror) if mirror.is_file() else None,
        "summary_sha256": sha256_file(run_dir / "summary.json"),
        "config_sha256": run_config_sha,
        "resolved_config_sha256": sha256_file(run_dir / "resolved_config.yaml"),
        "manifests": summary["identity"]["start"]["manifests"],
        "split_meta_sha256": summary["identity"]["start"]["split_meta_sha256"],
        "snapshot": summary["identity"]["start"]["snapshot"],
        "trained_epochs": summary["training"]["epochs_completed"],
        "selection": summary["selection"],
        "strict_load": True,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "trainable_parameters_left": int(sum(p.requires_grad for p in model.parameters())),
    }
    return FrozenRun(run_dir=run_dir, config=config, model=model, processor=processor,
                     device=device, best=payload, identity=identity)


def forward_logits(model: Any, item: dict[str, Any], device: str) -> torch.Tensor:
    """Single-sample FP32 logits on CPU (B1, inference_mode)."""
    pixels = item["pixel_values"].unsqueeze(0).to(device)
    if model.training or pixels.dtype != torch.float32 or pixels.shape[0] != 1:
        raise FrozenRunError("forward requires eval mode and B1 FP32 inputs")
    with torch.inference_mode():
        logits = model(pixel_values=pixels).logits
    if logits.dtype != torch.float32 or logits.ndim != 4 or logits.shape[:2] != (1, 2) or \
            not bool(torch.isfinite(logits).all()):
        raise FrozenRunError("forward requires finite B1 two-channel FP32 logits")
    return logits[0].float().cpu()


def probability_map(model: Any, item: dict[str, Any], device: str,
                    config: dict[str, Any]) -> np.ndarray:
    """Original-size defect probability for one item, under the IEEE FP32 policy.

    ``config`` is required on purpose: without the ``evaluation_precision``
    context this host keeps ``cudnn.allow_tf32=True``, which shifts a few
    boundary pixels relative to the scored path. Single-image prediction must
    not quietly use a different precision policy than evaluation.
    """
    with evaluation_precision(config, device=device):
        logits = forward_logits(model, item, device)
    return restore_defect_probability(logits, item["geometry"]).numpy()


def stream_split(
    frozen: FrozenRun,
    dataset: Any,
    *,
    threshold: float,
    on_sample: Callable[[int, dict[str, Any], np.ndarray], None],
    keep_ids: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate a split one image at a time; never hold all probabilities.

    Returns ``(evaluator_result, kept_probabilities)``. Only ids listed in
    ``keep_ids`` (the six fixed preview images) keep their float map, which is
    what the preview comparison needs and nothing more.
    """
    evaluator = OriginalSpaceEvaluator(threshold=threshold)
    normal_ids = {row["image"] for row in dataset.rows if row["label"] == "normal"}
    kept: dict[str, np.ndarray] = {}
    with evaluation_precision(frozen.config, device=frozen.device) as precision_record:
        for index in range(len(dataset)):
            item = dataset[index]
            logits = forward_logits(frozen.model, item, frozen.device)
            probability = restore_defect_probability(logits, item["geometry"]).numpy()
            evaluator.add_sample(
                logits, item["original_mask"], item["geometry"], image_id=item["image_id"],
                label="normal" if item["image_id"] in normal_ids else "anomaly",
                split=dataset.split, flipped=bool(item["flipped"]))
            on_sample(index, item, probability)
            if keep_ids and item["image_id"] in keep_ids:
                kept[item["image_id"].replace("/", "_")] = probability
        result = evaluator.result()
    result["evaluation"] = {
        "batch_size": 1,
        "mode": "model.eval() + torch.inference_mode()",
        "precision": "ieee_fp32 (TF32 disabled, autocast disabled)",
        "precision_record": precision_record,
        "threshold": threshold,
    }
    return result, kept


def prepare_image(processor: Any, image_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Reproduce ``PCB1Dataset.prepared_sample`` for an image that has no mask.

    Kept here (not in predict.py) so the single-image path and the dataset path
    cannot diverge; ``tests/test_inference_eval.py`` asserts equality against a
    real dataset sample.
    """
    input_size = int(config["input_size"])
    pad_rgb = tuple(config["image_pad_rgb"])
    with Image.open(image_path) as handle:
        handle.load()
        image = handle.convert("RGB")
        width, height = image.size
    box = geometry.compute_letterbox(width, height, input_size)
    canvas = geometry.letterbox_image(image, box, pad_rgb)
    pixel_values = processor(images=canvas, return_tensors="pt")["pixel_values"][0].float()
    return {
        "pixel_values": pixel_values,
        "geometry": {
            "orig_w": int(width), "orig_h": int(height), "input_size": input_size,
            "scale": float(box.scale), "new_w": int(box.new_w), "new_h": int(box.new_h),
            "pad_left": int(box.pad_left), "pad_top": int(box.pad_top),
            "pad_right": int(box.pad_right), "pad_bottom": int(box.pad_bottom),
            "valid_area": int(box.valid_area),
            "padding_rows": [int(box.pad_top), int(box.pad_top + box.new_h)],
            "ignore_index": int(config["ignore_index"]),
        },
        "source_size": [int(height), int(width)],
    }


def build_dataset(config: dict[str, Any], split: str, processor: Any | None = None) -> Any:
    """Split dataset with augmentation off and original-size GT kept."""
    return data.PCB1Dataset(config, split, processor=processor, flip="none",
                            return_original_mask=True)


def save_binary_mask(path: Path, mask: np.ndarray) -> None:
    """uint8 0/1 PNG at original size (display conversion belongs elsewhere)."""
    if mask.dtype != np.uint8:
        raise ValueError("mask must be uint8")
    values = set(np.unique(mask).tolist())
    if not values <= {0, 1}:
        raise ValueError(f"mask must contain only 0/1, found {sorted(values)[:5]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(path, format="PNG", optimize=True)
