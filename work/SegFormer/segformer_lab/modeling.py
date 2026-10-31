"""Load the pinned MiT-B0 encoder into a two-class SegFormer and prove it worked.

The point of this module is not "the model object exists" but "the encoder
weights really came from the pinned snapshot". ``from_pretrained`` with
``ignore_mismatched_sizes=True`` will happily leave a randomly initialised
encoder in place if the keys do not line up, and the usual log line
("some weights were not initialized") is easy to skim past. So after loading we
compare every encoder tensor against the checkpoint file itself and report the
result as counts, not as prose.

What is expected to differ
--------------------------
* ``decode_head.*`` is new: it must be *missing* from the checkpoint.
* ``decode_head.classifier.*`` is also new and additionally *mismatched* in
  shape, because the checkpoint (if any) has 1000 classes and we ask for 2.
* Anything else missing or mismatched is a real problem and fails the check.

The encoder is never frozen here, and nothing in this module writes to the
snapshot directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

ENCODER_PREFIX = "segformer.encoder."
SEGMENTATION_EXPECTED_MISSING_PREFIXES = ("decode_head.",)
EXPECTED_UNUSED = {"classifier.weight", "classifier.bias"}


@dataclass
class LoadedModel:
    model: Any
    processor: Any
    loading_info: dict[str, Any]
    processor_info: dict[str, Any]
    encoder_check: dict[str, Any]
    param_summary: dict[str, Any]
    snapshot: dict[str, Any] = field(default_factory=dict)


def load_processor(model_dir: Path) -> tuple[Any, dict[str, Any]]:
    """Build the image processor with resizing disabled.

    The experiment does its own letterbox; letting the processor resize again
    would silently squash the 512x512 tensor we already prepared.
    """
    from transformers import SegformerImageProcessor

    processor = SegformerImageProcessor.from_pretrained(
        str(model_dir),
        local_files_only=True,
        do_resize=False,
        do_reduce_labels=False,
        do_rescale=True,
        do_normalize=True,
    )
    info = {
        "class": type(processor).__name__,
        "do_resize": bool(getattr(processor, "do_resize", True)),
        "do_rescale": bool(getattr(processor, "do_rescale", True)),
        "do_normalize": bool(getattr(processor, "do_normalize", True)),
        "do_reduce_labels": bool(getattr(processor, "do_reduce_labels", True)),
        "size": dict(getattr(processor, "size", {}) or {}),
        "rescale_factor": getattr(processor, "rescale_factor", None),
        "image_mean": list(getattr(processor, "image_mean", []) or []),
        "image_std": list(getattr(processor, "image_std", []) or []),
    }
    return processor, info


def load_segformer(
    model_dir: Path,
    num_labels: int = 2,
    id2label: dict[int, str] | None = None,
    ignore_index: int = 255,
) -> tuple[Any, dict[str, Any]]:
    """Load ``SegformerForSemanticSegmentation`` from a local snapshot only."""
    from transformers import SegformerForSemanticSegmentation

    id2label = id2label or {0: "background", 1: "defect"}
    label2id = {name: index for index, name in id2label.items()}
    model, info = SegformerForSemanticSegmentation.from_pretrained(
        str(model_dir),
        local_files_only=True,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        semantic_loss_ignore_index=ignore_index,
        ignore_mismatched_sizes=True,
        output_loading_info=True,
    )
    return model, dict(info)


def _checkpoint_tensors(model_dir: Path) -> tuple[dict[str, torch.Tensor], str]:
    """Read the raw checkpoint tensors without going through transformers."""
    safetensors_path = model_dir / "model.safetensors"
    bin_path = model_dir / "pytorch_model.bin"
    if safetensors_path.is_file():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path), device="cpu"), safetensors_path.name
    if bin_path.is_file():
        state = torch.load(str(bin_path), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        return {str(k): v for k, v in state.items()}, bin_path.name
    raise FileNotFoundError(f"no weight file in {model_dir}")


def verify_encoder_weights(model: Any, model_dir: Path, max_report: int = 10) -> dict[str, Any]:
    """Compare every encoder tensor in the model with the checkpoint file.

    Matching is by name after removing the ``segformer.`` wrapper that the
    segmentation head adds, so ``segformer.encoder.x`` is compared with
    ``encoder.x`` from the MiT checkpoint.
    """
    checkpoint, filename = _checkpoint_tensors(model_dir)
    model_state = model.state_dict()
    encoder_items = {k: v for k, v in model_state.items() if k.startswith(ENCODER_PREFIX)}

    matched_exact = 0
    matched_allclose = 0
    max_abs_diff = 0.0
    shape_mismatch: list[dict[str, Any]] = []
    missing_in_model: list[str] = []
    used_checkpoint_keys: set[str] = set()

    for model_key, tensor in encoder_items.items():
        stripped = model_key[len("segformer.") :]
        candidates = [stripped, "segformer." + stripped]
        source_key = next((key for key in candidates if key in checkpoint), None)
        if source_key is None:
            missing_in_model.append(model_key)
            continue
        source = checkpoint[source_key]
        used_checkpoint_keys.add(source_key)
        if tuple(source.shape) != tuple(tensor.shape):
            shape_mismatch.append(
                {"model_key": model_key, "checkpoint_key": source_key,
                 "model_shape": list(tensor.shape), "checkpoint_shape": list(source.shape)}
            )
            continue
        if source.dtype != torch.float32 or tensor.dtype != torch.float32:
            shape_mismatch.append({"model_key": model_key, "reason": "encoder must be FP32"})
            continue
        diff = (source - tensor).abs().max().item()
        max_abs_diff = max(max_abs_diff, float(diff))
        if diff == 0.0:
            matched_exact += 1
        else:
            shape_mismatch.append(
                {"model_key": model_key, "checkpoint_key": source_key, "max_abs_diff": float(diff)}
            )

    unused_checkpoint = sorted(set(checkpoint) - used_checkpoint_keys)
    encoder_param_count = sum(t.numel() for t in encoder_items.values())
    checkpoint_param_count = sum(int(t.numel()) for t in checkpoint.values())

    ok = (
        not shape_mismatch
        and not missing_in_model
        and matched_exact == len(encoder_items)
        and len(encoder_items) > 0
        and max_abs_diff == 0.0
        and set(unused_checkpoint) == EXPECTED_UNUSED
    )
    return {
        "checkpoint_file": filename,
        "checkpoint_tensor_count": len(checkpoint),
        "checkpoint_param_count": checkpoint_param_count,
        "model_encoder_tensor_count": len(encoder_items),
        "model_encoder_param_count": encoder_param_count,
        "matched_exact": matched_exact,
        "matched_allclose": matched_allclose,
        "max_abs_diff": max_abs_diff,
        "shape_mismatch": shape_mismatch,
        "model_encoder_tensors_without_checkpoint": missing_in_model,
        "checkpoint_tensors_not_used": unused_checkpoint,
        "checkpoint_tensors_not_used_count": len(unused_checkpoint),
        "ok": bool(ok),
    }


def classifier_head_report(model: Any) -> dict[str, Any]:
    """Describe how the new two-class head was initialised."""
    report: dict[str, Any] = {"found": False}
    classifier = getattr(getattr(model, "decode_head", None), "classifier", None)
    if classifier is None:
        return report
    weight = classifier.weight.detach().float()
    bias = classifier.bias.detach().float() if classifier.bias is not None else None
    report.update(
        found=True,
        type=type(classifier).__name__,
        in_channels=int(classifier.in_channels),
        out_channels=int(classifier.out_channels),
        kernel_size=list(classifier.kernel_size) if isinstance(classifier.kernel_size, tuple) else classifier.kernel_size,
        weight_shape=list(weight.shape),
        weight_mean=float(weight.mean()),
        weight_std=float(weight.std()),
        weight_min=float(weight.min()),
        weight_max=float(weight.max()),
        weight_all_zero=bool(torch.count_nonzero(weight) == 0),
        bias_mean=float(bias.mean()) if bias is not None else None,
        bias_std=float(bias.std()) if bias is not None else None,
    )
    return report


def param_summary(model: Any) -> dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder = sum(p.numel() for name, p in model.named_parameters() if name.startswith(ENCODER_PREFIX))
    head = total - encoder
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "encoder_params": int(encoder),
        "head_params": int(head),
        "encoder_trainable": all(p.requires_grad for name, p in model.named_parameters()
                                 if name.startswith(ENCODER_PREFIX)),
    }


def classify_loading_keys(loading_info: dict[str, Any]) -> dict[str, Any]:
    """Split missing/mismatched keys into expected (head) and unexpected."""
    missing = list(loading_info.get("missing_keys", []) or [])
    unexpected = list(loading_info.get("unexpected_keys", []) or [])
    mismatched = [entry[0] if isinstance(entry, (list, tuple)) else entry
                  for entry in (loading_info.get("mismatched_keys", []) or [])]

    expected_missing = [k for k in missing if k.startswith(SEGMENTATION_EXPECTED_MISSING_PREFIXES)]
    unexpected_missing = [k for k in missing if k not in expected_missing]
    expected_mismatched = [k for k in mismatched if k.startswith(SEGMENTATION_EXPECTED_MISSING_PREFIXES)]
    unexpected_mismatched = [k for k in mismatched if k not in expected_mismatched]

    return {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "mismatched_keys": mismatched,
        "missing_head_keys": expected_missing,
        "missing_other_keys": unexpected_missing,
        "mismatched_head_keys": expected_mismatched,
        "mismatched_other_keys": unexpected_mismatched,
        "unexpected_other_keys": [k for k in unexpected if k not in EXPECTED_UNUSED],
        "encoder_fully_loaded": (not unexpected_missing and not unexpected_mismatched
                                 and not (set(unexpected) - EXPECTED_UNUSED)),
    }


def loading_checks(loaded: LoadedModel) -> dict[str, bool]:
    """Acceptance gates shared by the reusable loader and runtime report."""
    processor = loaded.processor_info
    params = loaded.param_summary
    head = params.get("classifier_head", {})
    return {
        "encoder_exact_fp32": loaded.encoder_check.get("ok") is True,
        "loading_keys_expected": loaded.loading_info.get("encoder_fully_loaded") is True,
        "processor_flags": all(processor.get(k) is v for k, v in {
            "do_resize": False, "do_reduce_labels": False,
            "do_rescale": True, "do_normalize": True}.items()),
        "binary_classifier": head.get("found") is True and head.get("out_channels") == 2,
        "all_parameters_trainable": (params.get("encoder_trainable") is True
                                     and params.get("total_params", 0) > 0
                                     and params.get("total_params") == params.get("trainable_params")),
    }


def load_all(model_dir: Path, num_labels: int = 2, ignore_index: int = 255) -> LoadedModel:
    from .download_model import MODEL_ID, PINNED_REVISION, verify_snapshot, sha256_file

    source_path = model_dir / "source.json"
    if not source_path.is_file():
        raise ValueError("source.json is required before loading")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("model_id") != MODEL_ID or source.get("revision") != PINNED_REVISION:
        raise ValueError("source.json identity differs from pinned nvidia/mit-b0 commit")
    problems = verify_snapshot(model_dir, source)
    if problems:
        raise ValueError(f"source.json snapshot verification failed: {problems}")
    if num_labels != 2 or ignore_index != 255:
        raise ValueError("this protocol requires num_labels=2 and ignore_index=255")
    processor, processor_info = load_processor(model_dir)
    model, loading_info = load_segformer(model_dir, num_labels=num_labels, ignore_index=ignore_index)
    encoder_check = verify_encoder_weights(model, model_dir)
    keys = classify_loading_keys(loading_info)
    head = classifier_head_report(model)
    params = param_summary(model)

    snapshot = source

    loaded = LoadedModel(
        model=model,
        processor=processor,
        loading_info=keys,
        processor_info=processor_info,
        encoder_check=encoder_check,
        param_summary={**params, "classifier_head": head},
        snapshot={
            "dir": str(model_dir),
            "revision": snapshot.get("revision"),
            "model_id": snapshot.get("model_id"),
            "weights": snapshot.get("weights"),
            "files": snapshot.get("files", []),
            "source_sha256": sha256_file(source_path),
            "verified_against_manifest": True,
        },
    )
    checks = loading_checks(loaded)
    if not all(checks.values()):
        raise ValueError(f"model acceptance failed: {[k for k, ok in checks.items() if not ok]}")
    return loaded
