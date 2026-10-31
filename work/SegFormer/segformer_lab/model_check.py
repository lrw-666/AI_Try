"""Stage 5 runtime check: manual tensors, one forward, one backward, one step.

This is deliberately *not* the training loop and *not* the Dataset. It answers
five concrete questions with numbers attached:

1. Does the pinned snapshot load into a two-class SegFormer with the encoder
   fully matched and only the new decode head reported as missing?
2. What are the real logits shapes for a 512x512 input (not what the docs say)?
3. Is the loss computed on the valid region identical to the loss computed with
   ``ignore_index=255`` over the whole padded canvas?
4. Do the encoder and the decode head both receive finite, non-zero gradients,
   and do their parameters actually change after one optimiser step?
5. How much GPU memory does batch 1 need, and does batch 2 still fit?

The debug step updates the in-memory model only. The snapshot directory is
hashed before and after to prove no weight file was overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from . import paths
from .config import load_config, write_resolved_config
from .modeling import ENCODER_PREFIX, load_all, loading_checks
from .geometry import compute_letterbox

SCHEMA_LOADING = "segformer-lab/model-loading/1"
SCHEMA_CHECK = "segformer-lab/model-check/1"
PAD_ROWS = 61  # 1404x1070 letterboxed to 512x390 -> 61 px top and bottom


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_hashes(snapshot_files: list[dict[str, Any]], root: Path) -> dict[str, str]:
    return {item["path"]: sha256_file(root / item["path"]) for item in snapshot_files}


def make_batch(batch_size: int, input_size: int = 512, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic RGB images plus a 0/1/255 label that mimics the letterbox.

    The label is built with exactly the same convention as the real pipeline:
    255 on the padding rows, 0 for background, 1 for a defect block inside the
    valid area. Using random images is fine here: the purpose is to exercise
    shapes, loss and gradients, not to look meaningful.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    images = torch.randn(batch_size, 3, input_size, input_size, generator=generator)
    box = compute_letterbox(1404, 1070, input_size)
    top, left, bottom, right = box.crop_box
    labels = torch.full((batch_size, input_size, input_size), 255, dtype=torch.long)
    labels[:, top:bottom, left:right] = 0
    # Scale the reference defects with the canvas; never overwrite padding.
    for y0, y1, x0, x1 in ((150, 210, 260, 330), (300, 312, 120, 128)):
        y0 = max(top, min(bottom - 1, round(y0 * input_size / 512)))
        y1 = max(y0 + 1, min(bottom, round(y1 * input_size / 512)))
        x0 = max(left, min(right - 1, round(x0 * input_size / 512)))
        x1 = max(x0 + 1, min(right, round(x1 * input_size / 512)))
        labels[:, y0:y1, x0:x1] = 1
    return images, labels


def _first_param_name(model: torch.nn.Module, prefix: str) -> str:
    for name, _ in model.named_parameters():
        if name.startswith(prefix):
            return name
    raise LookupError(f"no parameter with prefix {prefix!r}")


def _grad_report(model: torch.nn.Module, name: str) -> dict[str, Any]:
    param = dict(model.named_parameters())[name]
    grad = param.grad
    if grad is None:
        return {"name": name, "exists": False, "shape": list(param.shape),
                "grad_finite": False, "grad_nonzero": False}
    return {
        "name": name,
        "exists": True,
        "shape": list(param.shape),
        "grad_norm": float(grad.norm().item()),
        "grad_max_abs": float(grad.abs().max().item()),
        "grad_finite": bool(torch.isfinite(grad).all().item()),
        "grad_nonzero": bool(torch.count_nonzero(grad).item() > 0),
        "param_mean": float(param.detach().float().mean().item()),
        "param_std": float(param.detach().float().std().item()),
    }


def measure_memory(model: torch.nn.Module, batch_size: int, input_size: int, device: torch.device,
                   seed: int = 42) -> dict[str, Any]:
    """Forward+backward peak for one batch size; no optimiser step here."""
    images, labels = make_batch(batch_size, input_size, seed=seed)
    images = images.to(device)
    labels = labels.to(device)
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    logits = model(pixel_values=images).logits
    upsampled = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    loss = F.cross_entropy(upsampled, labels, ignore_index=255)
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_alloc = torch.cuda.max_memory_allocated() / 1024 ** 2
    peak_reserved = torch.cuda.max_memory_reserved() / 1024 ** 2
    total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 2
    report = {
        "batch_size": batch_size,
        "logits_shape": list(logits.shape),
        "loss": float(loss.item()),
        "seconds": round(elapsed, 3),
        "peak_allocated_mib": round(peak_alloc, 1),
        "peak_reserved_mib": round(peak_reserved, 1),
        "device_total_mib": round(total, 1),
        "peak_fraction_percent": round(peak_reserved / total * 100.0, 2),
        "precision": "float32",
        "amp_enabled": False,
        "scope": "model + inputs + CE forward/backward; no optimizer or optimizer states",
        "loss_finite": bool(torch.isfinite(loss).item()),
    }
    del images, labels, logits, upsampled, loss
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return report


def runtime_checks(forward: dict[str, Any], grads: list[dict[str, Any]], step: list[dict[str, Any]],
                   snapshot_intact: bool, size: int) -> dict[str, bool]:
    """Fail closed on the measured shape, loss and sampled gradient/update evidence."""
    return {
        "snapshot_files_unchanged": snapshot_intact,
        "expected_shapes": (forward.get("input_shape") == [2, 3, size, size]
                            and forward.get("label_shape") == [2, size, size]
                            and forward.get("logits_shape") == [2, 2, size // 4, size // 4]
                            and forward.get("upsampled_shape") == [2, 2, size, size]),
        "label_semantics": forward.get("label_unique") == [0, 1, 255],
        "ce_finite": all(math.isfinite(forward.get(k, float("nan")))
                         for k in ("ce_full_padded", "ce_valid_region", "ce_difference")),
        "padding_ce_equal": forward.get("ce_difference", float("inf")) <= 1e-6,
        "sampled_gradients_finite_nonzero": bool(grads) and all(
            g.get("exists", False) and g.get("grad_nonzero", False) and g.get("grad_finite", False) for g in grads),
        "sampled_parameters_changed_finitely": bool(step) and all(
            item.get("changed", False) and math.isfinite(item.get("max_abs_delta", float("nan")))
            and math.isfinite(item.get("mean_abs_delta", float("nan"))) for item in step),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--loading-report", default="reports/model_loading.json")
    parser.add_argument("--check-report", default="reports/model_check.json")
    parser.add_argument("--input-size", type=int, default=512)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.input_size != cfg["input_size"]:
        raise SystemExit("--input-size must match the fixed data.input_size in config")
    seed = int(cfg["raw"]["experiment"].get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)  # Must precede the random decode-head initialization.
    torch.cuda.manual_seed_all(seed)
    model_dir: Path = cfg["paths"]["model.pretrained_dir"]
    if not model_dir.is_dir():
        raise SystemExit(f"model snapshot not found: {model_dir}")

    print("SegFormer stage 5: load and check")
    print(f"  snapshot : {model_dir}")
    print(f"  config   : {cfg['path']}")

    weights_rel = "pytorch_model.bin"
    source_path = model_dir / "source.json"
    source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.is_file() else {}
    if source.get("weights"):
        weights_rel = source["weights"]
    loaded = load_all(model_dir, num_labels=cfg["num_labels"], ignore_index=cfg["ignore_index"])
    tracked_files = [*loaded.snapshot["files"], {"path": "source.json"}]
    before_hashes = snapshot_hashes(tracked_files, model_dir)

    loading = {
        "schema": SCHEMA_LOADING,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "transformers_version": __import__("transformers").__version__,
        "snapshot": loaded.snapshot,
        "processor": loaded.processor_info,
        "loading_info": loaded.loading_info,
        "encoder_check": loaded.encoder_check,
        "parameters": loaded.param_summary,
        "seed": seed,
        "acceptance": loading_checks(loaded),
        "ok": all(loading_checks(loaded).values()),
    }
    loading_path = paths.ensure_parent(args.loading_report)
    loading_path.write_text(json.dumps(loading, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  encoder  : {loaded.encoder_check['matched_exact']}/{loaded.encoder_check['model_encoder_tensor_count']} "
          f"tensors exact, max|diff|={loaded.encoder_check['max_abs_diff']}")
    print(f"  missing  : head={len(loaded.loading_info['missing_head_keys'])} "
          f"other={len(loaded.loading_info['missing_other_keys'])}")
    print(f"  mismatched: head={len(loaded.loading_info['mismatched_head_keys'])} "
          f"other={len(loaded.loading_info['mismatched_other_keys'])}")

    if not loaded.encoder_check["ok"]:
        print("  RESULT: FAILED - encoder weights do not match the snapshot", file=sys.stderr)
        return 1
    if loaded.loading_info["missing_other_keys"] or loaded.loading_info["mismatched_other_keys"]:
        print("  RESULT: FAILED - unexpected missing/mismatched keys outside the decode head", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA is not available; this check is defined on GPU")
    model = loaded.model.to(device)
    model.train()

    # Small first, before constructing AdamW or retaining any debug-step graph.
    batch1 = measure_memory(model, 1, args.input_size, device, seed=seed)
    batch2 = measure_memory(model, 2, args.input_size, device, seed=seed)

    # ---------------------------------------------------------------- forward ----
    images, labels = make_batch(2, args.input_size, seed=seed)
    images = images.to(device)
    labels = labels.to(device)
    logits = model(pixel_values=images).logits
    upsampled = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    loss_full = F.cross_entropy(upsampled, labels, ignore_index=255)
    box = compute_letterbox(1404, 1070, args.input_size)
    valid = slice(box.pad_top, box.pad_top + box.new_h)
    loss_valid = F.cross_entropy(upsampled[:, :, valid, :], labels[:, valid, :], ignore_index=255)
    forward = {
        "input_shape": list(images.shape),
        "label_shape": list(labels.shape),
        "label_unique": sorted({int(v) for v in torch.unique(labels)}),
        "logits_shape": list(logits.shape),
        "upsampled_shape": list(upsampled.shape),
        "output_stride": args.input_size // int(logits.shape[-1]),
        "ce_full_padded": float(loss_full.item()),
        "ce_valid_region": float(loss_valid.item()),
        "ce_difference": float(abs(loss_full.item() - loss_valid.item())),
        "padding_rows": [0, box.pad_top, box.pad_top + box.new_h, args.input_size],
    }

    # --------------------------------------------------------------- backward ----
    encoder_name = _first_param_name(model, ENCODER_PREFIX)
    head_name = _first_param_name(model, "decode_head.")
    before = {name: dict(model.named_parameters())[name].detach().clone() for name in (encoder_name, head_name)}
    loss_full.backward()
    grads = [_grad_report(model, encoder_name), _grad_report(model, head_name)]

    # ------------------------------------------------------------------- step ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    optimizer.step()
    after = {name: dict(model.named_parameters())[name].detach() for name in (encoder_name, head_name)}
    step = []
    for name in (encoder_name, head_name):
        delta = (after[name].float() - before[name].float()).abs()
        step.append(
            {
                "name": name,
                "max_abs_delta": float(delta.max().item()),
                "mean_abs_delta": float(delta.mean().item()),
                "changed": bool(torch.count_nonzero(delta).item() > 0),
            }
        )
    optimizer.zero_grad(set_to_none=True)

    after_hashes = snapshot_hashes(tracked_files, model_dir)
    snapshot_intact = before_hashes == after_hashes
    checks = runtime_checks(forward, grads, step, snapshot_intact, args.input_size)
    checks["loading_accepted"] = loading["ok"]
    checks["memory_probes_finite_expected_shapes"] = all(
        probe["loss_finite"] and probe["logits_shape"] == [batch, 2, args.input_size // 4, args.input_size // 4]
        for batch, probe in ((1, batch1), (2, batch2)))

    check = {
        "schema": SCHEMA_CHECK,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "seed": seed,
        "precision": "float32",
        "amp_enabled": False,
        "execution_order": ["memory_batch1", "memory_batch2", "debug_batch2_forward_backward_step"],
        "provenance": {"source_sha256": loaded.snapshot["source_sha256"],
                       "config_sha256": sha256_file(cfg["path"]),
                       "modeling_sha256": sha256_file(Path(__file__).with_name("modeling.py")),
                       "model_check_sha256": sha256_file(Path(__file__)),
                       "snapshot_hashes_before": before_hashes, "snapshot_hashes_after": after_hashes},
        "forward": forward,
        "backward": {"scope": "one sampled parameter per encoder and decode head",
                     "gradients": grads, "loss_backward_finite": checks["ce_finite"] and all(g["grad_finite"] for g in grads)},
        "debug_update": {
            "optimizer": "AdamW(lr=1e-4, weight_decay=0.0), one step",
            "parameters": step,
            "scope": "one sampled parameter per encoder and decode head; in-memory only",
        },
        "memory": {"batch1": batch1, "batch2": batch2},
        "snapshot_integrity": {
            "weights": weights_rel,
            "sha256_before": before_hashes.get(weights_rel),
            "sha256_after": after_hashes.get(weights_rel),
            "unchanged": snapshot_intact,
        },
        "acceptance": checks,
        "ok": all(checks.values()),
    }
    check_path = paths.ensure_parent(args.check_report)
    check_path.write_text(json.dumps(check, indent=2, ensure_ascii=False), encoding="utf-8")
    write_resolved_config(cfg, Path(args.check_report).resolve().parent / "resolved_config.yaml")

    print("")
    print(f"  input        : {forward['input_shape']} labels {forward['label_unique']}")
    print(f"  logits       : {forward['logits_shape']} stride {forward['output_stride']}")
    print(f"  CE padded    : {forward['ce_full_padded']:.6f}")
    print(f"  CE valid     : {forward['ce_valid_region']:.6f} (diff {forward['ce_difference']:.2e})")
    for item in grads:
        print(f"  grad {item['name']}: norm={item['grad_norm']:.6g} nonzero={item['grad_nonzero']}")
    for item in step:
        print(f"  step {item['name']}: max|delta|={item['max_abs_delta']:.3e}")
    print(f"  memory B1    : {batch1['peak_allocated_mib']} MiB allocated / {batch1['peak_reserved_mib']} MiB reserved")
    print(f"  memory B2    : {batch2['peak_allocated_mib']} MiB allocated / {batch2['peak_reserved_mib']} MiB reserved")
    print(f"  snapshot     : unchanged={snapshot_intact}")
    print(f"  loading report: {loading_path}")
    print(f"  check report  : {check_path}")

    if not check["ok"]:
        failed = [k for k, v in check.items() if k in ("ok",) and not v]
        print(f"  RESULT: FAILED {failed}", file=sys.stderr)
        return 1
    print("  RESULT: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
