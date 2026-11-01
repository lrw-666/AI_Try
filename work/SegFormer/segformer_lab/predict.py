"""Stage 12: inference for a single image that has no mask or label.

Only the image path is required. The run supplies structure, weights, processor
and protocol; the geometry/restoration code is the same module ``evaluate.py``
uses, so a prediction cannot disagree with the scored path for a different
reason than "the image was different".

Outputs (all original size):
    mask.png           uint8 0/1, the scoring convention
    mask_display.png   uint8 0/255, display only
    probability.npz    float32 defect probability (single map)
    heatmap.png        probability rendered with a fixed 0-1 scale
    overlay.png        source image with the prediction outlined
    prediction_meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import load_config
from .inference import (guard_output_directory, load_frozen_run, prepare_image,
                        probability_map, save_binary_mask)
from .training import sha256_file


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def do_predict(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    from .evaluate import _code_identity, _manifest_of, _protocol_identity, _runtime_info

    code_before = _code_identity()
    config = load_config(args.config)
    protocol = _protocol_identity(config)
    threshold = float(config["raw"]["eval"]["threshold"])
    image_path = Path(args.image).resolve()
    if not image_path.is_file():
        raise SystemExit(f"image not found: {image_path}")
    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
        raise SystemExit(f"unsupported image extension: {image_path.suffix}")

    out_dir = guard_output_directory(args.out, args.run, config, inputs=(image_path,))
    image_sha_before = sha256_file(image_path)

    frozen = load_frozen_run(config, args.run, device=args.device)
    prepared = prepare_image(frozen.processor, image_path, config)
    probability = probability_map(frozen.model, prepared, args.device, config)
    mask = (probability >= threshold).astype(np.uint8)
    if probability.dtype != np.float32 or probability.shape != tuple(prepared["source_size"]) or \
            not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise SystemExit("prediction must be a finite original-size float32 probability in [0,1]")
    out_dir.mkdir(parents=True, exist_ok=False)

    save_binary_mask(out_dir / "mask.png", mask)
    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(
        out_dir / "mask_display.png", format="PNG")
    np.savez_compressed(out_dir / "probability.npz", probability=probability)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Image.open(image_path) as handle:
        handle.load()
        source = np.asarray(handle.convert("RGB"))

    figure, axis = plt.subplots(figsize=(7.2, 5.6), dpi=110)
    image = axis.imshow(probability, cmap="magma", vmin=0.0, vmax=1.0)
    axis.set_title(f"defect probability (fixed 0-1 scale) | {image_path.name}", fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])
    figure.colorbar(image, ax=axis, fraction=0.036, pad=0.02)
    figure.tight_layout()
    figure.savefig(out_dir / "heatmap.png")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 5.6), dpi=110)
    axis.imshow(source)
    if mask.any():
        axis.contour(mask, levels=[0.5], colors=["#00A6A6"], linewidths=1.0)
    axis.set_title(f"prediction >= {threshold} | {image_path.name}", fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(out_dir / "overlay.png")
    plt.close(figure)

    code_after = _code_identity()
    if code_after != code_before or sha256_file(image_path) != image_sha_before or \
            _manifest_of(frozen.run_dir, exclude=set())["files"] != frozen.identity["source_files"]:
        raise SystemExit("code, input image or original training run changed during prediction")
    meta = {
        "schema": "segformer-lab/predict-meta/1",
        "stage": "12",
        "command": {"argv": sys.argv},
        "image": {"path": str(image_path), "sha256": image_sha_before,
                  "bytes": image_path.stat().st_size},
        "image_uses_gt": False,
        "threshold": threshold,
        "device": args.device,
        "source_size": prepared["source_size"],
        "predicted_pixels": int(mask.sum()),
        "predicted_ratio": round(float(mask.mean()), 10),
        "probability": {"min": float(probability.min()), "max": float(probability.max()),
                        "mean": float(probability.mean())},
        "identity": frozen.identity,
        "process": {"pid": os.getpid(), **_runtime_info(args.device)},
        "started_at": started_at,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": protocol,
        "code": code_before,
        "code_after": code_after,
        "code_and_original_run_unchanged": True,
        "restore_order": "logits -> bilinear 512 -> crop padding -> bilinear original H/W "
                         "-> softmax channel 1 -> threshold >= 0.5",
        "outputs": {"mask": "mask.png (0/1)", "display": "mask_display.png (0/255, display only)",
                    "probability": "probability.npz (float32)", "heatmap": "heatmap.png",
                    "overlay": "overlay.png"},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    _json_dump(out_dir / "prediction_meta.json", meta)
    _json_dump(out_dir / "artifact_manifest.json", _manifest_of(out_dir, exclude={"artifact_manifest.json"}))

    print(f"  image   : {image_path.name} {prepared['source_size'][1]}x{prepared['source_size'][0]}")
    print(f"  predict : {int(mask.sum())} px >= {threshold} ({mask.mean():.6f} of pixels)")
    print(f"  identity: best {frozen.identity['best_sha256'][:12]} epoch "
          f"{frozen.identity['best_epoch']}")
    print(f"  wrote   : {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="stage 12: single-image prediction (no GT needed)")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--run", required=True, help="trained run directory holding best.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    return do_predict(args)


if __name__ == "__main__":
    raise SystemExit(main())
