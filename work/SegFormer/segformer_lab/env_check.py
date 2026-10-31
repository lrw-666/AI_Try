"""Record the real runtime facts of the SegFormer project environment.

The point of this module is evidence, not decoration: every field written here
is read from the live process, so the report can be compared later and used to
explain why a run on another machine differs.

What is recorded
----------------
* interpreter, prefix, user-site state and whether third-party packages really
  live inside the project prefix (catches silent leakage from another Python);
* OS, CPU count, RAM, disks and GPU properties;
* versions of torch / torchvision / transformers / numpy / Pillow / matplotlib;
* the cache and temp paths the libraries actually resolve to;
* optional probes: a CUDA matrix forward/backward, and a SegFormer forward
  pass with random weights (no download, no pretrained claim).

What is deliberately NOT recorded
---------------------------------
The full environment block, tokens, proxy credentials or anything account
specific. Only a small whitelist of cache-related variables is kept.

Examples
--------
    python -m segformer_lab.env_check --out reports/environment.json
    python -m segformer_lab.env_check --out reports/environment.json --probe full
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
import site
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__, paths

SCHEMA = "segformer-lab/env-check/1"

# Modules whose file location proves which environment is actually in use.
PROBE_MODULES = ("torch", "torchvision", "transformers", "numpy", "PIL", "matplotlib", "yaml", "huggingface_hub", "tensorboard")

# Only these variables are reported; the rest of the environment stays private.
REPORTED_ENV_VARS = (
    "SEGFORMER_RESOURCE",
    "PYTHONNOUSERSITE",
    "PYTHONPYCACHEPREFIX",
    "PYTHONUTF8",
    "PIP_CONFIG_FILE",
    "PIP_CACHE_DIR",
    "CONDA_PKGS_DIRS",
    "CONDA_ENVS_PATH",
    "CONDA_REGISTER_ENVS",
    "CONDA_NUMBER_CHANNEL_NOTICES",
    "CONDARC",
    "HF_HOME",
    "HF_HUB_CACHE",
    "TORCH_HOME",
    "TORCH_EXTENSIONS_DIR",
    "CUDA_CACHE_PATH",
    "XDG_CACHE_HOME",
    "MPLCONFIGDIR",
    "TEMP",
    "TMP",
    "TMPDIR",
)


class _MemoryStatusEx(ctypes.Structure):
    """Win32 MEMORYSTATUSEX, used so no psutil dependency is required."""

    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _windows_memory() -> tuple[float | None, float | None]:
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None, None
    return round(status.ullTotalPhys / 1024 ** 3, 2), round(status.ullAvailPhys / 1024 ** 3, 2)


def _ram_gib() -> float | None:
    """Total physical memory in GiB."""
    if sys.platform == "win32":
        return _windows_memory()[0]
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3, 2)
    except (ValueError, OSError, AttributeError):
        return None


def _avail_ram_gib() -> float | None:
    """Currently available physical memory in GiB."""
    if sys.platform == "win32":
        return _windows_memory()[1]
    try:
        return round(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024 ** 3, 2)
    except (ValueError, OSError, AttributeError):
        return None


def _disk_report() -> dict[str, dict[str, float]]:
    """Free space for the two roots plus the drive that hosts them."""
    report: dict[str, dict[str, float]] = {}
    for label, root in (("project_root", paths.PROJECT_ROOT), ("resource_root", paths.RESOURCE_ROOT)):
        try:
            usage = shutil.disk_usage(root)
        except OSError:
            continue
        report[label] = {
            "path": str(root),
            "total_gib": round(usage.total / 1024 ** 3, 2),
            "free_gib": round(usage.free / 1024 ** 3, 2),
        }
    return report


def _package_facts() -> dict[str, Any]:
    """Versions and file locations; a module missing here is reported, not raised."""
    import importlib

    facts: dict[str, Any] = {}
    for name in PROBE_MODULES:
        entry: dict[str, Any] = {}
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - report, do not crash the checker
            entry["import_error"] = f"{type(exc).__name__}: {exc}"
            facts[name] = entry
            continue
        entry["version"] = getattr(module, "__version__", None)
        entry["file"] = getattr(module, "__file__", None)
        facts[name] = entry

    try:
        import torch

        entry = facts["torch"]
        entry["cuda_build"] = torch.version.cuda
        entry["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        entry["is_available"] = bool(torch.cuda.is_available())
        entry["hip_build"] = getattr(torch.version, "hip", None)
        entry["threads"] = torch.get_num_threads()
    except Exception as exc:  # noqa: BLE001
        facts.setdefault("torch", {})["probe_error"] = f"{type(exc).__name__}: {exc}"
    return facts


def _import_isolation(packages: dict[str, Any]) -> dict[str, Any]:
    """Flag any probed package that resolves outside the project prefix."""
    prefix = Path(sys.prefix).resolve()
    outside: list[dict[str, str]] = []
    checked = 0
    for name, entry in packages.items():
        file = entry.get("file")
        if not file:
            continue
        checked += 1
        try:
            Path(file).resolve().relative_to(prefix)
        except ValueError:
            outside.append({"module": name, "file": file})
    return {
        "sys_prefix": str(prefix),
        "packages_checked": checked,
        "outside_prefix": outside,
        "ok": not outside,
    }


def _site_paths_outside_prefix() -> list[str]:
    """sys.path entries under another Python installation (leak detector)."""
    prefix = Path(sys.prefix).resolve()
    suspects: list[str] = []
    for raw in sys.path:
        if not raw:
            continue
        candidate = Path(raw)
        if not candidate.is_absolute() or not candidate.exists():
            continue
        text = str(candidate).lower()
        if "site-packages" not in text and "dist-packages" not in text:
            continue
        try:
            candidate.resolve().relative_to(prefix)
        except ValueError:
            suspects.append(str(candidate))
    return suspects


def _effective_paths() -> dict[str, Any]:
    """Where the libraries actually write, not what the variables claim.

    Reading ``os.environ`` is not proof: a library can ignore a variable, and
    Python's tempfile prefers TMPDIR over TEMP. These values come from the
    library APIs themselves.
    """
    import tempfile

    effective: dict[str, Any] = {"temp_dir": tempfile.gettempdir()}
    try:
        import torch

        effective["torch_hub_dir"] = torch.hub.get_dir()
    except Exception as exc:  # noqa: BLE001
        effective["torch_hub_dir_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import matplotlib

        effective["matplotlib_cache_dir"] = matplotlib.get_cachedir()
    except Exception as exc:  # noqa: BLE001
        effective["matplotlib_cache_dir_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from huggingface_hub import constants as hf_constants

        effective["hf_home"] = hf_constants.HF_HOME
        effective["hf_hub_cache"] = hf_constants.HF_HUB_CACHE
    except Exception as exc:  # noqa: BLE001
        effective["hf_cache_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from transformers.utils import hub as tf_hub

        effective["transformers_cache_constant"] = getattr(tf_hub, "TRANSFORMERS_CACHE", None)
    except Exception as exc:  # noqa: BLE001
        effective["transformers_cache_error"] = f"{type(exc).__name__}: {exc}"

    resource = paths.RESOURCE_ROOT.resolve()
    outside = []
    for key, value in effective.items():
        if not isinstance(value, str) or key.endswith("_error"):
            continue
        candidate = Path(value)
        try:
            candidate.resolve().relative_to(resource)
        except (ValueError, OSError):
            outside.append({"key": key, "path": value})
    effective["outside_resource_root"] = outside
    return effective


def _cuda_device_facts() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {"available": False, "device_count": 0}
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": props.name,
                "capability": f"{props.major}.{props.minor}",
                "total_memory_mib": round(props.total_memory / 1024 ** 2),
                "multi_processor_count": props.multi_processor_count,
            }
        )
    return {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "devices": devices,
    }


def probe_cuda() -> dict[str, Any]:
    """A tiny matmul forward/backward; proves kernels run, nothing more."""
    import torch

    if not torch.cuda.is_available():
        return {"ok": False, "error": "torch.cuda.is_available() is False"}

    started = time.perf_counter()
    torch.manual_seed(42)  # comparable numbers across repeated checks
    x = torch.randn(256, 256, device="cuda", requires_grad=True)
    loss = (x @ x.T).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    grad_ok = bool(torch.isfinite(x.grad).all())
    return {
        "ok": bool(torch.isfinite(loss).item()) and grad_ok,
        "loss": float(loss.item()),
        "grad_finite": grad_ok,
        "device_name": torch.cuda.get_device_name(0),
        "elapsed_ms": round(elapsed_ms, 2),
    }


def probe_segformer(input_size: int = 512) -> dict[str, Any]:
    """Forward/backward through HF SegFormer with RANDOM weights.

    No pretrained weights are downloaded and no accuracy is implied. The probe
    answers one question: does the transformers SegFormer implementation run on
    this GPU with a 2-class head, the expected output stride and finite
    gradients in both the encoder and the decode head?
    """
    import torch
    import torch.nn.functional as F
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    if not torch.cuda.is_available():
        return {"ok": False, "error": "torch.cuda.is_available() is False"}

    # Fix the weight initialisation so repeated checks report comparable
    # gradient norms; otherwise every run prints different numbers and the
    # report stops being usable as evidence.
    torch.manual_seed(42)

    config = SegformerConfig(
        num_labels=2,
        id2label={0: "background", 1: "defect"},
        label2id={"background": 0, "defect": 1},
        semantic_loss_ignore_index=255,
    )
    model = SegformerForSemanticSegmentation(config).cuda()
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    encoder_param = next(model.segformer.parameters())
    head_param = next(model.decode_head.parameters())
    encoder_before = encoder_param.detach().clone()
    head_before = head_param.detach().clone()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    generator = torch.Generator(device="cpu").manual_seed(42)
    pixel_values = torch.randn(2, 3, input_size, input_size, generator=generator).cuda()
    labels = torch.zeros(2, input_size, input_size, dtype=torch.long, device="cuda")
    labels[:, input_size // 4: input_size // 2, input_size // 4: input_size // 2] = 1
    labels[:, 0:8, :] = 255  # stand-in for padding/ignore region

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    logits = model(pixel_values=pixel_values).logits
    upsampled = F.interpolate(logits, size=(input_size, input_size), mode="bilinear", align_corners=False)
    loss = F.cross_entropy(upsampled, labels, ignore_index=255)
    loss.backward()

    encoder_grad = encoder_param.grad
    head_grad = head_param.grad
    encoder_grad_norm = float(encoder_grad.norm().item()) if encoder_grad is not None else 0.0
    head_grad_norm = float(head_grad.norm().item()) if head_grad is not None else 0.0

    optimizer.step()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    encoder_changed = not torch.equal(encoder_before, encoder_param.detach())
    head_changed = not torch.equal(head_before, head_param.detach())

    summary = {
        "ok": (
            torch.isfinite(loss).item()
            and encoder_grad_norm > 0.0
            and head_grad_norm > 0.0
            and encoder_changed
            and head_changed
        ),
        "note": "randomly initialised SegformerConfig; weights are NOT pretrained",
        "input_shape": list(pixel_values.shape),
        "logits_shape": list(logits.shape),
        "upsampled_shape": list(upsampled.shape),
        "output_stride": input_size // int(logits.shape[-1]),
        "loss_ce": float(loss.item()),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "encoder_grad_norm": round(encoder_grad_norm, 6),
        "head_grad_norm": round(head_grad_norm, 6),
        "encoder_updated": encoder_changed,
        "head_updated": head_changed,
        "elapsed_ms": round(elapsed_ms, 1),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 1024 ** 2, 1),
    }
    del model, optimizer, pixel_values, labels, logits, upsampled
    torch.cuda.empty_cache()
    return summary


def collect(probe: str = "cuda", input_size: int = 512) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "package_version": __version__,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "project_root": str(paths.PROJECT_ROOT),
        "resource_root": str(paths.RESOURCE_ROOT),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "version_info": list(sys.version_info[:3]),
            "prefix": sys.prefix,
            "base_prefix": getattr(sys, "base_prefix", sys.prefix),
            # A Conda prefix reports base_prefix == prefix even though it is a
            # separate environment, so this flag carries no isolation meaning.
            # Real isolation is proven by import_isolation below.
            "note": "Conda prefix: base_prefix equals prefix by design; check package locations instead",
            "site_packages_writable": os.access(
                Path(sys.prefix) / ("Lib/site-packages" if sys.platform == "win32" else "lib"), os.W_OK
            ),
        },
        "site": {
            "ENABLE_USER_SITE": site.ENABLE_USER_SITE,
            "suspect_sys_path": _site_paths_outside_prefix(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_compiler": platform.python_compiler(),
        },
        "hardware": {
            "cpu_logical": os.cpu_count(),
            "ram_total_gib": _ram_gib(),
            "ram_available_gib": _avail_ram_gib(),
        },
        "disk": _disk_report(),
        "caches": {name: os.environ.get(name) for name in REPORTED_ENV_VARS},
        "effective_paths": _effective_paths(),
        "packages": _package_facts(),
    }

    try:
        report["gpu"] = _cuda_device_facts()
    except Exception as exc:
        report["gpu"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    report["import_isolation"] = _import_isolation(report["packages"])

    requested_probes = {"none": (), "cuda": ("cuda_probe",), "full": ("cuda_probe", "segformer_probe")}
    for key in requested_probes[probe]:
        try:
            report[key] = probe_cuda() if key == "cuda_probe" else probe_segformer(input_size=input_size)
        except Exception as exc:
            report[key] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    errors: list[str] = []
    expected_prefix = paths.ENV_DIR.resolve()
    if Path(sys.prefix).resolve() != expected_prefix:
        errors.append(f"Python prefix differs from project environment: expected {expected_prefix}")
    if Path(sys.executable).resolve() != (expected_prefix / "python.exe").resolve():
        errors.append("Python executable differs from the project interpreter")
    if report["site"]["ENABLE_USER_SITE"] is not False:
        errors.append("Python user site is not disabled")
    for name in PROBE_MODULES:
        entry = report["packages"].get(name, {})
        if entry.get("import_error") or not entry.get("file"):
            errors.append(f"required package {name} did not import successfully: {entry.get('import_error', 'missing module location')}")
    for key, value in report["effective_paths"].items():
        if key.endswith("_error"):
            errors.append(f"required cache path inventory failed ({key}): {value}")
    if not report["import_isolation"]["ok"]:
        errors.append("some packages resolve outside the project prefix")
    if report["site"]["suspect_sys_path"]:
        errors.append("sys.path contains site-packages from another Python")
    if report["effective_paths"]["outside_resource_root"]:
        keys = ", ".join(item["key"] for item in report["effective_paths"]["outside_resource_root"])
        errors.append(f"cache/temp paths outside the resource root: {keys}")
    if not report["gpu"].get("available"):
        errors.append("CUDA is not available")
    for key in ("cuda_probe", "segformer_probe"):
        block = report.get(key)
        if isinstance(block, dict) and block.get("ok") is False:
            errors.append(f"{key} failed")
    report["errors"] = errors
    report["status"] = "ok" if not errors else "failed"
    return report


def _print_summary(report: dict[str, Any], out_path: Path) -> None:
    packages = report["packages"]
    print("SegFormer environment check")
    print(f"  python      : {report['python']['executable']}")
    print(f"  version     : {report['python']['version'].splitlines()[0]}")
    print(f"  prefix      : {report['python']['prefix']}")
    print(f"  user site   : {report['site']['ENABLE_USER_SITE']}")
    print(f"  resource    : {report['resource_root']}")
    for name in ("torch", "torchvision", "transformers", "numpy", "PIL", "matplotlib", "yaml"):
        entry = packages.get(name, {})
        state = entry.get("version") or entry.get("import_error") or "missing"
        print(f"  {name:<13}: {state}")
    torch_entry = packages.get("torch", {})
    print(f"  cuda build  : {torch_entry.get('cuda_build')}")
    gpu = report["gpu"]
    print(f"  cuda avail  : {gpu.get('available')}")
    for device in gpu.get("devices", []):
        print(f"  gpu         : {device['name']} ({device['total_memory_mib']} MiB, cc {device['capability']})")
    cuda_probe = report.get("cuda_probe")
    if cuda_probe:
        print(f"  cuda probe  : ok={cuda_probe.get('ok')} loss={cuda_probe.get('loss')} grad_finite={cuda_probe.get('grad_finite')}")
    seg_probe = report.get("segformer_probe")
    if seg_probe:
        print(
            f"  segformer   : ok={seg_probe.get('ok')} logits={seg_probe.get('logits_shape')} "
            f"params={seg_probe.get('trainable_params')}/{seg_probe.get('total_params')}"
        )
        print(
            f"                encoder_grad={seg_probe.get('encoder_grad_norm')} "
            f"head_grad={seg_probe.get('head_grad_norm')} peak={seg_probe.get('peak_allocated_mib')} MiB"
        )
    print(f"  isolation   : outside_prefix={len(report['import_isolation']['outside_prefix'])} "
          f"suspect_sys_path={len(report['site']['suspect_sys_path'])}")
    effective = report["effective_paths"]
    print(f"  temp dir    : {effective.get('temp_dir')}")
    print(f"  hf hub      : {effective.get('hf_hub_cache')}")
    print(f"  torch hub   : {effective.get('torch_hub_dir')}")
    outside = effective.get("outside_resource_root", [])
    print(f"  cache leak  : {len(outside)}" + (f" -> {[item['key'] for item in outside]}" if outside else ""))
    print(f"  status      : {report['status']}")
    for error in report["errors"]:
        print(f"  error       : {error}")
    print(f"  report      : {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="reports/environment.json", help="JSON output path, relative to the project root")
    parser.add_argument(
        "--probe",
        choices=("none", "cuda", "full"),
        default="cuda",
        help="none: inventory only; cuda: + matmul forward/backward; full: + random-weight SegFormer forward/backward",
    )
    parser.add_argument("--input-size", type=int, default=512, help="SegFormer probe input size (square)")
    args = parser.parse_args(argv)

    out_path = paths.ensure_parent(args.out)
    report = collect(probe=args.probe, input_size=args.input_size)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_summary(report, out_path)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
