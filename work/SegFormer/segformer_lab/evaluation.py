"""Reusable FP32 evaluation policy for the frozen PyTorch 2.10 runtime.

Use ``with evaluation_precision(config, device=device):`` around evaluation.
Backend flags are process-wide while the context is active, then restored;
autocast is disabled only inside this context. Training AMP remains separate.

This project consistently uses the legacy ``allow_tf32`` setters/getters in
PyTorch 2.10, including reporting. Do not mix the new ``fp32_precision`` controls
with these APIs. In particular, 2.10's cudnn.flags helper also touches the new
controls, so this module saves/restores the two legacy properties directly.
Disabling TF32 requests IEEE FP32 arithmetic; it does not promise bitwise
batch invariance or turn on deterministic algorithms.
"""
from contextlib import contextmanager
from typing import Any, Iterator

import torch


def _tf32_flags() -> dict[str, bool]:
    return {"cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}


@contextmanager
def evaluation_precision(config: dict[str, Any], *, device: str) -> Iterator[dict[str, Any]]:
    """Apply configured IEEE FP32 evaluation, restoring flags even on failure.

    No model conversion, GPU allocation, environment update or precision-budget
    adjustment is performed. Callers supply FP32 model weights and inputs.
    """
    policy = config["raw"]["eval"].get("precision")
    if policy != "ieee_fp32":
        raise ValueError(f"eval.precision must explicitly be 'ieee_fp32', got {policy!r}")
    device_type = torch.device(device).type
    if device_type not in ("cpu", "cuda"):
        raise ValueError("evaluation precision supports cpu/cuda only")
    before = _tf32_flags()
    record = {"policy": policy, "api": "torch2.10 allow_tf32 (legacy API only)",
              "before": before, "autocast_enabled": False,
              "scope": "evaluation context; restore backend flags and caller autocast on exit",
              "training_amp_config_unchanged": True,
              "bitwise_batch_invariance_guaranteed": False}
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        record["effective"] = _tf32_flags()
        if any(record["effective"].values()):
            raise RuntimeError("IEEE FP32 evaluation policy did not disable TF32")
        with torch.autocast(device_type=device_type, enabled=False):
            yield record
    finally:
        torch.backends.cudnn.allow_tf32 = before["cudnn_allow_tf32"]
        torch.backends.cuda.matmul.allow_tf32 = before["matmul_allow_tf32"]
        record["restored_after_exit"] = _tf32_flags() == before
