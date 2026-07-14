"""Device selection and training helpers."""
import argparse
import sys
import typing

import torch


def resolve_device(device: typing.Optional[str] = None) -> torch.device:
    """
    Resolve compute device.

    Args:
        device: One of auto, cuda, mps, cpu. Defaults to auto.
    """
    choice = (device or "auto").lower()

    if choice == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            try:
                validate_cuda_device(device)
            except RuntimeError:
                print("Falling back to CPU because the installed PyTorch build cannot run on this GPU.")
                return torch.device("cpu")
            return device
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        device = torch.device("cuda")
        validate_cuda_device(device)
        return device

    if choice == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")

    if choice == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unknown device '{device}'. Use auto, cuda, mps, or cpu.")


def validate_cuda_device(device: torch.device) -> None:
    """Fail early with a helpful message when the PyTorch CUDA build lacks GPU kernels."""
    if device.type != "cuda":
        return

    gpu_name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    torch_version = torch.__version__
    torch_cuda = getattr(torch.version, "cuda", "unknown")

    try:
        torch.zeros(1, device=device)
        _ = torch.randn(4, 4, device=device) @ torch.randn(4, 4, device=device)
    except RuntimeError as exc:
        if capability[0] >= 12:
            raise RuntimeError(
                f"PyTorch cannot execute CUDA kernels on {gpu_name} (sm_{capability[0]}{capability[1]}). "
                f"Installed build: torch {torch_version} (CUDA {torch_cuda}). "
                "Install a CUDA-enabled PyTorch build that supports your GPU from "
                "https://pytorch.org/get-started/locally/ "
                "then run: pip install -r requirements.txt"
            ) from exc
        raise RuntimeError(
            f"PyTorch cannot execute CUDA kernels on {gpu_name}. "
            f"Installed build: torch {torch_version} (CUDA {torch_cuda}). "
            "Install a PyTorch build that matches your GPU from https://pytorch.org/get-started/locally/"
        ) from exc


def default_num_workers() -> int:
    """Use single-process loading on Windows to avoid spawn/pickle issues."""
    return 0 if sys.platform == "win32" else 4


def use_pin_memory(device: torch.device) -> bool:
    return device.type == "cuda"


def supports_cuda_amp(device: torch.device) -> bool:
    return device.type == "cuda"


def autocast_context(device: torch.device, enabled: bool):
    if not enabled or not supports_cuda_amp(device):
        return torch.autocast(device_type="cpu", enabled=False)
    if hasattr(torch, "amp"):
        return torch.amp.autocast("cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def create_grad_scaler(device: torch.device, enabled: bool):
    use_amp = enabled and supports_cuda_amp(device)
    if hasattr(torch, "amp"):
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def add_device_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Compute device: auto (cuda > mps > cpu), cuda, mps, or cpu.",
    )
