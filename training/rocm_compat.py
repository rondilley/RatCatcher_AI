"""ROCm compatibility shims for training on AMD hardware.

This module exists for one machine class: RDNA3.5 integrated GPUs
(gfx1150/gfx1151/gfx1152, the Radeon 8050S/8060S in the Ryzen AI MAX+
family). It is a no-op everywhere else, including on CUDA and on
discrete AMD cards, because every shim here is applied only after a
probe shows the defect it works around is actually present.

Two defects were measured on this box (Ryzen AI MAX+ 395, gfx1151,
ROCm 7.1.1, torch 2.13.0+rocm7.1) on 2026-08-30.

**The bundled HSA runtime segfaults.** Not handled here -- it cannot
be, because the process dies inside the first kernel launch before any
Python runs. `check_hsa_runtime()` only reports it. The fix is to
replace the copy of `libhsa-runtime64.so` inside the torch wheel with a
symlink to the system one; see `training/README.md`. The system ROCm
7.1.1 runtime dispatches gfx1151 kernels correctly, verified with a
plain HIP program, so the fault is in the wheel's bundled copy rather
than in the driver or the card.

**MIOpen cannot compile its OpenCL batchnorm kernel.**
`MIOpenBatchNormFwdTrainSpatial.cl` carries inline assembly using
`row_bcast:15` and `row_bcast:31` DPP modifiers. Those are GFX9
instructions. RDNA3 dropped them, so the code object build fails and
every `batch_norm` call raises `miopenStatusUnknownError`. YOLOv8 is
Conv-BN-SiLU throughout, so this makes training impossible rather than
merely slow.

Disabling MIOpen wholesale is the obvious response and it is the wrong
one. Measured here on a 16x64x160x160 tensor:

    MIOpen on    conv3x3   2.40 ms    batchnorm  fails
    MIOpen off   conv3x3  15.26 ms    batchnorm   1.70 ms

Turning MIOpen off costs 6.4x on the convolutions, which are nearly all
of the work in a detector. Falling back to the system MIOpen instead
does not help either: Ubuntu ships MIOpen as `+dfsg`, with the Winograd
assembly kernels stripped, so its convolutions drop to 15.01 ms -- the
same slow path, reached by a different route.

So the shim is narrow: MIOpen stays on for everything, and only
`batch_norm` is routed around it. PyTorch reads
`torch.backends.cudnn.enabled` at each call site, which is what makes
this possible per-operation instead of per-process. Native batchnorm at
1.70 ms is not a compromise -- it is faster than the MIOpen path was on
the hardware where that path works.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

# RDNA3.5 integrated parts. Kept as a prefix tuple rather than a single
# string so a sibling part in the same family is covered without an edit.
_AFFECTED_ARCH_PREFIXES = ("gfx1150", "gfx1151", "gfx1152", "gfx1153")

_batch_norm_patched = False


def gpu_arch() -> str | None:
    """Return the GPU's LLVM target name, or None if there is no GPU.

    Answers "gfx1151" on this box. Returns None rather than raising when
    torch has no GPU, so a caller can use it as a plain predicate.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except (RuntimeError, AssertionError, IndexError):
        # A GPU that answers is_available() but not a property query is a
        # broken install, which is the caller's problem to report, not
        # this function's to raise inside.
        return None


def is_affected_igpu() -> bool:
    """True on the RDNA3.5 integrated GPUs this module has shims for."""
    arch = gpu_arch()
    return bool(arch and arch.startswith(_AFFECTED_ARCH_PREFIXES))


def check_hsa_runtime() -> str | None:
    """Report a torch wheel carrying its own HSA runtime on an affected GPU.

    Returns a message to show the user, or None when nothing is wrong.
    Cannot fix anything: the bundled runtime segfaults inside the first
    kernel launch, which kills the interpreter outright. Detection is by
    file type -- the repair described in `training/README.md` leaves a
    symlink where the wheel shipped a regular file.
    """
    if not is_affected_igpu():
        return None
    try:
        import torch
    except ImportError:
        return None

    lib = os.path.join(os.path.dirname(torch.__file__), "lib", "libhsa-runtime64.so")
    if not os.path.exists(lib) or os.path.islink(lib):
        return None
    return (
        f"torch is using its own bundled HSA runtime at {lib}. On "
        f"{gpu_arch()} that copy segfaults on the first kernel launch. "
        "Replace it with a symlink to the system runtime; see "
        "training/README.md, section 'AMD ROCm setup'."
    )


def _miopen_batchnorm_works() -> bool:
    """Probe whether MIOpen can actually run a batchnorm on this GPU.

    Runs the smallest possible real batchnorm with MIOpen enabled and
    reports whether it survived. This is the reason the shim below is
    conditional: when MIOpen fixes the RDNA3 DPP assembly, this probe
    starts passing and no patch is applied.
    """
    import torch
    import torch.nn as nn

    previous = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = True
    try:
        layer = nn.BatchNorm2d(4).cuda()
        layer(torch.randn(2, 4, 8, 8, device="cuda"))
        torch.cuda.synchronize()
        return True
    except RuntimeError:
        # miopenStatusUnknownError arrives as a plain RuntimeError with
        # no distinguishing type, so the exception class is all there is
        # to match on. Any RuntimeError here means the op is unusable.
        return False
    finally:
        torch.backends.cudnn.enabled = previous


def patch_batch_norm() -> bool:
    """Route batch_norm around MIOpen, keeping MIOpen for everything else.

    Returns True if the patch was applied. Idempotent, and a no-op unless
    the GPU is an affected part AND the probe shows batchnorm is broken.
    """
    global _batch_norm_patched
    if _batch_norm_patched or not is_affected_igpu():
        return False

    import torch
    import torch.nn.functional as F

    if _miopen_batchnorm_works():
        log.info("MIOpen batchnorm works on %s, no patch needed", gpu_arch())
        return False

    original = F.batch_norm

    def batch_norm(*args, **kwargs):
        previous = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        try:
            return original(*args, **kwargs)
        finally:
            torch.backends.cudnn.enabled = previous

    F.batch_norm = batch_norm
    _batch_norm_patched = True
    log.info(
        "Patched batch_norm to bypass MIOpen on %s (MIOpen kept for conv)",
        gpu_arch(),
    )
    return True


def apply() -> list[str]:
    """Apply every shim this GPU needs. Returns notes worth printing."""
    notes: list[str] = []

    warning = check_hsa_runtime()
    if warning:
        notes.append(f"[WARNING] {warning}")

    if patch_batch_norm():
        notes.append(
            f"[INFO] {gpu_arch()}: batch_norm routed around MIOpen "
            "(MIOpen kept for convolution)"
        )
    return notes
