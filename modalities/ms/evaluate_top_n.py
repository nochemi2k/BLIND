#!/usr/bin/env python3
"""
Entry point for evaluating a model with MS2 spectra on NMR test data (top-N sampling).
Run from the repository root; --no-ms hides the MS2 spectra, --no-ea the elemental analysis, --no-hetero
the heteronuclear spectra; --empty-hetero gives an empty spectrum for each nucleus absent from the molecule.
--ms-dir, --flow-path, --ea-noise and --model-size (base or large) as in train.py. Molecules with MS2 spectra only: evaluate_ms.py.
"""
import enable_ms

from blind.evaluation_top_n import evaluate_top_n, parse_args

if __name__ == "__main__":
    enable_ms.enable_ea.MODEL_SIZE = enable_ms.pop_argument("--model-size") or enable_ms.enable_ea.MODEL_SIZE
    enable_ms.MS_DIR = enable_ms.pop_argument("--ms-dir") or enable_ms.MS_DIR
    enable_ms.enable_ea.FLOW_PATH = enable_ms.pop_argument("--flow-path") or enable_ms.enable_ea.FLOW_PATH
    enable_ms.enable_ea.EA_NOISE = enable_ms.pop_argument("--ea-noise") or enable_ms.enable_ea.EA_NOISE
    enable_ms.USE_MS = not enable_ms.pop_argument("--no-ms", is_flag=True)
    enable_ms.enable_ea.USE_EA = not enable_ms.pop_argument("--no-ea", is_flag=True)
    enable_ms.enable_hetero.USE_HETERO = not enable_ms.pop_argument("--no-hetero", is_flag=True)
    enable_ms.enable_hetero.EMPTY_HETERO = enable_ms.pop_argument("--empty-hetero", is_flag=True)
    evaluate_top_n(parse_args())
