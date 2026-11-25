#!/usr/bin/env python3
"""
Entry point for fine-tuning a trained model with elemental analysis (modalities/ea/) with MS2 spectra.
Run from the repository root; takes the arguments of train.py plus --init-checkpoint (a model with
elemental analysis) or --resume-checkpoint (a model with MS2 spectra, trained further with all its
weights), --ms-dir (default data/ms/processed), --flow-path and --ea-noise as in modalities/ea,
--model-size base (default) or large (modalities/ea/enable_ea.py MODEL_SIZES).
"""
import enable_ms

from blind.training import train, parse_args

if __name__ == "__main__":
    enable_ms.enable_ea.MODEL_SIZE = enable_ms.pop_argument("--model-size") or enable_ms.enable_ea.MODEL_SIZE
    enable_ms.MS_DIR = enable_ms.pop_argument("--ms-dir") or enable_ms.MS_DIR
    enable_ms.enable_ea.FLOW_PATH = enable_ms.pop_argument("--flow-path") or enable_ms.enable_ea.FLOW_PATH
    enable_ms.enable_ea.EA_NOISE = enable_ms.pop_argument("--ea-noise") or enable_ms.enable_ea.EA_NOISE
    enable_ms.INIT_CHECKPOINT = enable_ms.pop_argument("--init-checkpoint")
    enable_ms.RESUME_CHECKPOINT = enable_ms.pop_argument("--resume-checkpoint")
    train(parse_args())
