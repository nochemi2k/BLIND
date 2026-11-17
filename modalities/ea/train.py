#!/usr/bin/env python3
"""
Entry point for fine-tuning a trained model with heteronuclear spectra (modalities/hetero/) with elemental analysis.
Run from the repository root; takes the arguments of train.py plus --init-checkpoint.
--flow-path picks the generator checkpoint (default data/ea/noise_flow.pt), --ea-noise flow|resampling
the noise model.
"""
import enable_ea

from blind.training import train, parse_args

if __name__ == "__main__":
    enable_ea.FLOW_PATH = enable_ea.pop_argument("--flow-path") or enable_ea.FLOW_PATH
    enable_ea.EA_NOISE = enable_ea.pop_argument("--ea-noise") or enable_ea.EA_NOISE
    enable_ea.INIT_CHECKPOINT = enable_ea.pop_argument("--init-checkpoint")
    train(parse_args())
