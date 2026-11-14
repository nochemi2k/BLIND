#!/usr/bin/env python3
"""
Entry point for fine-tuning a trained model with heteronuclear spectra (hetero/) with elemental analysis.
Run from the repository root; takes the arguments of train.py plus --init-checkpoint.
"""
import enable_ea

from blind.training import train, parse_args

if __name__ == "__main__":
    enable_ea.INIT_CHECKPOINT = enable_ea.pop_argument("--init-checkpoint")
    train(parse_args())
