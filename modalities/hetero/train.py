#!/usr/bin/env python3
"""
Entry point for fine-tuning a trained BLIND model with heteronuclear spectra.
Run from the repository root; takes the arguments of train.py plus --init-checkpoint.
"""
import enable_hetero

from blind.training import train, parse_args

if __name__ == "__main__":
    enable_hetero.INIT_CHECKPOINT = enable_hetero.pop_argument("--init-checkpoint")
    train(parse_args())
