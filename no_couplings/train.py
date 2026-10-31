#!/usr/bin/env python3
"""
Entry point for training the BLIND model without J couplings. Run from the repository root.
"""
import disable_couplings  # noqa: F401

from blind.training import train, parse_args

if __name__ == "__main__":
    train(parse_args())
