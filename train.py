#!/usr/bin/env python3
"""
Entry point for training the BLIND model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from blind.training import train, parse_args

if __name__ == "__main__":
    train(parse_args())

