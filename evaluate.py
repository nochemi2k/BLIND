#!/usr/bin/env python3
"""
Entry point for evaluating the BLIND model (greedy decoding).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from blind.evaluation import evaluate, parse_args

if __name__ == "__main__":
    evaluate(parse_args())

