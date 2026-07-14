#!/usr/bin/env python3
"""
Entry point for evaluating the BLIND model with top-N sampling metrics.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from blind.evaluation_top_n import evaluate_top_n, parse_args

if __name__ == "__main__":
    evaluate_top_n(parse_args())

