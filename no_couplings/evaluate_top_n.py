#!/usr/bin/env python3
"""
Entry point for evaluating a model trained without J couplings (top-N sampling).
Run from the repository root.
"""
import disable_couplings  # noqa: F401

from blind.evaluation_top_n import evaluate_top_n, parse_args

if __name__ == "__main__":
    evaluate_top_n(parse_args())
