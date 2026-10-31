#!/usr/bin/env python3
"""
Entry point for evaluating a model trained without J couplings (greedy decoding).
Run from the repository root.
"""
import disable_couplings  # noqa: F401

from blind.evaluation import evaluate, parse_args

if __name__ == "__main__":
    evaluate(parse_args())
