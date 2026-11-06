#!/usr/bin/env python3
"""
Entry point for evaluating a model with heteronuclear spectra (top-N sampling).
Run from the repository root; --no-hetero hides all heteronuclear spectra;
--empty-hetero gives an empty spectrum for each nucleus absent from the molecule.
"""
import enable_hetero

from blind.evaluation_top_n import evaluate_top_n, parse_args

if __name__ == "__main__":
    enable_hetero.USE_HETERO = not enable_hetero.pop_argument("--no-hetero", is_flag=True)
    enable_hetero.EMPTY_HETERO = enable_hetero.pop_argument("--empty-hetero", is_flag=True)
    evaluate_top_n(parse_args())
