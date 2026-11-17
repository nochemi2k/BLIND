#!/usr/bin/env python3
"""
Entry point for evaluating a model with elemental analysis (top-N sampling).
Run from the repository root; --no-ea hides the elemental analysis, --no-hetero the heteronuclear
spectra; --empty-hetero gives an empty spectrum for each nucleus absent from the molecule.
--flow-path picks the generator checkpoint (default data/ea/noise_flow.pt), --ea-noise flow|resampling
the noise model.
"""
import enable_ea

from blind.evaluation_top_n import evaluate_top_n, parse_args

if __name__ == "__main__":
    enable_ea.FLOW_PATH = enable_ea.pop_argument("--flow-path") or enable_ea.FLOW_PATH
    enable_ea.EA_NOISE = enable_ea.pop_argument("--ea-noise") or enable_ea.EA_NOISE
    enable_ea.USE_EA = not enable_ea.pop_argument("--no-ea", is_flag=True)
    enable_ea.enable_hetero.USE_HETERO = not enable_ea.pop_argument("--no-hetero", is_flag=True)
    enable_ea.enable_hetero.EMPTY_HETERO = enable_ea.pop_argument("--empty-hetero", is_flag=True)
    evaluate_top_n(parse_args())
