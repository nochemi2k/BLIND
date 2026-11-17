#!/usr/bin/env python3
"""
Entry point for building the elemental analysis noise models of the resampling generator from the
experimental dataset (see src/blind/prepare_elemental_analysis.py).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from blind.prepare_elemental_analysis import prepare, parse_args

if __name__ == "__main__":
    prepare(parse_args())
