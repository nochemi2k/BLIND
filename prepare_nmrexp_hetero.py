#!/usr/bin/env python3
"""
Entry point for adding NMRexp heteronuclear spectra (19F, 31P, 11B, 29Si) to the
prepared 1H/13C splits. Run after prepare_nmrexp.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from blind.prepare_nmrexp_hetero import prepare, parse_args

if __name__ == "__main__":
    prepare(parse_args())
