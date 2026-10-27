#!/usr/bin/env python3
"""
Entry point for converting the NMRexp database into BLIND training data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from blind.prepare_nmrexp import prepare, parse_args

if __name__ == "__main__":
    prepare(parse_args())
