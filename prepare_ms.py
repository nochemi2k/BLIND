#!/usr/bin/env python3
"""
Entry point for converting the experimental MS2 spectra into BLIND splits
(see src/blind/prepare_ms.py).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from blind.prepare_ms import prepare, parse_args

if __name__ == "__main__":
    prepare(parse_args())
