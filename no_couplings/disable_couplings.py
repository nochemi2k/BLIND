"""
Ablation of J couplings without touching the main code.

Importing this module makes every coupling slot of every peak COUPLING_PAD, so the model
only ever sees the learned no_coupling vectors. Architecture, vocabulary and data are unchanged.
"""
import sys
import typing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from blind.training import NMRDataset


def _no_coupling_tensor(cls, peaks: typing.List[typing.Dict[str, typing.Any]]) -> torch.Tensor:
    return torch.full((len(peaks), cls.MAX_COUPLINGS), cls.COUPLING_PAD, dtype=torch.float32)


NMRDataset._coupling_tensor = classmethod(_no_coupling_tensor)
print("J couplings are DISABLED: all coupling slots use no_coupling vectors.")
