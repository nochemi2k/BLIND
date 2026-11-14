"""
Simulated elemental analysis (CHN combustion) for molecules without measured data.

The exact composition is computed from the SMILES; realistic "Found" values are produced by
the empirical noise model of prepare_elemental_analysis.py: which elements are reported, an
occasional solvate added to the formula, and a Found - Calcd vector resampled from published
analyses, which keeps the real spread, tails, correlations and the share of values fitted to
the calculated ones.
"""
import collections
import json
import re
import typing

import numpy as np
from rdkit import Chem, RDLogger

ATOMIC_MASS = {
    "H": 1.008, "D": 2.014, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Na": 22.990,
    "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45, "K": 39.098, "Ca": 40.078,
    "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546,
    "Zn": 65.38, "Ga": 69.723, "Ge": 72.63, "As": 74.922, "Se": 78.971, "Br": 79.904, "Rb": 85.468, "Sr": 87.62,
    "Y": 88.906, "Zr": 91.224, "Nb": 92.906, "Mo": 95.95, "Ru": 101.07, "Rh": 102.91, "Pd": 106.42, "Ag": 107.87,
    "Cd": 112.41, "In": 114.82, "Sn": 118.71, "Sb": 121.76, "Te": 127.60, "I": 126.904, "Cs": 132.91, "Ba": 137.33,
    "La": 138.91, "Ce": 140.12, "Pr": 140.91, "Nd": 144.24, "Sm": 150.36, "Eu": 151.96, "Gd": 157.25, "Tb": 158.93,
    "Dy": 162.50, "Ho": 164.93, "Er": 167.26, "Yb": 173.05, "Lu": 174.97, "Hf": 178.49, "Ta": 180.95, "W": 183.84,
    "Re": 186.21, "Os": 190.23, "Ir": 192.22, "Pt": 195.08, "Au": 196.97, "Hg": 200.59, "Tl": 204.38, "Pb": 207.2,
    "Bi": 208.98, "Th": 232.04, "U": 238.03, "Li": 6.94, "Be": 9.012, "Sc": 44.956,
}
# A Found - Calcd vector is resampled among analyses that report exactly the same elements
# when at least this many exist; otherwise the elements are covered by smaller groups.
MIN_POOL_SIZE = 50


def mass_percent(counts: typing.Mapping[str, float]) -> typing.Dict[str, float]:
    total = sum(ATOMIC_MASS[e] * n for e, n in counts.items())
    return {e: 100.0 * ATOMIC_MASS[e] * n / total for e, n in counts.items()}


def formula_counts(smiles: str) -> typing.Optional[typing.Dict[str, float]]:
    """Atom counts with implicit hydrogens; all fragments (salts, counterions) are included."""
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    counts: typing.Counter = collections.Counter()
    for atom in mol.GetAtoms():
        counts[atom.GetSymbol()] += 1
        counts["H"] += atom.GetTotalNumHs()
    if any(e not in ATOMIC_MASS for e in counts):
        return None
    return {e: float(n) for e, n in counts.items() if n}


def formula_counts_plain(formula: str) -> typing.Dict[str, float]:
    """'CH2Cl2' -> {'C': 1, 'H': 2, 'Cl': 2}."""
    counts: typing.Counter = collections.Counter()
    for element, number in re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", formula):
        counts[element] += float(number) if number else 1.0
    return dict(counts)


class ElementalAnalysisNoise:
    def __init__(self, noise_model_path: str) -> None:
        model = json.load(open(noise_model_path))
        self.report_prob: typing.Dict[str, float] = model["report_prob"]
        self.report_absent_prob: typing.Dict[str, float] = model["report_absent_prob"]
        self.solvate_prob: float = model["solvate_prob"]
        self.solvates = [(formula, multiplier) for formula, multiplier, _ in model["solvates"]]
        counts = np.array([n for _, _, n in model["solvates"]], dtype=np.float64)
        self.solvate_weights = counts / counts.sum()
        self.solvate_counts = [formula_counts_plain(formula) for formula, _ in self.solvates]

        pools: typing.Dict[typing.FrozenSet[str], typing.List[typing.Dict[str, float]]] = collections.defaultdict(list)
        for delta in model["deltas"]:
            pools[frozenset(delta)].append(delta)
        self.pools = dict(pools)
        # Large groups ordered by size, used to cover element sets without enough examples
        self.cover_pools = sorted((k for k, v in pools.items() if len(v) >= MIN_POOL_SIZE), key=len, reverse=True)
        self.marginal = {
            e: np.array([d[e] for d in model["deltas"] if e in d]) for e in self.report_prob
        }

    def _delta(self, elements: typing.FrozenSet[str], rng: np.random.Generator) -> typing.Dict[str, float]:
        pool = self.pools.get(elements, [])
        if len(pool) >= MIN_POOL_SIZE:
            return pool[rng.integers(len(pool))]
        delta: typing.Dict[str, float] = {}
        for key in self.cover_pools:
            if key <= elements and not key & delta.keys():
                cover = self.pools[key]
                delta.update(cover[rng.integers(len(cover))])
        for e in elements - delta.keys():
            delta[e] = float(rng.choice(self.marginal[e]))
        return delta

    def sample(
        self,
        counts: typing.Mapping[str, float],
        rng: np.random.Generator,
    ) -> typing.Dict[str, float]:
        """{element: Found, %} for a molecule given its atom counts."""
        counts = collections.Counter(counts)
        if rng.random() < self.solvate_prob:
            k = rng.choice(len(self.solvates), p=self.solvate_weights)
            for e, n in self.solvate_counts[k].items():
                counts[e] += self.solvates[k][1] * n
        exact = mass_percent(counts)
        elements = frozenset(
            e
            for e, p in self.report_prob.items()
            if rng.random() < (p if counts.get(e) else self.report_absent_prob.get(e, 0.0))
        )
        delta = self._delta(elements, rng)
        return {e: round(max(exact.get(e, 0.0) + delta[e], 0.0), 2) for e in sorted(elements)}

