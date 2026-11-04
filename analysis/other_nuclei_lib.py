"""
Helpers for other_nuclei.ipynb: 19F / 31P / 11B / 29Si spectra of NMRexp and how they
overlap with the 1H/13C train/val/test splits built by prepare_nmrexp.py.
"""
import ast
import gzip
import json
import multiprocessing
import sys
import typing
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from blind.prepare_nmrexp import normalize_multiplicity, parse_couplings, split_of  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# NMRexp NMR_type -> nucleus; unlabelled "B NMR" / "P NMR" / "Si NMR" are the common isotopes.
NUCLEI = {
    "19F NMR": "19F",
    "31P NMR": "31P",
    "P NMR": "31P",
    "{1H}31P NMR": "31P",
    "11B NMR": "11B",
    "B NMR": "11B",
    "29Si NMR": "29Si",
    "Si NMR": "29Si",
}
ELEMENT = {"19F": "F", "31P": "P", "11B": "B", "29Si": "Si"}
# Generous physical shift windows (ppm); anything outside is almost surely a parsing error.
SHIFT_RANGES = {"19F": (-400.0, 300.0), "31P": (-300.0, 300.0), "11B": (-150.0, 150.0), "29Si": (-250.0, 150.0)}
SPLITS = ("train", "val", "test")


def read_raw(parquet_path: str) -> pd.DataFrame:
    """All rows of the heteronuclear NMR types, with the columns needed for the analysis."""
    columns = ["SMILES", "NMR_type", "NMR_solvent", "NMR_frequency", "NMR_processed", "Atom_number", "Atom_number_diff_env"]
    table = pq.read_table(parquet_path, columns=columns, filters=[("NMR_type", "in", list(NUCLEI))])
    df = table.to_pandas()
    df["nucleus"] = df["NMR_type"].map(NUCLEI)
    return df.reset_index(drop=True)


def parse_peaks(processed: str) -> typing.Optional[typing.List[typing.Tuple[float, str, typing.Tuple[float, ...]]]]:
    """[(shift, multiplicity, couplings)] or None. A shift range [hi, lo] becomes its midpoint."""
    try:
        raw = ast.literal_eval(processed)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    peaks = []
    for item in raw:
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            return None
        shift, mult, couplings = item
        try:
            if isinstance(shift, (list, tuple)):
                shift = sum(float(s) for s in shift) / len(shift)
            shift = float(shift)
            couplings = parse_couplings(couplings)
        except (ValueError, TypeError, SyntaxError, ZeroDivisionError):
            return None
        if couplings is None or not np.isfinite(shift):
            return None
        peaks.append((round(shift, 4), normalize_multiplicity(mult, "s" if not couplings else "m"), couplings))
    return peaks


def _keys(smiles: str) -> typing.Tuple[str, typing.Optional[str], typing.Optional[str], dict]:
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return smiles, None, None, {}
    counts = Counter(a.GetSymbol() for a in mol.GetAtoms())
    # Topologically distinct atoms of each element: expected number of signals without accidental overlap.
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    distinct = {
        e: len({ranks[a.GetIdx()] for a in mol.GetAtoms() if a.GetSymbol() == e}) for e in ELEMENT.values()
    }
    info = {f"n_{e}": counts.get(e, 0) for e in ELEMENT.values()}
    info.update({f"n_{e}_distinct": distinct[e] for e in ELEMENT.values()})
    info["heavy_atoms"] = mol.GetNumHeavyAtoms()
    return smiles, Chem.MolToSmiles(mol), Chem.MolToSmiles(mol, isomericSmiles=False), info


def canonical_table(smiles: typing.Iterable[str], workers: int = 0) -> pd.DataFrame:
    """raw SMILES -> canonical isomeric SMILES, split key (no stereo) and element counts."""
    unique = sorted(set(smiles))
    with multiprocessing.Pool(workers or multiprocessing.cpu_count()) as pool:
        rows = pool.map(_keys, unique, chunksize=1000)
    out = pd.DataFrame([{"raw_smiles": r, "smiles": s, "key": k, **info} for r, s, k, info in rows])
    return out.set_index("raw_smiles")


def _flat(smiles: str) -> typing.Tuple[str, typing.Optional[str]]:
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    return smiles, Chem.MolToSmiles(mol, isomericSmiles=False) if mol is not None else None


def split_membership(data_dir: str, workers: int = 0) -> pd.DataFrame:
    """One row per molecule of the processed 1H/13C splits: smiles, key, split, number of 1H and 13C spectra."""
    rows = []
    for split in SPLITS:
        with gzip.open(Path(data_dir) / f"{split}.jsonl.gz", "rt") as f:
            for line in f:
                r = json.loads(line)
                rows.append((r["smiles"], split, r["spectrum_type"]))
    df = pd.DataFrame(rows, columns=["smiles", "split", "spectrum_type"])
    mol = df.pivot_table(index=["smiles", "split"], columns="spectrum_type", aggfunc="size", fill_value=0).reset_index()
    mol.columns.name = None
    with multiprocessing.Pool(workers or multiprocessing.cpu_count()) as pool:
        flat = dict(pool.map(_flat, mol["smiles"].tolist(), chunksize=2000))
    mol["key"] = mol["smiles"].map(flat)
    return mol.rename(columns={"C_NMR": "n_13C", "H_NMR": "n_1H"})


def spectrum_table(raw: pd.DataFrame, canon: pd.DataFrame) -> pd.DataFrame:
    """One row per heteronuclear spectrum with parsed peaks and consistency columns."""
    df = raw.join(canon, on="SMILES")
    df["peaks"] = df["NMR_processed"].map(parse_peaks)
    df["parsed"] = df["peaks"].notna() & df["peaks"].map(lambda p: bool(p) if p is not None else False)
    df["valid_smiles"] = df["key"].notna()
    df["n_signals"] = df["peaks"].map(lambda p: len(p) if p else 0)
    df["shifts"] = df["peaks"].map(lambda p: [x[0] for x in p] if p else [])
    df["multiplicities"] = df["peaks"].map(lambda p: [x[1] for x in p] if p else [])
    df["couplings"] = df["peaks"].map(lambda p: [x[2] for x in p] if p else [])
    lo = df["nucleus"].map(lambda n: SHIFT_RANGES[n][0])
    hi = df["nucleus"].map(lambda n: SHIFT_RANGES[n][1])
    df["in_range"] = [all(l <= s <= h for s in ss) for ss, l, h in zip(df["shifts"], lo, hi)]
    element = df["nucleus"].map(ELEMENT)
    df["n_element"] = [row.get(f"n_{e}", np.nan) for row, e in zip(df[[f"n_{e}" for e in ELEMENT.values()]].to_dict("records"), element)]
    df["n_element_distinct"] = [
        row.get(f"n_{e}_distinct", np.nan)
        for row, e in zip(df[[f"n_{e}_distinct" for e in ELEMENT.values()]].to_dict("records"), element)
    ]
    df["has_element"] = df["n_element"] > 0
    # Peak list as a hashable fingerprint, for counting exact duplicate spectra of one molecule.
    df["fingerprint"] = [
        (n, tuple((round(s, 2), m, c) for s, m, c in p)) if p else None for n, p in zip(df["nucleus"], df["peaks"])
    ]
    df["split_by_hash"] = df["key"].map(lambda k: split_of(k, 0.2, 0.1) if isinstance(k, str) else None)
    return df


def wilson(k: float, n: float, z: float = 1.96) -> typing.Tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def element_counts(smiles: str) -> typing.Dict[str, int]:
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    counts = Counter(a.GetSymbol() for a in mol.GetAtoms()) if mol is not None else Counter()
    return {e: counts.get(e, 0) for e in ELEMENT.values()}


def element_table(smiles: typing.List[str], workers: int = 0) -> pd.DataFrame:
    with multiprocessing.Pool(workers or multiprocessing.cpu_count()) as pool:
        return pd.DataFrame(pool.map(element_counts, smiles, chunksize=2000))


def load_tables(
    raw_path: str, data_dir: str, cache_path: str
) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
    """(spectrum_table, split_membership), computed once and then read from cache_path (a pickle)."""
    cache = Path(cache_path)
    if cache.exists():
        return pd.read_pickle(cache)
    raw = read_raw(raw_path)
    tables = spectrum_table(raw, canonical_table(raw["SMILES"])), split_membership(data_dir)
    pd.to_pickle(tables, cache)
    return tables
