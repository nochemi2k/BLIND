"""
Add heteronuclear spectra (19F, 31P, 11B, 29Si) from NMRexp to the prepared 1H/13C splits.

Molecules, splits and tokenizer stay exactly those of prepare_nmrexp.py: every line of the
base {train,val,test}.jsonl.gz is copied, and the heteronuclear spectra of the same molecules
(canonical isomeric SMILES, as in the base splits) are appended as F_NMR / P_NMR / B_NMR /
Si_NMR records. Molecules that only have heteronuclear spectra are not added.
"""
import argparse
import ast
import gzip
import json
import math
import multiprocessing
import os
import shutil
import typing
from collections import Counter

import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm

from .prepare_nmrexp import COUPLING_RE, SPLITS, normalize_multiplicity

NMR_TYPES = {
    "19F NMR": "F_NMR",
    "31P NMR": "P_NMR",
    "P NMR": "P_NMR",
    "{1H}31P NMR": "P_NMR",
    "11B NMR": "B_NMR",
    "B NMR": "B_NMR",
    "29Si NMR": "Si_NMR",
    "Si NMR": "Si_NMR",
}
# Physically plausible shift windows (ppm); spectra with a shift outside are dropped.
SHIFT_RANGES = {"F_NMR": (-400.0, 300.0), "P_NMR": (-300.0, 300.0), "B_NMR": (-150.0, 150.0), "Si_NMR": (-250.0, 150.0)}
# 1J(P,F) and similar reach ~1500 Hz, so the 1000 Hz limit of the 1H/13C pipeline does not apply.
MAX_COUPLING_HZ = 3000.0

# Peak = (signal, intensity, multiplicity, couplings)
Peak = typing.Tuple[float, float, str, typing.Tuple[float, ...]]


def parse_couplings(raw: typing.Any) -> typing.Optional[typing.Tuple[float, ...]]:
    """Return J values in Hz sorted descending; None if any value is implausible."""
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            raw = [raw]
    if raw is None:
        return ()
    values = []
    for item in raw if isinstance(raw, (list, tuple)) else [raw]:
        if isinstance(item, (int, float)):
            value = float(item)
        else:
            match = COUPLING_RE.search(str(item))
            if match is None:
                continue
            value = float(match.group())
        if value >= MAX_COUPLING_HZ:
            return None
        if value > 0:
            values.append(value)
    return tuple(sorted(values, reverse=True))


def parse_peaks(processed: str, spectrum_type: str, max_peaks: int) -> typing.Optional[typing.Tuple[Peak, ...]]:
    """A shift range [hi, lo] becomes its midpoint; intensity is 1 (no integrals for these nuclei)."""
    try:
        raw = ast.literal_eval(processed)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    if not isinstance(raw, list) or not 0 < len(raw) <= max_peaks:
        return None
    low, high = SHIFT_RANGES[spectrum_type]
    peaks = []
    for item in raw:
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            return None
        shift, mult, couplings = item
        try:
            if isinstance(shift, (list, tuple)):
                shift = sum(float(s) for s in shift) / len(shift)
            shift = float(shift)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
        couplings = parse_couplings(couplings)
        if couplings is None or not math.isfinite(shift) or not low <= shift <= high:
            return None
        # A coupled peak without a reported label is not a singlet.
        default = "m" if couplings else "s"
        peaks.append((round(shift, 4), 1.0, normalize_multiplicity(mult, default), couplings))
    return tuple(peaks)


def _canonical(smiles: str) -> typing.Tuple[str, typing.Optional[str]]:
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return smiles, Chem.MolToSmiles(mol) if mol is not None else None


def prepare(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    stats: Counter = Counter()

    table = pq.read_table(
        args.input_path,
        columns=["SMILES", "NMR_type", "NMR_solvent", "NMR_processed"],
        filters=[("NMR_type", "in", list(NMR_TYPES))],
    ).to_pydict()
    rows = [
        (smiles, NMR_TYPES[nmr_type], processed)
        for smiles, nmr_type, solvent, processed in zip(
            table["SMILES"], table["NMR_type"], table["NMR_solvent"], table["NMR_processed"]
        )
        if solvent == args.source_solvent
    ]
    stats["rows_in_solvent"] = len(rows)
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        canonical = dict(pool.map(_canonical, sorted({r[0] for r in rows}), chunksize=1000))

    # smiles -> {(spectrum_type, peaks): None}; the dict keeps order and dedups
    hetero: typing.Dict[str, typing.Dict[typing.Tuple[str, typing.Tuple[Peak, ...]], None]] = {}
    for raw_smiles, spectrum_type, processed in tqdm(rows, desc="Parsing heteronuclear spectra"):
        smiles = canonical[raw_smiles]
        if smiles is None:
            stats["invalid_smiles"] += 1
            continue
        peaks = parse_peaks(processed, spectrum_type, args.max_peaks)
        if peaks is None:
            stats[f"rejected_{spectrum_type}"] += 1
            continue
        entry = hetero.setdefault(smiles, {})
        if (spectrum_type, peaks) in entry:
            stats[f"duplicate_{spectrum_type}"] += 1
        entry[(spectrum_type, peaks)] = None

    matched: typing.Set[str] = set()
    for split in SPLITS:
        src = os.path.join(args.base_dir, f"{split}.jsonl.gz")
        dst = os.path.join(args.output_dir, f"{split}.jsonl.gz")
        seen: typing.Set[str] = set()
        with gzip.open(src, "rt", encoding="utf-8") as fin, gzip.open(dst, "wt", encoding="utf-8") as fout:
            for line in tqdm(fin, desc=f"Copying {split}"):
                fout.write(line)
                seen.add(json.loads(line)["smiles"])
            for smiles in seen & hetero.keys():
                matched.add(smiles)
                for spectrum_type, peaks in hetero[smiles]:
                    stats[f"{split}_{spectrum_type}"] += 1
                    record = {
                        "smiles": smiles,
                        "spectrum_type": spectrum_type,
                        "solvent": args.solvent,
                        "spectrum": [
                            {"signal": s, "intensity": i, "multiplicity": m, "couplings": list(j)}
                            for s, i, m, j in peaks
                        ],
                    }
                    fout.write(json.dumps(record) + "\n")
        stats[f"{split}_molecules_with_hetero"] = len(seen & hetero.keys())
    stats["molecules_only_hetero_skipped"] = len(hetero.keys() - matched)

    for name in ("smiles_tokenizer.model", "smiles_tokenizer.vocab"):
        if os.path.exists(os.path.join(args.base_dir, name)):
            shutil.copy(os.path.join(args.base_dir, name), os.path.join(args.output_dir, name))

    summary = {"filters": dict(sorted(stats.items())), "args": vars(args)}
    with open(os.path.join(args.output_dir, "stats.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add NMRexp heteronuclear spectra to BLIND JSONL splits.")
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/nmrexp/raw/NMRexp_10to24_1_1004.parquet",
        help="NMRexp parquet file.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="data/nmrexp/processed",
        help="Output directory of prepare_nmrexp.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/nmrexp/processed_hetero",
        help="Output directory for splits with heteronuclear spectra.",
    )
    parser.add_argument(
        "--source-solvent", type=str, default="CDCl3", help="NMR_solvent value to keep."
    )
    parser.add_argument(
        "--solvent", type=str, default="CDCl_3", help="Solvent name written to JSONL."
    )
    parser.add_argument("--max-peaks", type=int, default=400, help="Drop spectra with more peaks.")
    parser.add_argument(
        "--num-workers", type=int, default=multiprocessing.cpu_count(), help="RDKit worker processes."
    )
    return parser.parse_args()
