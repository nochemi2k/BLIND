"""
Convert the collected experimental MS2 spectra (data/ms/cleaned_all.jsonl.gz) into model splits.

Every MS2 spectrum is kept whatever the ionization method; in-silico spectra are dropped. Peaks
above the precursor m/z (+ PRECURSOR_MARGIN) are removed, the MAX_PEAKS most intense are kept and
intensities are scaled to a maximum of 1 (src/blind/ms_spectra.py). Identical spectra of a
molecule are merged. The SMILES are used as they are (already canonical); a molecule is dropped if
the NMR tokenizer cannot encode it within the SMILES length limit, or if it is in the external
validation set. Splits use the stereo-agnostic key and hash of prepare_nmrexp.py, so a molecule
with NMR spectra lands in the same split as in data/nmrexp/processed.

Output <output-dir>/<split>.npz (arrays below) and stats.json:
    smiles             [M] molecules of the split
    spectrum_smiles    [S] molecule index of each spectrum
    offsets            [S + 1] peak range of spectrum s: offsets[s]:offsets[s + 1]
    mz, intensity      [P] peaks
    precursor_mz       [S] NaN when not reported
    collision_energy   [S] NaN when not reported
    adduct, polarity, ionization, instrument, charge   [S] indices into the vocabularies of ms_spectra.py
"""
import argparse
import collections
import gzip
import hashlib
import json
import multiprocessing
import os
import typing
from array import array

import numpy as np
import sentencepiece as spm
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm

from .ms_spectra import (
    META_FIELDS,
    clean_peaks,
    is_simulated,
    normalize_adduct,
    normalize_charge,
    normalize_instrument,
    normalize_ionization,
    normalize_polarity,
)
from .prepare_nmrexp import load_excluded_keys, split_of

_tokenizer: typing.Optional[spm.SentencePieceProcessor] = None


def _init_worker(tokenizer_path: str) -> None:
    global _tokenizer
    RDLogger.DisableLog("rdApp.*")
    _tokenizer = spm.SentencePieceProcessor()
    _tokenizer.load(tokenizer_path)


def _molecule(args: typing.Tuple[str, int]) -> typing.Tuple[str, typing.Optional[str], str]:
    """(SMILES, stereo-agnostic key or None, reason when rejected)."""
    smiles, max_smiles_len = args
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles, None, "invalid_smiles"
    ids = _tokenizer.encode(smiles)
    if _tokenizer.unk_id() in ids:
        return smiles, None, "smiles_unk_token"
    if len(ids) + 2 > max_smiles_len:
        return smiles, None, "smiles_too_long"
    return smiles, Chem.MolToSmiles(mol, isomericSmiles=False), ""


def _records(path: str) -> typing.Iterator[dict]:
    with gzip.open(path, "rt") as f:
        for line in f:
            yield json.loads(line)


def prepare(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    stats: collections.Counter = collections.Counter()

    molecules = set()
    for obj in tqdm(_records(args.input_path), desc="Molecules"):
        if obj["record"].get("ms_level") == 2:
            molecules.add(obj["smiles"])
    excluded = load_excluded_keys(args.exclude_csv)
    split_by_smiles: typing.Dict[str, str] = {}
    with multiprocessing.Pool(args.num_workers, initializer=_init_worker, initargs=(args.tokenizer_path,)) as pool:
        for smiles, key, reason in tqdm(
            pool.imap_unordered(_molecule, ((s, args.max_smiles_len) for s in sorted(molecules)), chunksize=256),
            total=len(molecules), desc="SMILES",
        ):
            if key is None:
                stats[reason] += 1
            elif key in excluded:
                stats["excluded_molecules"] += 1
            else:
                split_by_smiles[smiles] = split_of(key, args.val_frac, args.test_frac)

    def empty_split() -> dict:
        return {
            "smiles": {}, "spectrum_smiles": array("i"), "offsets": array("q", [0]), "mz": array("f"),
            "intensity": array("f"), "precursor_mz": array("f"), "collision_energy": array("f"),
            **{field: array("b") for field in META_FIELDS},
        }

    splits = {name: empty_split() for name in ("train", "val", "test")}
    seen = set()
    for obj in tqdm(_records(args.input_path), desc="Spectra"):
        record = obj["record"]
        if record.get("ms_level") != 2:
            stats["not_ms2"] += 1
            continue
        if is_simulated(record):
            stats["in_silico"] += 1
            continue
        split = split_by_smiles.get(obj["smiles"])
        if split is None:
            stats["molecule_rejected"] += 1
            continue
        precursor = record.get("precursor_mz")
        try:
            precursor = float(precursor) if precursor is not None else None
        except (TypeError, ValueError):
            precursor = None
        if precursor is not None and not (0 < precursor < 1e5):
            precursor = None
        peaks = clean_peaks(record.get("spectrum") or [], precursor)
        if peaks is None:
            stats["no_peaks"] += 1
            continue
        digest = hashlib.md5(obj["smiles"].encode() + np.round(peaks, 3).tobytes()).digest()
        if digest in seen:
            stats["duplicate_spectrum"] += 1
            continue
        seen.add(digest)
        energy = record.get("collision_energy")
        try:
            energy = float(energy) if energy is not None else float("nan")
        except (TypeError, ValueError):
            energy = float("nan")

        out = splits[split]
        out["spectrum_smiles"].append(out["smiles"].setdefault(obj["smiles"], len(out["smiles"])))
        out["mz"].extend(peaks[:, 0].tolist())
        out["intensity"].extend(peaks[:, 1].tolist())
        out["offsets"].append(len(out["mz"]))
        out["precursor_mz"].append(precursor if precursor is not None else float("nan"))
        out["collision_energy"].append(energy)
        out["adduct"].append(normalize_adduct(record.get("adduct")))
        out["polarity"].append(normalize_polarity(record))
        out["ionization"].append(normalize_ionization(record))
        out["instrument"].append(normalize_instrument(record.get("instrument")))
        out["charge"].append(normalize_charge(record.get("charge")))
        stats["kept_spectra"] += 1

    for name, out in splits.items():
        arrays = {
            "smiles": np.array(list(out["smiles"]), dtype=object),
            "spectrum_smiles": np.frombuffer(out["spectrum_smiles"], dtype=np.int32),
            "offsets": np.frombuffer(out["offsets"], dtype=np.int64),
            "mz": np.frombuffer(out["mz"], dtype=np.float32),
            "intensity": np.frombuffer(out["intensity"], dtype=np.float32),
            "precursor_mz": np.frombuffer(out["precursor_mz"], dtype=np.float32),
            "collision_energy": np.frombuffer(out["collision_energy"], dtype=np.float32),
            **{field: np.frombuffer(out[field], dtype=np.int8) for field in META_FIELDS},
        }
        np.savez(os.path.join(args.output_dir, f"{name}.npz"), **arrays)
        stats[f"{name}_molecules"] = len(out["smiles"])
        stats[f"{name}_spectra"] = len(out["spectrum_smiles"])
        stats[f"{name}_peaks"] = len(out["mz"])

    nmr = set()
    for name in ("train", "val", "test"):
        path = os.path.join(args.nmr_dir, f"{name}.jsonl.gz")
        if os.path.exists(path):
            with gzip.open(path, "rt") as f:
                nmr |= {json.loads(line)["smiles"] for line in f}
    for name, out in splits.items():
        stats[f"{name}_molecules_with_nmr"] = sum(s in nmr for s in out["smiles"])

    with open(os.path.join(args.output_dir, "stats.json"), "w") as f:
        json.dump({"stats": dict(sorted(stats.items())), "args": vars(args)}, f, indent=2)
    print(json.dumps(dict(sorted(stats.items())), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert experimental MS2 spectra into BLIND splits.")
    parser.add_argument("--input-path", default="data/ms/cleaned_all.jsonl.gz")
    parser.add_argument("--output-dir", default="data/ms/processed")
    parser.add_argument("--tokenizer-path", default="data/nmrexp/processed/smiles_tokenizer.model")
    parser.add_argument("--nmr-dir", default="data/nmrexp/processed", help="NMR splits, only to count shared molecules.")
    parser.add_argument("--exclude-csv", default="data/external_validation_results.csv")
    parser.add_argument("--max-smiles-len", type=int, default=275, help="Including BOS/EOS, as in training.")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 1)
    return parser.parse_args()
