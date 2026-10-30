"""
Convert the NMRexp database (https://zenodo.org/records/17296666) into BLIND training data.

Keeps 1H and 13C spectra in the selected solvent, canonicalizes SMILES with RDKit,
deduplicates spectra, splits by molecule (stereo-agnostic key, so stereoisomers never
cross splits), trains a new SentencePiece tokenizer on the training SMILES and drops
molecules whose tokenized SMILES do not fit the model.
"""
import argparse
import ast
import gzip
import hashlib
import json
import math
import multiprocessing
import os
import re
import typing
from collections import Counter

import pyarrow.parquet as pq
import sentencepiece as spm
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm

from .data_io import load_external_validation_csv

NMR_TYPES = {"1H NMR": "H_NMR", "13C NMR": "C_NMR"}
SHIFT_RANGES = {"H_NMR": (-5.0, 20.0), "C_NMR": (-50.0, 300.0)}
H_COUNT_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*H\s*$")
SPLITS = ("train", "val", "test")

MAX_COUPLING_HZ = 1000.0
COUPLING_RE = re.compile(r"\d+(?:\.\d+)?")

# Peak = (signal, intensity, multiplicity, couplings)
Peak = typing.Tuple[float, float, str, typing.Tuple[float, ...]]


def normalize_multiplicity(label: typing.Any, default: str) -> str:
    """'br s' -> 'brs', 'app t' -> 'appt', 'quint.' -> 'quint' (keys of multiplicity_codes.json)."""
    if label is None:
        return default
    label = re.sub(r"[^0-9A-Za-z_]", "", str(label))
    return label or default


def parse_couplings(raw: typing.Any) -> typing.Optional[typing.Tuple[float, ...]]:
    """Return J values in Hz sorted descending; None if any value is implausible.

    NMRexp stores them as None, a float, a list, or a stringified list of '7.2Hz' items.
    """
    if isinstance(raw, str):
        raw = ast.literal_eval(raw)
    if raw is None:
        return ()
    values = []
    for item in raw if isinstance(raw, list) else [raw]:
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


def parse_h_peaks(processed: str) -> typing.Optional[typing.List[Peak]]:
    peaks = []
    for mult, raw_couplings, count, hi, lo in ast.literal_eval(processed):
        match = H_COUNT_RE.match(str(count))
        couplings = parse_couplings(raw_couplings)
        if match is None or hi is None or lo is None or couplings is None:
            return None
        peaks.append(
            (
                round((float(hi) + float(lo)) / 2, 4),
                float(match.group(1)),
                normalize_multiplicity(mult, default="m"),
                couplings,
            )
        )
    return peaks


def parse_c_peaks(processed: str) -> typing.Optional[typing.List[Peak]]:
    peaks = []
    for shift, mult, raw_couplings in ast.literal_eval(processed):
        couplings = parse_couplings(raw_couplings)
        if shift is None or couplings is None:
            return None
        # A coupled carbon without a reported label is not a singlet.
        default = "m" if couplings else "s"
        peaks.append((round(float(shift), 4), 1.0, normalize_multiplicity(mult, default), couplings))
    return peaks


def is_valid_spectrum(peaks: typing.List[Peak], spectrum_type: str, max_peaks: int) -> bool:
    if not 0 < len(peaks) <= max_peaks:
        return False
    low, high = SHIFT_RANGES[spectrum_type]
    for signal, intensity, _, _ in peaks:
        if not (math.isfinite(signal) and low <= signal <= high):
            return False
        if not (math.isfinite(intensity) and 0 < intensity <= 200):
            return False
    return True


def canonicalize(smiles: str) -> typing.Tuple[typing.Optional[str], typing.Optional[str]]:
    """Return (canonical isomeric SMILES, stereo-agnostic split key)."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None, None
    return Chem.MolToSmiles(mol), Chem.MolToSmiles(mol, isomericSmiles=False)


_smiles_cache: typing.Dict[str, typing.Tuple[typing.Optional[str], typing.Optional[str]]] = {}


def _init_worker() -> None:
    RDLogger.DisableLog("rdApp.*")


def _process_chunk(
    args: typing.Tuple[typing.List[typing.Tuple[str, str, str]], int]
) -> typing.Tuple[typing.List[typing.Tuple[str, str, str, typing.Tuple[Peak, ...]]], Counter]:
    rows, max_peaks = args
    records = []
    stats: Counter = Counter()
    for raw_smiles, spectrum_type, processed in rows:
        if raw_smiles not in _smiles_cache:
            if len(_smiles_cache) > 200_000:
                _smiles_cache.clear()
            _smiles_cache[raw_smiles] = canonicalize(raw_smiles)
        smiles, key = _smiles_cache[raw_smiles]
        if smiles is None:
            stats["invalid_smiles"] += 1
            continue

        try:
            parse = parse_h_peaks if spectrum_type == "H_NMR" else parse_c_peaks
            peaks = parse(processed)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            peaks = None
        if peaks is None:
            stats["unparsable_peaks"] += 1
            continue
        if not is_valid_spectrum(peaks, spectrum_type, max_peaks):
            stats["invalid_spectrum"] += 1
            continue

        stats["parsed_spectra"] += 1
        for peak in peaks:
            stats[f"peaks_with_{len(peak[3])}_couplings"] += 1
        records.append((smiles, key, spectrum_type, tuple(peaks)))
    return records, stats


def iter_parquet_chunks(
    path: str, solvent: str, chunk_size: int, max_peaks: int, stats: Counter
) -> typing.Iterator[typing.Tuple[typing.List[typing.Tuple[str, str, str]], int]]:
    parquet = pq.ParquetFile(path)
    columns = ["SMILES", "NMR_type", "NMR_solvent", "NMR_processed"]
    chunk: typing.List[typing.Tuple[str, str, str]] = []
    for batch in parquet.iter_batches(batch_size=100_000, columns=columns):
        data = batch.to_pydict()
        for raw_smiles, nmr_type, nmr_solvent, processed in zip(
            data["SMILES"], data["NMR_type"], data["NMR_solvent"], data["NMR_processed"]
        ):
            stats["rows_total"] += 1
            if nmr_type not in NMR_TYPES:
                stats["skipped_nucleus"] += 1
                continue
            if nmr_solvent != solvent:
                stats["skipped_solvent"] += 1
                continue
            chunk.append((raw_smiles, NMR_TYPES[nmr_type], processed))
            if len(chunk) >= chunk_size:
                yield chunk, max_peaks
                chunk = []
    if chunk:
        yield chunk, max_peaks


def split_of(key: str, val_frac: float, test_frac: float) -> str:
    bucket = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < test_frac:
        return "test"
    if bucket < test_frac + val_frac:
        return "val"
    return "train"


def load_excluded_keys(path: typing.Optional[str]) -> typing.Set[str]:
    if not path:
        return set()
    keys = set()
    for smiles in load_external_validation_csv(path):
        _, key = canonicalize(smiles)
        if key is not None:
            keys.add(key)
    return keys


def train_tokenizer(smiles: typing.Iterable[str], output_dir: str, vocab_size: int) -> str:
    """Train a BPE tokenizer with the same settings as models/smiles_tokenizer.model."""
    corpus_path = os.path.join(output_dir, "tokenizer_corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for s in smiles:
            f.write(s + "\n")

    model_prefix = os.path.join(output_dir, "smiles_tokenizer")
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        model_type="bpe",
        vocab_size=vocab_size,
        character_coverage=1.0,
        bos_id=0,
        eos_id=1,
        pad_id=2,
        unk_id=3,
        input_sentence_size=0,
        num_threads=os.cpu_count() or 1,
    )
    os.remove(corpus_path)
    return model_prefix + ".model"


def prepare(args: argparse.Namespace) -> None:
    RDLogger.DisableLog("rdApp.*")
    os.makedirs(args.output_dir, exist_ok=True)

    stats: Counter = Counter()
    # smiles -> (split key, {(spectrum_type, peaks): None}); the dict keeps order and dedups
    molecules: typing.Dict[
        str, typing.Tuple[str, typing.Dict[typing.Tuple[str, typing.Tuple[Peak, ...]], None]]
    ] = {}

    with multiprocessing.Pool(processes=args.num_workers, initializer=_init_worker) as pool:
        chunks = iter_parquet_chunks(
            args.input_path, args.source_solvent, args.chunk_size, args.max_peaks, stats
        )
        for records, chunk_stats in tqdm(
            pool.imap(_process_chunk, chunks), desc="Converting NMRexp chunks"
        ):
            stats.update(chunk_stats)
            for smiles, key, spectrum_type, peaks in records:
                entry = molecules.setdefault(smiles, (key, {}))
                if (spectrum_type, peaks) in entry[1]:
                    stats["duplicate_spectrum"] += 1
                else:
                    entry[1][(spectrum_type, peaks)] = None

    excluded_keys = load_excluded_keys(args.exclude_csv)
    split_by_smiles: typing.Dict[str, str] = {}
    for smiles, (key, _) in molecules.items():
        if key in excluded_keys:
            stats["excluded_molecules"] += 1
            continue
        split_by_smiles[smiles] = split_of(key, args.val_frac, args.test_frac)

    tokenizer_path = train_tokenizer(
        (s for s, split in split_by_smiles.items() if split == "train"),
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
    )
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)

    split_counts: Counter = Counter()
    paths = {name: os.path.join(args.output_dir, f"{name}.jsonl.gz") for name in SPLITS}
    files = {name: gzip.open(path, "wt", encoding="utf-8") for name, path in paths.items()}
    try:
        for smiles, split in tqdm(split_by_smiles.items(), desc="Writing splits"):
            ids = tokenizer.encode(smiles)
            if len(ids) + 2 > args.max_smiles_len:
                stats["smiles_too_long"] += 1
                continue
            if tokenizer.unk_id() in ids:
                stats["smiles_unk_token"] += 1
                continue

            split_counts[f"{split}_molecules"] += 1
            for spectrum_type, peaks in molecules[smiles][1]:
                split_counts[f"{split}_{spectrum_type}"] += 1
                record = {
                    "smiles": smiles,
                    "spectrum_type": spectrum_type,
                    "solvent": args.solvent,
                    "spectrum": [
                        {"signal": s, "intensity": i, "multiplicity": m, "couplings": list(j)}
                        for s, i, m, j in peaks
                    ],
                }
                files[split].write(json.dumps(record) + "\n")
    finally:
        for f in files.values():
            f.close()

    summary = {
        "filters": dict(stats),
        "splits": dict(split_counts),
        "tokenizer_path": tokenizer_path,
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "stats.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert NMRexp parquet into BLIND JSONL splits.")
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/nmrexp/raw/NMRexp_10to24_1_1004.parquet",
        help="Path to NMRexp parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/nmrexp/processed",
        help="Output directory for splits, tokenizer and stats.",
    )
    parser.add_argument(
        "--exclude-csv",
        type=str,
        default="data/external_validation_results.csv",
        help="Molecules from this CSV are removed from all splits (empty string to disable).",
    )
    parser.add_argument(
        "--source-solvent", type=str, default="CDCl3", help="NMR_solvent value to keep."
    )
    parser.add_argument(
        "--solvent", type=str, default="CDCl_3", help="Solvent name written to JSONL."
    )
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--vocab-size", type=int, default=512, help="Tokenizer vocabulary size.")
    parser.add_argument(
        "--max-smiles-len",
        type=int,
        default=275,
        help="Drop SMILES longer than this many tokens (including BOS/EOS).",
    )
    parser.add_argument(
        "--max-peaks",
        type=int,
        default=400,
        help="Drop spectra with more peaks (model positional encoding limit is 400).",
    )
    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--chunk-size", type=int, default=5000)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
