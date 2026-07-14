"""
Data loading utilities for BLIND training and evaluation.
"""
import ast
import csv
import gzip
import json
import os
import typing
from pathlib import Path

from tqdm.auto import tqdm


def parse_spectrum_cell(cell: str) -> typing.Optional[typing.List[dict]]:
    if not cell or not str(cell).strip():
        return None
    return ast.literal_eval(cell)


def external_validation_row_to_entries(
    row: dict,
    solvent: str = "CDCl_3",
) -> typing.List[dict]:
    """
    Convert one row from external_validation_results.csv into JSONL-style entries.

    Respects the spectra_type column (H_only, C_only, both) so evaluation
    matches the intended modality for each compound.
    """
    smiles = row["true_smiles"]
    spectra_type = (row.get("spectra_type") or "").strip().lower()
    entries = []

    include_h = spectra_type in {"", "h_only", "both"}
    include_c = spectra_type in {"", "c_only", "both"}

    if include_h:
        h_spectrum = parse_spectrum_cell(row.get("H_spectrum", ""))
        if h_spectrum:
            entries.append(
                {
                    "smiles": smiles,
                    "spectrum_type": "H_NMR",
                    "solvent": solvent,
                    "spectrum": h_spectrum,
                }
            )

    if include_c:
        c_spectrum = parse_spectrum_cell(row.get("C_spectrum", ""))
        if c_spectrum:
            entries.append(
                {
                    "smiles": smiles,
                    "spectrum_type": "C_NMR",
                    "solvent": solvent,
                    "spectrum": c_spectrum,
                }
            )

    return entries


def load_external_validation_csv(
    path: str,
    solvent: str = "CDCl_3",
) -> typing.Dict[str, typing.List[dict]]:
    data: typing.Dict[str, typing.List[dict]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in tqdm(rows, desc=f"Loading {os.path.basename(path)}"):
        for entry in external_validation_row_to_entries(row, solvent=solvent):
            data.setdefault(entry["smiles"], []).append(entry)

    return data


def load_jsonl_data(path: str) -> typing.Dict[str, typing.List[dict]]:
    data: typing.Dict[str, typing.List[dict]] = {}
    open_fn = gzip.open if str(path).endswith(".gz") else open
    mode = "rt" if str(path).endswith(".gz") else "r"

    with open_fn(path, mode, encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Loading {os.path.basename(path)}"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data.setdefault(obj["smiles"], []).append(obj)

    return data


def load_spectrum_data(path: str, solvent: str = "CDCl_3") -> typing.Dict[str, typing.List[dict]]:
    """Load evaluation data from JSONL (.jsonl, .jsonl.gz) or external validation CSV."""
    if Path(path).suffix.lower() == ".csv":
        return load_external_validation_csv(path, solvent=solvent)
    return load_jsonl_data(path)
