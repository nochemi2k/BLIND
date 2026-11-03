#!/usr/bin/env python3
"""
Check that train / val / test splits share no molecules.

Molecules are compared at several levels, from strict to loose:
  smiles       canonical isomeric SMILES as stored in the JSONL
  no_stereo    canonical SMILES without stereo (the split key of prepare_nmrexp.py)
  inchikey     full standard InChIKey
  skeleton     first InChIKey block (connectivity; ignores stereo, isotopes, protonation)
Identical spectra shared between splits are reported too, as is overlap with the
external validation set that prepare_nmrexp.py excludes.
"""
import argparse
import gzip
import itertools
import json
import multiprocessing
import sys
import typing
from collections import defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SPLITS = ("train", "val", "test")
LEVELS = ("smiles", "no_stereo", "inchikey", "skeleton")


def read_split(path: Path) -> typing.Tuple[typing.Set[str], typing.Dict[str, typing.Set[str]]]:
    """Return unique SMILES and {spectrum fingerprint: smiles} for one split."""
    smiles: typing.Set[str] = set()
    spectra: typing.Dict[str, typing.Set[str]] = defaultdict(set)
    with gzip.open(path, "rt") as f:
        for line in f:
            record = json.loads(line)
            smiles.add(record["smiles"])
            peaks = tuple(
                (p["signal"], p["intensity"], p["multiplicity"], tuple(p["couplings"]))
                for p in record["spectrum"]
            )
            spectra[f"{record['spectrum_type']}|{peaks}"].add(record["smiles"])
    return smiles, spectra


def molecule_keys(smiles: str) -> typing.Tuple[str, str, str, str]:
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles, "", "", ""
    inchikey = Chem.MolToInchiKey(mol) or ""
    return smiles, Chem.MolToSmiles(mol, isomericSmiles=False), inchikey, inchikey[:14]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/nmrexp/processed"))
    parser.add_argument("--external-csv", type=Path, default=Path("data/external_validation_results.csv"))
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--examples", type=int, default=5, help="Overlapping molecules to print per pair.")
    parser.add_argument(
        "--flagged-output",
        type=Path,
        default=Path("data/nmrexp/processed/test_flagged.json"),
        help="Where to write test molecules that overlap train/val at any level (for analysis).",
    )
    args = parser.parse_args()

    smiles_by_split = {}
    spectra_by_split = {}
    for split in SPLITS:
        smiles_by_split[split], spectra_by_split[split] = read_split(args.data_dir / f"{split}.jsonl.gz")
        print(f"{split}: {len(smiles_by_split[split])} molecules, {len(spectra_by_split[split])} unique spectra")

    all_smiles = sorted(set().union(*smiles_by_split.values()))
    print(f"Computing keys for {len(all_smiles)} molecules on {args.workers} workers...")
    with multiprocessing.Pool(args.workers) as pool:
        keys = {k[0]: k for k in pool.imap_unordered(molecule_keys, all_smiles, chunksize=2000)}
    failed = sum(1 for k in keys.values() if not k[2])
    if failed:
        print(f"WARNING: {failed} molecules failed to parse or produce an InChIKey (ignored for those levels)")

    leaks = 0
    flagged: typing.Dict[str, typing.Set[str]] = defaultdict(set)
    print("\nMolecule overlap between splits:")
    for level_idx, level in enumerate(LEVELS):
        sets = {
            split: {keys[s][level_idx] for s in smiles_by_split[split]} - {""}
            for split in SPLITS
        }
        for a, b in itertools.combinations(SPLITS, 2):
            common = sets[a] & sets[b]
            leaks += len(common) if level in ("smiles", "no_stereo") else 0
            if b == "test":
                for s in smiles_by_split["test"]:
                    if keys[s][level_idx] in common:
                        flagged[s].add(f"{level}:{a}")
            print(f"  {level:<10} {a:>5} ∩ {b:<5}: {len(common)}")
            if common and args.examples:
                by_key = defaultdict(lambda: defaultdict(set))
                for split in (a, b):
                    for s in smiles_by_split[split]:
                        if keys[s][level_idx] in common:
                            by_key[keys[s][level_idx]][split].add(s)
                for key in list(common)[: args.examples]:
                    print(f"      {key}: {a}={sorted(by_key[key][a])} {b}={sorted(by_key[key][b])}")

    print("\nIdentical spectra shared between splits:")
    for a, b in itertools.combinations(SPLITS, 2):
        common = spectra_by_split[a].keys() & spectra_by_split[b].keys()
        print(f"  {a:>5} ∩ {b:<5}: {len(common)}")
        if b == "test":
            for spectrum in common:
                for s in spectra_by_split["test"][spectrum]:
                    flagged[s].add(f"spectrum:{a}")
        for spectrum in list(common)[: args.examples]:
            print(f"      {a}={sorted(spectra_by_split[a][spectrum])} {b}={sorted(spectra_by_split[b][spectrum])}")

    if args.external_csv.exists():
        from blind.data_io import load_external_validation_csv

        external = {molecule_keys(s)[1] for s in load_external_validation_csv(str(args.external_csv))} - {""}
        print(f"\nExternal validation set ({len(external)} molecules), stereo-free overlap:")
        for split in SPLITS:
            common = external & {keys[s][1] for s in smiles_by_split[split]}
            leaks += len(common)
            print(f"  {split}: {len(common)}")

    with open(args.flagged_output, "w") as f:
        json.dump({s: sorted(r) for s, r in sorted(flagged.items())}, f, indent=1)
    print(f"\n{len(flagged)} test molecules overlap train/val at a looser level -> {args.flagged_output}")

    print("\nRESULT:", "no leakage by SMILES" if leaks == 0 else f"LEAKAGE FOUND ({leaks} shared keys)")
    sys.exit(1 if leaks else 0)


if __name__ == "__main__":
    main()
