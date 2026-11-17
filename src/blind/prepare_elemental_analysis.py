"""
Empirical noise models of elemental analysis (CHN combustion) for the resampling generator of
src/blind/elemental_analysis.py, from the experimental dataset of modalities/ea/build_dataset.py.

Statements whose calculated values match the formula are kept. A noise model holds, per element,
how often it is reported when present, the neutral solvates added to formulas, and the reported
Found - Calcd vectors, which ElementalAnalysisNoise resamples:
    data/ea/ea_noise.json            compounds without metals (the molecules of the NMR data)
    data/ea/ea_noise_complexes.json  metal compounds
    data/ea/ea_noise_all.json        all
"""
import argparse
import collections
import gzip
import json
import os
import re
import typing

from .elemental_analysis import ATOMIC_MASS, mass_percent

# Acids of salts: in the NMR data the salt is part of the SMILES, so only neutral solvates are modelled
SALT_FORMERS = {"HCl", "HBr", "HI", "HF", "C2H2O4", "C2HF3O2", "H2SO4", "H3PO4", "CH4O3S", "HNO3", "HClO4", "C4H4O4"}
# Calculated values must match the formula this closely (abs. %), otherwise the statement
# has a typo in the formula or in the values and its Found - Calcd is meaningless.
CALC_TOLERANCE = 0.06


def is_consistent(row: dict) -> bool:
    # Found may list fewer elements than Calcd (at least two)
    if not row["formula"] or not row["calc"] or len(row["found"]) < 2 or not set(row["found"]) <= set(row["calc"]):
        return False
    expected = mass_percent(row["formula"])
    return all(abs(expected.get(e, 0.0) - v) < CALC_TOLERANCE for e, v in row["calc"].items())



def noise_models_from_dataset(dataset_path: str, output_dir: str, max_abs_delta: float) -> None:
    """Noise models from the experimental dataset of modalities/ea/build_dataset.py (all analyses, with or without
    SMILES): ea_noise.json from compounds without metals (the molecules of the NMR data),
    ea_noise_complexes.json from metal compounds, ea_noise_all.json from all."""
    from .elemental_analysis import formula_counts_plain
    nonmetals = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I", "As", "Te"}
    rows = []
    for line in gzip.open(dataset_path, "rt"):
        r = json.loads(line)
        r["formula"] = formula_counts_plain(r["formula"])
        r["solvates"] = [tuple(x) for x in r["solvates"]]
        rows.append(r)
    groups = {
        "ea_noise.json": [r for r in rows if set(r["formula"]) <= nonmetals],
        "ea_noise_complexes.json": [r for r in rows if not set(r["formula"]) <= nonmetals],
        "ea_noise_all.json": rows,
    }
    for name, group in groups.items():
        model = build_noise_model(group, max_abs_delta, source=f"Experimental elemental analyses ({os.path.basename(dataset_path)}): {name}")
        json.dump(model, open(os.path.join(output_dir, name), "w"))
        summary = {k: v for k, v in model.items() if k not in ("deltas", "solvates")}
        summary["num_deltas"] = len(model["deltas"])
        print(name, json.dumps(summary))


def build_noise_model(rows: typing.List[dict], max_abs_delta: float, source: str = None) -> dict:
    unique = list({json.dumps([r["formula"], r["calc"], r["found"]], sort_keys=True): r for r in rows}.values())
    kept = [r for r in unique if is_consistent(r)]
    # The elements reported are those with a found value (a few papers list more calculated ones)

    present: typing.Counter = collections.Counter()
    reported: typing.Counter = collections.Counter()
    reported_absent: typing.Counter = collections.Counter()
    for r in kept:
        for element in ATOMIC_MASS:
            if r["formula"].get(element):
                present[element] += 1
                reported[element] += element in r["found"]
            elif element in r["found"]:
                reported_absent[element] += 1
    report_prob = {e: reported[e] / present[e] for e in present if present[e] >= 100 and reported[e]}
    # Mostly "N, 0.00" for compounds without nitrogen
    report_absent_prob = {e: reported_absent[e] / (len(kept) - present[e]) for e in report_prob if reported_absent[e]}

    neutral = [r for r in kept if not any(s in SALT_FORMERS for s, _ in r["solvates"])]
    solvated = [r for r in neutral if r["solvates"]]
    solvates = collections.Counter((s, m) for r in solvated for s, m in r["solvates"])

    deltas = []
    for r in kept:
        delta = {e: round(r["found"][e] - r["calc"][e], 2) for e in r["calc"] if e in r["found"]}
        # Larger deviations are typos (values of another compound), not measurement errors
        if all(abs(d) <= max_abs_delta for d in delta.values()):
            deltas.append(delta)
    return {
        "source": source or "PubMed Central open-access articles, 'Anal. Calcd ... Found ...' statements",
        "num_statements": len(rows),
        "num_unique": len(unique),
        "num_consistent": len(kept),
        "max_abs_delta": max_abs_delta,
        "report_prob": report_prob,
        "report_absent_prob": report_absent_prob,
        "solvate_prob": len(solvated) / len(neutral),
        "solvates": [
            [s, m, n] for (s, m), n in solvates.most_common() if n >= 3 and re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", s)
        ],
        "deltas": deltas,
    }


def prepare(args: argparse.Namespace) -> None:
    noise_models_from_dataset(args.dataset, args.output_dir, args.max_abs_delta)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build elemental analysis noise models from the experimental dataset.")
    parser.add_argument("--dataset", default="data/ea/dataset/ea_dataset.jsonl.gz", help="Output of modalities/ea/build_dataset.py.")
    parser.add_argument("--output-dir", type=str, default="data/ea", help="Where the noise models are written.")
    parser.add_argument("--max-abs-delta", type=float, default=2.0, help="Drop Found - Calcd vectors with a larger deviation.")
    return parser.parse_args()
