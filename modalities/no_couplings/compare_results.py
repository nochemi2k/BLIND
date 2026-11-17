#!/usr/bin/env python3
"""
Paired comparison of two evaluate.py result files on the same data (McNemar test).
"""
import argparse
import json
import math
import typing


def load_predictions(path: str) -> typing.List[dict]:
    with open(path) as f:
        return json.load(f)["predictions"]


def modality(record: dict) -> str:
    if record["has_carbon"] and record["has_hydrogen"]:
        return "Both"
    return "Only C" if record["has_carbon"] else "Only H"


def report(name: str, pairs: typing.List[typing.Tuple[dict, dict]]) -> None:
    if not pairs:
        return
    with_j = sum(a["exact_match"] for a, _ in pairs)
    without_j = sum(b["exact_match"] for _, b in pairs)
    only_with_j = sum(a["exact_match"] and not b["exact_match"] for a, b in pairs)
    only_without_j = sum(b["exact_match"] and not a["exact_match"] for a, b in pairs)

    discordant = only_with_j + only_without_j
    chi2 = (abs(only_with_j - only_without_j) - 1) ** 2 / discordant if discordant else 0.0
    p_value = math.erfc(math.sqrt(chi2 / 2)) if discordant else 1.0

    n = len(pairs)
    print(
        f"{name}: n={n}; with J={with_j / n:.4f}; without J={without_j / n:.4f}; "
        f"diff={(with_j - without_j) / n:+.4f}; only with J={only_with_j}; "
        f"only without J={only_without_j}; McNemar p={p_value:.3g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare results of models with and without J.")
    parser.add_argument("--with-j", type=str, required=True, help="evaluate.py output with J.")
    parser.add_argument("--without-j", type=str, required=True, help="evaluate.py output without J.")
    args = parser.parse_args()

    with_j = load_predictions(args.with_j)
    without_j = load_predictions(args.without_j)
    if len(with_j) != len(without_j):
        raise ValueError(f"Different number of records: {len(with_j)} vs {len(without_j)}")

    pairs = list(zip(with_j, without_j))
    for a, b in pairs:
        if (a["true_smiles"], modality(a)) != (b["true_smiles"], modality(b)):
            raise ValueError("Result files are not aligned: were they run on the same data file?")

    for name in ("Only C", "Only H", "Both"):
        report(name, [pair for pair in pairs if modality(pair[0]) == name])
    report("Overall", pairs)


if __name__ == "__main__":
    main()
