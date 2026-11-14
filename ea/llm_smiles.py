#!/usr/bin/env python3
"""
LLM translations of the compound names that OPSIN and ea/ligands.py could not turn into SMILES.

ea/build_dataset.py writes the analyses left as llm_todo.jsonl (candidate names from the text before
the statement + target formula, without solvates). Then
    python ea/llm_smiles.py split --output-dir data/ea/dataset      # llm_batches/batch_XXXX.jsonl
    (an LLM writes llm_batches/batch_XXXX.answers.jsonl: {"key": ..., "smiles": ...} per line,
     using `check` to compare its SMILES with the formula and fix them)
    python ea/llm_smiles.py check data/ea/dataset/llm_batches/batch_0000.answers.jsonl
    python ea/llm_smiles.py context PMC1234567 --file <file of the item> --grep L1 3a   # article text
    python ea/llm_smiles.py merge --output-dir data/ea/dataset      # llm_smiles.json
and ea/build_dataset.py is run again: it reads llm_smiles.json and keeps an LLM SMILES only if its
element composition equals the formula of the statement, like any other source.
"""
import argparse
import glob
import json
import os
import sys
import typing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rdkit import Chem, RDLogger

from blind.elemental_analysis import formula_counts, formula_counts_plain

RDLogger.DisableLog("rdApp.*")


def compare(smiles: str, formula: str) -> typing.Tuple[bool, str]:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return False, "invalid SMILES"
    counts = formula_counts(Chem.MolToSmiles(mol))
    target = formula_counts_plain(formula)
    if not counts or not target:
        return False, "composition not computable"
    diff = {e: counts.get(e, 0) - target.get(e, 0) for e in set(counts) | set(target)}
    diff = {e: d for e, d in diff.items() if abs(d) > 1e-6}
    if not diff:
        return True, "ok"
    return False, "SMILES minus formula: " + " ".join(f"{e}{d:+g}" for e, d in sorted(diff.items()))


def split(args) -> None:
    items = [json.loads(line) for line in open(os.path.join(args.output_dir, "llm_todo.jsonl"))]
    if args.metal_only:
        items = [t for t in items if t["metal"]]
    # Complexes first; analyses of one article together, so its text is read once
    items.sort(key=lambda t: (not t["metal"], t.get("pmcid", "")))
    batch_dir = os.path.join(args.output_dir, "llm_batches")
    os.makedirs(batch_dir, exist_ok=True)
    done = {json.loads(line)["key"] for p in glob.glob(os.path.join(batch_dir, "*.jsonl")) if not p.endswith(".answers.jsonl") for line in open(p)}
    items = [t for t in items if t["key"] not in done]
    start = len([p for p in glob.glob(os.path.join(batch_dir, "batch_*.jsonl")) if not p.endswith(".answers.jsonl")])
    for i in range(0, len(items), args.batch_size):
        path = os.path.join(batch_dir, f"batch_{start + i // args.batch_size:04d}.jsonl")
        with open(path, "w") as f:
            for t in items[i:i + args.batch_size]:
                f.write(json.dumps({k: t[k] for k in ("key", "names", "labels", "formula", "formula_text", "pmcid", "file") if k in t}, ensure_ascii=False) + "\n")
    print(f"{len(items)} names in {(len(items) + args.batch_size - 1) // args.batch_size} new batches in {batch_dir}")


def article_paragraphs(pmcid: str, path: str) -> typing.List[str]:
    import gzip
    import re
    if path.endswith(".txt.gz"):
        return [p for p in gzip.open(path, "rt", errors="ignore").read().split("\n\n") if p.strip()]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_dataset import _article_text
    number = pmcid.replace("PMC", "")
    for article in re.split(r"(?=<article[ >])", gzip.open(path, "rt", errors="ignore").read())[1:]:
        if re.search(rf'<article-id pub-id-type="pmc[^"]*">(?:PMC)?{number}<', article):
            return [_article_text(b.group(2))[0] for b in re.finditer(r"<(title|p|td|th|caption)(?:\s[^>]*)?>(.*?)</\1>", article, re.S)]
    return []


def context(args) -> None:
    """Paragraphs of the article that mention the given labels or words (whole tokens), or the whole text."""
    import re
    paragraphs = article_paragraphs(args.pmcid, args.file)
    if args.grep:
        patterns = [re.compile(rf"(?<![\w-]){re.escape(g)}(?![\w-])") for g in args.grep]
        paragraphs = [p for p in paragraphs if any(pt.search(p) for pt in patterns)]
    out, size = [], 0
    for p in paragraphs:
        if size + len(p) > args.max_chars and out:
            out.append(f"... ({len(paragraphs) - len(out)} more paragraphs; narrow --grep or raise --max-chars)")
            break
        out.append(p)
        size += len(p)
    print("\n\n".join(out))


def check(args) -> None:
    ok = total = 0
    for line in open(args.answers):
        if not line.strip():
            continue
        a = json.loads(line)
        if not a.get("smiles"):
            continue
        total += 1
        good, message = compare(a["smiles"], a["key"].rsplit("\t", 1)[1])
        ok += good
        if not good or args.verbose:
            print(f"{message}\t{a['key']}\t{a['smiles']}")
    print(f"formula matches {ok} of {total} SMILES")


def merge(args) -> None:
    path = os.path.join(args.output_dir, "llm_smiles.json")
    llm = json.load(open(path)) if os.path.exists(path) else {}
    added = 0
    for p in sorted(glob.glob(os.path.join(args.output_dir, "llm_batches", "*.answers.jsonl"))):
        for line in open(p):
            if not line.strip():
                continue
            a = json.loads(line)
            options = llm.setdefault(a["key"], [])  # an empty list marks a name already tried
            if a.get("smiles") and compare(a["smiles"], a["key"].rsplit("\t", 1)[1])[0] and a["smiles"] not in options:
                options.append(a["smiles"])
                added += 1
    json.dump(llm, open(path, "w"), ensure_ascii=False)
    print(f"{added} new verified SMILES; {sum(bool(v) for v in llm.values())} of {len(llm)} analyses translated")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("split")
    p.add_argument("--output-dir", default="data/ea/dataset")
    p.add_argument("--batch-size", type=int, default=150)
    p.add_argument("--metal-only", action="store_true")
    p.set_defaults(func=split)
    p = sub.add_parser("check")
    p.add_argument("answers")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=check)
    p = sub.add_parser("context")
    p.add_argument("pmcid")
    p.add_argument("--file", required=True)
    p.add_argument("--grep", nargs="*")
    p.add_argument("--max-chars", type=int, default=8000)
    p.set_defaults(func=context)
    p = sub.add_parser("merge")
    p.add_argument("--output-dir", default="data/ea/dataset")
    p.set_defaults(func=merge)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
