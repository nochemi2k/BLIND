#!/usr/bin/env python3
"""
Data for training one model from scratch on all inputs (modalities/joint/train.py).

1. A SMILES tokenizer (BPE, the settings of prepare_nmrexp.py) trained on the training molecules of
   both datasets: NMR (data/nmrexp/processed_hetero/train.jsonl.gz) and MS2 (the training split of
   data/ms/cleaned_all.jsonl.gz by the molecule hash of prepare_nmrexp.py).
2. The MS2 splits of prepare_ms.py, filtered with that tokenizer, in <output-dir>/ms/.
The NMR splits stay where they are; molecules the new tokenizer cannot encode within the length limit
are dropped when training starts.

    python modalities/joint/prepare_joint.py --output-dir data/joint
Output: <output-dir>/smiles_tokenizer.model, <output-dir>/ms/{train,val,test}.npz, <output-dir>/stats.json
"""
import argparse
import gzip
import json
import multiprocessing
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import sentencepiece as spm
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm

from blind import prepare_ms
from blind.prepare_nmrexp import load_excluded_keys, split_of, train_tokenizer


def _key(smiles: str):
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    return smiles, Chem.MolToSmiles(mol, isomericSmiles=False) if mol is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--output-dir", default="data/joint")
    parser.add_argument("--nmr-dir", default="data/nmrexp/processed_hetero")
    parser.add_argument("--ms-input", default="data/ms/cleaned_all.jsonl.gz")
    parser.add_argument("--exclude-csv", default="data/external_validation_results.csv")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--max-smiles-len", type=int, default=275)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with gzip.open(os.path.join(args.nmr_dir, "train.jsonl.gz"), "rt") as f:
        nmr_train = {json.loads(line)["smiles"] for line in tqdm(f, desc="NMR train")}
    ms_molecules = set()
    with gzip.open(args.ms_input, "rt") as f:
        for line in tqdm(f, desc="MS2 molecules"):
            obj = json.loads(line)
            if obj["record"].get("ms_level") == 2:
                ms_molecules.add(obj["smiles"])
    excluded = load_excluded_keys(args.exclude_csv)
    ms_train = set()
    with multiprocessing.Pool(args.num_workers) as pool:
        for smiles, key in tqdm(pool.imap_unordered(_key, sorted(ms_molecules), chunksize=256), total=len(ms_molecules), desc="MS2 splits"):
            if key is not None and key not in excluded and split_of(key, args.val_frac, args.test_frac) == "train":
                ms_train.add(smiles)

    corpus = sorted(nmr_train | ms_train)
    print(f"Tokenizer corpus: {len(nmr_train)} NMR + {len(ms_train)} MS2 training molecules, {len(corpus)} unique")
    tokenizer_path = train_tokenizer(corpus, args.output_dir, args.vocab_size)
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)

    stats = {"tokenizer_corpus": len(corpus), "nmr_train_molecules": len(nmr_train), "ms_train_molecules": len(ms_train)}
    for split in ("train", "val", "test"):
        with gzip.open(os.path.join(args.nmr_dir, f"{split}.jsonl.gz"), "rt") as f:
            smiles = {json.loads(line)["smiles"] for line in f}
        ids = [tokenizer.encode(s) for s in smiles]
        stats[f"nmr_{split}_molecules"] = len(smiles)
        stats[f"nmr_{split}_dropped_too_long_or_unk"] = sum(
            len(i) + 2 > args.max_smiles_len or tokenizer.unk_id() in i for i in ids
        )

    prepare_ms.prepare(argparse.Namespace(
        input_path=args.ms_input,
        output_dir=os.path.join(args.output_dir, "ms"),
        tokenizer_path=tokenizer_path,
        nmr_dir=args.nmr_dir,
        exclude_csv=args.exclude_csv,
        max_smiles_len=args.max_smiles_len,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        num_workers=args.num_workers,
    ))
    stats["ms"] = json.load(open(os.path.join(args.output_dir, "ms", "stats.json")))["stats"]
    with open(os.path.join(args.output_dir, "stats.json"), "w") as f:
        json.dump({"stats": stats, "args": vars(args)}, f, indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "ms"}, indent=2))


if __name__ == "__main__":
    main()
