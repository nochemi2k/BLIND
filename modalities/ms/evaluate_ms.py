#!/usr/bin/env python3
"""
Evaluates a model with MS2 spectra on molecules given by their MS2 spectrum alone (greedy decoding):
no NMR spectra, with or without an elemental analysis from the generator of modalities/ea.

    python modalities/ms/evaluate_ms.py --checkpoint checkpoints/ms_epoch_29.pt --output-json results/ms_29.json
    python modalities/ms/evaluate_ms.py --checkpoint checkpoints/ms_epoch_29.pt --no-ea --output-json results/ms_29_no_ea.json

All molecules of the MS2 split (prepare_ms.py) are used, also those that have NMR spectra;
--spectra-per-molecule of their spectra (a fixed random choice), each an example. Exact-match accuracy
and SMILES validity overall and by polarity, ionization, instrument class and adduct.
"""
import argparse
import collections
import hashlib
import json
import typing

import enable_ms

import numpy as np
import sentencepiece as spm
import torch
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm

from blind.device import add_device_arg, default_num_workers, resolve_device, use_pin_memory
from blind.evaluation import generate_batch
from blind.ms_spectra import META_VOCABS

RDLogger.DisableLog("rdApp.*")
GROUPS = ("polarity", "ionization", "instrument", "adduct")


class MSEvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, ms_only: "enable_ms.MSOnlyDataset", spectra_per_molecule: int, use_ea: bool, max_molecules: typing.Optional[int], seed: int) -> None:
        rng = np.random.default_rng(seed)
        molecules = np.arange(len(ms_only))
        if max_molecules is not None and max_molecules < len(molecules):
            molecules = np.sort(rng.choice(molecules, size=max_molecules, replace=False))
        self.examples, self.info = [], []
        store = ms_only.ms_store
        for idx in tqdm(molecules.tolist(), desc="Examples"):
            smiles = ms_only.smiles[idx]
            spectra = store.spectra(smiles)
            for k, spectrum in enumerate(rng.permutation(spectra)[:spectra_per_molecule].tolist()):
                ea = None
                if use_ea:
                    ea_seed = int(hashlib.md5(f"{smiles}|{k}".encode()).hexdigest()[:8], 16)
                    ea = ms_only.simulate(smiles, np.random.default_rng(ea_seed))
                self.examples.append(ms_only.item(idx, spectrum, ea))
                info = {field: META_VOCABS[field][int(store.meta[field][spectrum])] for field in GROUPS}
                info["precursor"] = bool(np.isfinite(store.precursor_mz[spectrum]))
                self.info.append(info)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> typing.Dict[str, typing.Any]:
        return self.examples[idx]


def summarize(rows: typing.List[dict]) -> dict:
    n = len(rows)
    return {
        "count": n,
        "valid_ratio": sum(r["valid"] for r in rows) / max(n, 1),
        "correct_ratio": sum(r["exact_match"] for r in rows) / max(n, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--ms-dir", default=enable_ms.MS_DIR)
    parser.add_argument("--tokenizer-path", default="data/nmrexp/processed/smiles_tokenizer.model")
    parser.add_argument("--spectra-per-molecule", type=int, default=1)
    parser.add_argument("--max-molecules", type=int, default=None, help="Random subset of molecules (quick checks).")
    parser.add_argument("--no-ea", action="store_true", help="No elemental analysis.")
    parser.add_argument("--flow-path", default=enable_ms.enable_ea.FLOW_PATH)
    parser.add_argument("--ea-noise", default=enable_ms.enable_ea.EA_NOISE)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=default_num_workers())
    parser.add_argument("--max-tokens", type=int, default=275)
    parser.add_argument("--max-smiles-len", type=int, default=275)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", default=None)
    add_device_arg(parser)
    args = parser.parse_args()
    enable_ms.MS_DIR = args.ms_dir
    enable_ms.enable_ea.FLOW_PATH = args.flow_path
    enable_ms.enable_ea.EA_NOISE = args.ea_noise

    device = resolve_device(args.device)
    print(f"Using device: {device}; elemental analysis: {'OFF' if args.no_ea else 'ON'}")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(args.tokenizer_path)

    store = enable_ms.load_store(args.split)
    dataset = MSEvaluationDataset(
        enable_ms.MSOnlyDataset(store, exclude=set(), tokenizer=tokenizer),
        args.spectra_per_molecule, use_ea=not args.no_ea, max_molecules=args.max_molecules, seed=args.seed,
    )
    collate = enable_ms.CollateMSEvalWrapper(
        smiles_pad_id=tokenizer.pad_id(), spectrum_pad_token=-1000.0, max_smiles_len=args.max_smiles_len, max_spectrum_len=400,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
        num_workers=args.num_workers, pin_memory=use_pin_memory(device),
    )

    model = enable_ms.build_model(tokenizer, smiles_pad_token=tokenizer.pad_id()).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    rows: typing.List[dict] = []
    for batch in tqdm(loader, desc="Evaluating"):
        true_smiles = [tokenizer.decode(t.cpu().numpy().tolist()) for t in batch["smiles"]]
        generated = generate_batch(
            model=model, src_c_nmr=batch["C_NMR"], src_h_nmr=batch["H_NMR"],
            tokenizer=tokenizer, max_tokens=args.max_tokens, device=device,
        )
        for true, pred in zip(true_smiles, generated):
            info = dataset.info[len(rows)]
            rows.append({
                "true_smiles": true, "prediction": pred, "valid": Chem.MolFromSmiles(pred) is not None,
                "exact_match": pred == true, **info,
            })

    summary = {"overall": summarize(rows)}
    for field in GROUPS + ("precursor",):
        groups = collections.defaultdict(list)
        for r in rows:
            groups[str(r[field])].append(r)
        summary[field] = {k: summarize(v) for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))}
    o = summary["overall"]
    print(f"Overall: {o['count']}; Valid Ratio: {o['valid_ratio']:.4f}; Correct Ratio: {o['correct_ratio']:.4f}")
    for field in GROUPS:
        print(f"  by {field}: " + "; ".join(f"{k} {v['correct_ratio']:.3f} (n={v['count']})" for k, v in list(summary[field].items())[:6]))
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"predictions": rows, "summary": summary}, f)


if __name__ == "__main__":
    main()
