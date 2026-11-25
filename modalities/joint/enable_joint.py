"""
One model trained from scratch on every input at once: 1H, 13C, heteronuclear spectra, elemental
analysis, MS2 (the architecture of modalities/ms, MSTransformer), so nothing learned earlier is pushed
out by a later stage.

Data: the NMR splits (data/nmrexp/processed_hetero) and the MS2 splits of prepare_joint.py
(JOINT_DIR/ms), with the tokenizer of prepare_joint.py (trained on the SMILES of both datasets).

Every epoch has EPOCH_SIZE examples drawn by fixed shares (MIX) from four groups of molecules, so a
small group is not drowned by a large one, and a large one does not leave the small ones unlearned:
    plain    NMR spectra only (1H and/or 13C)
    hetero   NMR with heteronuclear spectra
    overlap  NMR and MS2 spectra: the only examples where NMR and MS2 meet, repeated many times
    ms_only  MS2 spectra only
A group is drawn without replacement when it is larger than its share, with replacement otherwise.
Inside an example every input is dropped on its own (the probabilities of modalities/hetero, ea, ms):
one of 1H/13C with 0.2 when both exist, each heteronuclear spectrum 0.2, the elemental analysis 0.3
(given to MS2-only molecules with MS_ONLY_EA_PROB), the MS2 spectrum 0.3; at least one spectrum stays.

Validation: the loss on a fixed draw of the validation splits, and greedy exact-match accuracy on fixed
small sets of validation examples for every combination of inputs (SCENARIOS), so forgetting of any
input shows at once.
"""
import hashlib
import sys
import typing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ms"))

import numpy as np
import sentencepiece as spm
import torch
from tqdm.auto import tqdm

import enable_ms
from blind.data_io import load_jsonl_data
from blind.evaluation import generate_batch
from blind.models.hetero_transformer import HETERO_TYPES
from blind.training import load_multiplicity_codes, move_to_device, process_data
from enable_hetero import HeteroNMRDataset
from enable_ms import CollateMSEvalWrapper, CollateMSTrainWrapper, MSNMRDataset, MSOnlyDataset, MSStore

JOINT_DIR = "data/joint"
NMR_DIR = "data/nmrexp/processed_hetero"
EPOCH_SIZE = 1_000_000
VAL_EPOCH_SIZE = 100_000
MIX = {"plain": 0.50, "hetero": 0.20, "overlap": 0.05, "ms_only": 0.25}
SCENARIO_SIZE = 512
# (group, inputs): H/C = 1H/13C spectra (heteronuclear spectra given when the molecule has them), EA, MS
SCENARIOS = {
    "H+C": ("nmr", {"H", "C"}),
    "H+C+EA": ("nmr", {"H", "C", "EA"}),
    "H": ("nmr", {"H"}),
    "H+EA": ("nmr", {"H", "EA"}),
    "C": ("nmr", {"C"}),
    "C+EA": ("nmr", {"C", "EA"}),
    "H+C+MS": ("overlap", {"H", "C", "MS"}),
    "H+C+MS+EA": ("overlap", {"H", "C", "MS", "EA"}),
    "H+MS": ("overlap", {"H", "MS"}),
    "H+MS+EA": ("overlap", {"H", "MS", "EA"}),
    "MS": ("ms_only", {"MS"}),
    "MS+EA": ("ms_only", {"MS", "EA"}),
}


def load_nmr(split: str, tokenizer: spm.SentencePieceProcessor, multiplicity_codes: dict, solvent: str, max_smiles_len: int) -> dict:
    data = load_jsonl_data(str(Path(NMR_DIR) / f"{split}.jsonl.gz"))
    keep, dropped = {}, 0
    for smiles, spectra in tqdm(data.items(), desc=f"NMR {split}"):
        ids = tokenizer.encode(smiles)
        if len(ids) + 2 > max_smiles_len or tokenizer.unk_id() in ids:
            dropped += 1
            continue
        keep[smiles] = [process_data(s, multiplicity_codes) for s in spectra]
        for cur in keep[smiles]:
            cur["solvent"] = solvent
    print(f"NMR {split}: {len(keep)} molecules ({dropped} dropped: too long or unknown token for the tokenizer)")
    return keep


def build_split(split: str, tokenizer, multiplicity_codes_path: str, solvent: str, max_smiles_len: int):
    enable_ms.MS_DIR = str(Path(JOINT_DIR) / "ms")
    store = enable_ms.load_store(split)
    nmr = MSNMRDataset(
        data=load_nmr(split, tokenizer, load_multiplicity_codes(multiplicity_codes_path), solvent, max_smiles_len),
        solvent=solvent, tokenizer=tokenizer, smiles_bos_id=tokenizer.bos_id(), smiles_eos_id=tokenizer.eos_id(),
        ms_store=store,
    )
    ms_only = MSOnlyDataset(store, exclude=set(nmr.smiles2idx), tokenizer=tokenizer)
    return nmr, ms_only, store


def groups(nmr: MSNMRDataset, num_ms_only: int) -> typing.Dict[str, np.ndarray]:
    has_ms = np.array([nmr.has_ms(i) for i in range(len(nmr))])
    has_hetero = np.array([HeteroNMRDataset.has_hetero(nmr, i) for i in range(len(nmr))])
    return {
        "plain": np.flatnonzero(~has_ms & ~has_hetero),
        "hetero": np.flatnonzero(~has_ms & has_hetero),
        "overlap": np.flatnonzero(has_ms),
        "ms_only": len(nmr) + np.arange(num_ms_only),
    }


class MixtureSampler(torch.utils.data.Sampler):
    """epoch_size indices by the shares of MIX; a new draw every epoch unless fixed."""

    def __init__(self, groups: typing.Dict[str, np.ndarray], epoch_size: int, seed: int, fixed: bool) -> None:
        self.groups = {k: v for k, v in groups.items() if len(v)}
        total = sum(MIX[k] for k in self.groups)
        self.counts = {k: int(round(epoch_size * MIX[k] / total)) for k in self.groups}
        self.seed, self.fixed, self.epoch = seed, fixed, 0

    def __len__(self) -> int:
        return sum(self.counts.values())

    def describe(self) -> str:
        return ", ".join(f"{k} {self.counts[k]} of {len(v)} molecules ({self.counts[k] / len(v):.2f} per molecule)" for k, v in self.groups.items())

    def __iter__(self) -> typing.Iterator[int]:
        rng = np.random.default_rng(self.seed if self.fixed else self.seed + self.epoch)
        self.epoch += 1
        order = np.concatenate([
            rng.choice(idx, size=self.counts[k], replace=self.counts[k] > len(idx)) for k, idx in self.groups.items()
        ])
        rng.shuffle(order)
        return iter(order.tolist())


def build_loaders(tokenizer, args) -> typing.Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, tuple]:
    collate = CollateMSTrainWrapper(smiles_pad_id=tokenizer.pad_id(), spectrum_pad_token=-1000.0, max_smiles_len=args.max_smiles_len)
    loaders, val_data = [], None
    for split, size, fixed in (("train", EPOCH_SIZE, False), ("val", VAL_EPOCH_SIZE, True)):
        nmr, ms_only, store = build_split(split, tokenizer, args.multiplicity_codes_path, args.solvent, args.max_smiles_len)
        sampler = MixtureSampler(groups(nmr, len(ms_only)), size, seed=args.seed, fixed=fixed)
        print(f"{split} epoch: {sampler.describe()}")
        loaders.append(torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset([nmr, ms_only]), batch_size=args.batch_size, sampler=sampler,
            collate_fn=collate, num_workers=args.num_workers, pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
        ))
        if split == "val":
            val_data = (nmr, ms_only, store)
    return loaders[0], loaders[1], val_data


def scenario_examples(nmr: MSNMRDataset, ms_only: MSOnlyDataset, store: MSStore, seed: int) -> typing.Dict[str, list]:
    """Fixed examples of every scenario: the first spectrum of each kind, an analysis seeded by the molecule."""
    rng = np.random.default_rng(seed)
    pools = {
        "nmr": [s for s in nmr.smiles2idx if not store.spectra(s)],
        "overlap": [s for s in nmr.smiles2idx if store.spectra(s)],
        "ms_only": ms_only.smiles,
    }
    ms_index = {s: i for i, s in enumerate(ms_only.smiles)}
    out = {}
    for name, (group, inputs) in SCENARIOS.items():
        pool = pools[group]
        if group != "ms_only":
            # Only molecules that have the requested NMR spectra
            pool = [s for s in pool if ("H" not in inputs or nmr.hydrogen_spectra[s]) and ("C" not in inputs or nmr.carbon_spectra[s])]
        chosen = [pool[i] for i in rng.permutation(len(pool))[:SCENARIO_SIZE]]
        examples = []
        for smiles in chosen:
            ea_rng = np.random.default_rng(int(hashlib.md5(smiles.encode()).hexdigest()[:8], 16))
            if group == "ms_only":
                idx = ms_index[smiles]
                ea = ms_only.simulate(smiles, ea_rng) if "EA" in inputs else None
                examples.append(ms_only.item(idx, store.spectra(smiles)[0], ea))
                continue
            examples.append({
                "smiles": nmr.encoded_smiles[smiles],
                "carbon_spectrum": nmr.carbon_spectra[smiles][0] if "C" in inputs else None,
                "hydrogen_spectrum": nmr.hydrogen_spectra[smiles][0] if "H" in inputs else None,
                "hetero": {t: (nmr.hetero_spectra[t][smiles] or [None])[0] for t in HETERO_TYPES},
                "ea": nmr.simulate(smiles, ea_rng) if "EA" in inputs else None,
                "ms": store.tensorize(store.spectra(smiles)[0]) if "MS" in inputs else None,
            })
        out[name] = examples
    return out


@torch.inference_mode()
def scenario_accuracy(model, examples: typing.Dict[str, list], tokenizer, device, batch_size: int) -> typing.Dict[str, float]:
    collate = CollateMSEvalWrapper(smiles_pad_id=tokenizer.pad_id(), spectrum_pad_token=-1000.0, max_smiles_len=275, max_spectrum_len=400)
    model.eval()
    accuracy = {}
    for name, items in examples.items():
        correct = 0
        for start in range(0, len(items), batch_size):
            batch = collate(items[start:start + batch_size])
            true = [tokenizer.decode(t.tolist()) for t in batch["smiles"]]
            pred = generate_batch(model, move_to_device(batch["C_NMR"], device), move_to_device(batch["H_NMR"], device), tokenizer, 275, device)
            correct += sum(p == t for p, t in zip(pred, true))
        accuracy[name] = correct / max(len(items), 1)
    return accuracy
