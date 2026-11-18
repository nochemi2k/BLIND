"""
MS2 spectra as an extra input to the model with elemental analysis, without touching the main code.

Importing this module points training and evaluation at MSTransformer (src/blind/models/ms_transformer.py)
and at the datasets of modalities/ea extended with one MS2 spectrum per example, from the splits of
prepare_ms.py (MS_DIR/<split>.npz). A spectrum is given as its peaks only (m/z, intensity), without the
precursor or acquisition metadata. The spectrum is added to the 13C batch dict as "MS/<field>" keys,
so the main loops run unchanged.

Training is a fine-tune of a trained model with elemental analysis (INIT_CHECKPOINT). Every epoch:
  - the NMR molecules as in modalities/ea (all with heteronuclear spectra or an MS2 spectrum, plus
    PLAIN_PER_HETERO times as many others), with their elemental analysis (dropped with EA_DROP_PROB)
    and, if they have one, a random MS2 spectrum (dropped with MS_DROP_PROB);
  - the molecules that have MS2 spectra but no NMR spectra (MS_ONLY_PER_EPOCH of them, all by default):
    a random MS2 spectrum and, with probability MS_ONLY_EA_PROB, an elemental analysis from the generator,
    so the model learns MS2 alone and MS2 with the analysis.
The NMR splits and the MS2 splits use the same molecule hash, so no test molecule is trained on.
In evaluation, example k of a molecule gets its k-th MS2 spectrum (if it has any).
"""
import sys
import typing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ea"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hetero"))

import numpy as np
import sentencepiece as spm
import torch
from tqdm.auto import tqdm

import enable_ea
import enable_hetero
from blind import evaluation, evaluation_top_n, training
from blind.data_io import load_jsonl_data
from blind.ea_noise_flow import ElementalAnalysisFlow
from blind.elemental_analysis import ElementalAnalysisNoise, formula_counts
from blind.models.hetero_transformer import HETERO_TYPES
from blind.models.ms_transformer import MS_PAD, MS_PREFIX, MSTransformer
from blind.ms_spectra import META_FIELDS
from blind.training import load_multiplicity_codes, process_data
from enable_ea import CollateEAEvalWrapper, CollateEATrainWrapper, EANMRDataset
from enable_hetero import BalancedHeteroSampler, pop_argument  # noqa: F401 (pop_argument used by entry points)

INIT_CHECKPOINT: typing.Optional[str] = None  # set by train.py: a model with elemental analysis
RESUME_CHECKPOINT: typing.Optional[str] = None  # set by train.py: a model with MS2 spectra to train further
USE_MS = True  # evaluate*.py --no-ms sets False
MS_DIR = "data/ms/processed"

MS_DROP_PROB = 0.3  # NMR molecules with an MS2 spectrum
MS_ONLY_EA_PROB = 0.5  # molecules with MS2 only: probability to get an elemental analysis
MS_ONLY_PER_EPOCH: typing.Optional[int] = None  # None: all molecules with MS2 only, every epoch


def build_model(tokenizer: spm.SentencePieceProcessor, smiles_pad_token: int) -> MSTransformer:
    return MSTransformer(
        ms_mz_hidden_dim=144,
        ms_intensity_hidden_dim=48,  # 192 per peak
        ms_num_heads=6,
        ms_num_layers=4,
        ms_d_ff=384,
        **enable_ea.model_kwargs(tokenizer, smiles_pad_token),
    )


def build_model_from_ea(tokenizer: spm.SentencePieceProcessor, smiles_pad_token: int) -> MSTransformer:
    model = build_model(tokenizer, smiles_pad_token)
    if RESUME_CHECKPOINT is not None:
        # A model with MS2 spectra: all weights as they are (init_from_ea would zero the MS2 cross-attentions)
        model.load_state_dict(torch.load(RESUME_CHECKPOINT, map_location="cpu"))
        print(f"Resumed from {RESUME_CHECKPOINT} (optimizer state starts anew)")
        return model
    if INIT_CHECKPOINT is None:
        raise ValueError("--init-checkpoint (trained model with elemental analysis) or --resume-checkpoint is required")
    model.init_from_ea(torch.load(INIT_CHECKPOINT, map_location="cpu"))
    print(f"Initialized from {INIT_CHECKPOINT}; MS2 encoder and cross-attentions are new")
    return model


class MSStore:
    """MS2 spectra of one split (prepare_ms.py), kept as flat arrays; tensors are built per item."""

    def __init__(self, path: str) -> None:
        data = np.load(path, allow_pickle=True)
        self.smiles: typing.List[str] = data["smiles"].tolist()
        self.offsets = data["offsets"]
        self.mz = data["mz"]
        self.intensity = data["intensity"]
        self.precursor_mz = data["precursor_mz"]
        self.collision_energy = data["collision_energy"]
        self.meta = {field: data[field] for field in META_FIELDS}
        self.by_smiles: typing.Dict[str, typing.List[int]] = {}
        for spectrum, molecule in enumerate(data["spectrum_smiles"].tolist()):
            self.by_smiles.setdefault(self.smiles[molecule], []).append(spectrum)
        print(f"{path}: {len(self.smiles)} molecules, {len(self.precursor_mz)} MS2 spectra")

    def spectra(self, smiles: str) -> typing.List[int]:
        return self.by_smiles.get(smiles, [])

    def tensorize(self, spectrum: int) -> typing.Dict[str, torch.Tensor]:
        """Peaks only: m/z and intensity (the metadata stay in the store, for evaluate_ms.py groups)."""
        lo, hi = self.offsets[spectrum], self.offsets[spectrum + 1]
        return {
            "mz": torch.from_numpy(self.mz[lo:hi].copy()),
            "intensity": torch.from_numpy(self.intensity[lo:hi].copy()),
        }


def load_store(split: str) -> typing.Optional[MSStore]:
    path = Path(MS_DIR) / f"{split}.npz"
    if not path.exists():
        print(f"No MS2 spectra at {path}")
        return None
    return MSStore(str(path))


class MSNMRDataset(EANMRDataset):
    """EANMRDataset + a random MS2 spectrum of the molecule (if it has one), dropped with MS_DROP_PROB."""

    def __init__(self, data, solvent, tokenizer, smiles_bos_id, smiles_eos_id, ms_store: typing.Optional[MSStore]) -> None:
        super().__init__(data, solvent, tokenizer, smiles_bos_id, smiles_eos_id)
        self.ms_store = ms_store

    def has_ms(self, idx: int) -> bool:
        return self.ms_store is not None and bool(self.ms_store.spectra(self.idx2smiles[idx]))

    def has_hetero(self, idx: int) -> bool:
        # BalancedHeteroSampler takes every molecule with heteronuclear or MS2 spectra
        return super().has_hetero(idx) or self.has_ms(idx)

    def __getitem__(self, idx: int) -> typing.Dict[str, typing.Any]:
        item = super().__getitem__(idx)
        spectra = self.ms_store.spectra(self.idx2smiles[idx]) if self.ms_store is not None else []
        item["ms"] = None
        if spectra and self.rng.random() >= MS_DROP_PROB:
            item["ms"] = self.ms_store.tensorize(spectra[self.rng.integers(len(spectra))])
        return item


class MSOnlyDataset(torch.utils.data.Dataset):
    """Molecules with MS2 spectra but no NMR spectra: a random MS2 spectrum (+ an analysis with MS_ONLY_EA_PROB)."""

    def __init__(self, ms_store: MSStore, exclude: typing.Set[str], tokenizer: spm.SentencePieceProcessor) -> None:
        self.ms_store = ms_store
        self.smiles = [s for s in ms_store.smiles if s not in exclude]
        self.encoded_smiles = [
            torch.tensor([tokenizer.bos_id()] + tokenizer.encode(s) + [tokenizer.eos_id()], dtype=torch.int64)
            for s in tqdm(self.smiles, desc="MS2-only SMILES")
        ]
        # The elemental analysis of modalities/ea, drawn the same way (EANMRDataset.simulate)
        self.formulas = {s: formula_counts(s) for s in tqdm(self.smiles, desc="Formulas")}
        if enable_ea.EA_NOISE == "flow":
            print(f"Generating {enable_ea.FLOW_SAMPLES} elemental analyses per MS2-only molecule with {enable_ea.FLOW_PATH}")
            found = ElementalAnalysisFlow(enable_ea.FLOW_PATH).sample(self.smiles, enable_ea.FLOW_SAMPLES)
            self.flow_found = dict(zip(self.smiles, found))
        else:
            self.noise = ElementalAnalysisNoise(enable_ea.NOISE_MODEL_PATH)
        self._rng: typing.Optional[np.random.Generator] = None

    rng = EANMRDataset.rng
    simulate = EANMRDataset.simulate

    def __len__(self) -> int:
        return len(self.smiles)

    def item(self, idx: int, spectrum: int, ea: typing.Optional[typing.Dict[str, torch.Tensor]]) -> typing.Dict[str, typing.Any]:
        return {
            "smiles": self.encoded_smiles[idx],
            "carbon_spectrum": None,
            "hydrogen_spectrum": None,
            "hetero": {spectrum_type: None for spectrum_type in HETERO_TYPES},
            "ea": ea,
            "ms": self.ms_store.tensorize(spectrum),
        }

    def __getitem__(self, idx: int) -> typing.Dict[str, typing.Any]:
        spectra = self.ms_store.spectra(self.smiles[idx])
        ea = self.simulate(self.smiles[idx], self.rng) if self.rng.random() < MS_ONLY_EA_PROB else None
        return self.item(idx, spectra[self.rng.integers(len(spectra))], ea)


class CombinedSampler(torch.utils.data.Sampler):
    """The NMR molecules of BalancedHeteroSampler, then MS_ONLY_PER_EPOCH molecules with MS2 only, shuffled together."""

    def __init__(self, nmr: MSNMRDataset, num_ms_only: int, seed: int, fixed: bool) -> None:
        self.nmr_sampler = BalancedHeteroSampler(nmr, seed=seed, fixed=fixed)
        self.offset = len(nmr)
        self.num_ms_only_total = num_ms_only
        self.num_ms_only = num_ms_only if MS_ONLY_PER_EPOCH is None else min(MS_ONLY_PER_EPOCH, num_ms_only)
        self.seed = seed
        self.fixed = fixed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.nmr_sampler) + self.num_ms_only

    def __iter__(self) -> typing.Iterator[int]:
        rng = np.random.default_rng(self.seed if self.fixed else self.seed + 1000 + self.epoch)
        self.epoch += 1
        ms_only = self.offset + rng.choice(self.num_ms_only_total, size=self.num_ms_only, replace=False)
        order = np.concatenate([np.fromiter(iter(self.nmr_sampler), dtype=np.int64), ms_only])
        rng.shuffle(order)
        return iter(order.tolist())


def add_ms(collated: typing.Dict[str, typing.Any], batch: typing.List[typing.Dict[str, typing.Any]]) -> typing.Dict[str, typing.Any]:
    for field, pad_value in (("mz", MS_PAD), ("intensity", 0.0)):
        sequences = [
            cur["ms"][field] if cur.get("ms") is not None else torch.tensor([pad_value], dtype=torch.float32)
            for cur in batch
        ]
        collated["C_NMR"][f"{MS_PREFIX}{field}"] = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=pad_value)
    return collated


class CollateMSTrainWrapper(CollateEATrainWrapper):
    def __call__(self, batch):
        return add_ms(super().__call__(batch), batch)


class CollateMSEvalWrapper(CollateEAEvalWrapper):
    def __call__(self, batch):
        return add_ms(super().__call__(batch), batch)


def build_dataloaders(
    tokenizer: spm.SentencePieceProcessor,
    multiplicity_codes_path: str,
    train_path: str,
    val_path: str,
    solvent: str,
    batch_size: int,
    num_workers: int,
    max_smiles_len: int,
    pin_memory: bool = False,
) -> typing.Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    multiplicity_codes = load_multiplicity_codes(multiplicity_codes_path)

    def load_split(path: str) -> typing.Dict[str, typing.List[dict]]:
        data = load_jsonl_data(path)
        processed = {
            k: [process_data(v_i, multiplicity_codes) for v_i in v]
            for k, v in tqdm(data.items(), desc="Processing spectra")
        }
        for value in processed.values():
            for cur in value:
                cur["solvent"] = solvent
        return processed

    collate = CollateMSTrainWrapper(smiles_pad_id=tokenizer.pad_id(), spectrum_pad_token=-1000.0, max_smiles_len=max_smiles_len)

    loaders = []
    for name, path in (("train", train_path), ("val", val_path)):
        store = load_store(name)
        nmr = MSNMRDataset(
            data=load_split(path),
            solvent=solvent,
            tokenizer=tokenizer,
            smiles_bos_id=tokenizer.bos_id(),
            smiles_eos_id=tokenizer.eos_id(),
            ms_store=store,
        )
        datasets: typing.List[torch.utils.data.Dataset] = [nmr]
        num_ms_only = 0
        if store is not None:
            ms_only = MSOnlyDataset(store, exclude=set(nmr.smiles2idx), tokenizer=tokenizer)
            datasets.append(ms_only)
            num_ms_only = len(ms_only)
        sampler = CombinedSampler(nmr, num_ms_only, seed=0, fixed=name == "val")
        print(
            f"{name}: {len(sampler.nmr_sampler.hetero_idx)} NMR molecules with heteronuclear or MS2 spectra + "
            f"{sampler.nmr_sampler.num_plain} of {len(sampler.nmr_sampler.plain_idx)} others + "
            f"{sampler.num_ms_only} of {num_ms_only} with MS2 only, per epoch"
        )
        loaders.append(
            torch.utils.data.DataLoader(
                torch.utils.data.ConcatDataset(datasets),
                batch_size=batch_size,
                sampler=sampler,
                collate_fn=collate,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=num_workers > 0,
            )
        )
    return loaders[0], loaders[1]


def split_name(data_path: str) -> str:
    name = Path(data_path).name
    return next((s for s in ("train", "val", "test") if s in name), "test")


def build_eval_dataset(
    tokenizer: spm.SentencePieceProcessor,
    multiplicity_codes_path: str,
    data_path: str,
    solvent: str,
) -> torch.utils.data.Dataset:
    dataset = enable_ea.build_eval_dataset(tokenizer, multiplicity_codes_path, data_path, solvent)
    store = load_store(split_name(data_path)) if USE_MS else None
    combo_idx: typing.Dict[str, int] = {}
    with_ms = 0
    for example, smiles in zip(dataset.examples, dataset.example_smiles):
        k = combo_idx.get(smiles, 0)
        combo_idx[smiles] = k + 1
        spectra = store.spectra(smiles) if store is not None else []
        example["ms"] = store.tensorize(spectra[k % len(spectra)]) if spectra else None
        with_ms += bool(spectra)
    print(f"MS2 spectra: {'ON' if USE_MS else 'OFF'}; {with_ms} of {len(dataset.examples)} examples have one")
    return dataset


training.build_dataloaders = build_dataloaders
training.build_model = build_model_from_ea
for module in (evaluation, evaluation_top_n):
    module.build_model = build_model
    module.build_eval_dataset = build_eval_dataset
    module.CollateEvalWrapper = CollateMSEvalWrapper
