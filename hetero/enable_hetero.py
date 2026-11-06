"""
Heteronuclear spectra (19F, 31P, 11B, 29Si) without touching the main code.

Importing this module points training and evaluation at HeteroTransformer and at datasets that
also return one spectrum of each available heteronucleus. Heteronuclear tensors are added to
the 13C batch dict as "<type>/<field>" keys, so the main loops run unchanged.

Training is a fine-tune of a trained 1H/13C model (INIT_CHECKPOINT): every epoch sees all
molecules with a heteronuclear spectrum plus PLAIN_PER_HETERO times as many randomly drawn
molecules without one.
Each heteronuclear spectrum is dropped with probability HETERO_DROP_PROB (the 13C/1H dropout of
NMRDataset is unchanged, so at least one spectrum always remains). A nucleus absent from the
molecule is given as an empty spectrum with probability EMPTY_SPECTRUM_PROB; a nucleus present
in the molecule but without a spectrum is never given as empty.
"""
import sys
import typing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import sentencepiece as spm
import torch
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm

from blind import evaluation, evaluation_top_n, training
from blind.data_io import load_jsonl_data, load_spectrum_data
from blind.evaluation import EvaluationDataset, collate_eval
from blind.models.hetero_transformer import HETERO_TYPES, SPECTRUM_PAD, HeteroTransformer
from blind.training import NMRDataset, collate_fn, load_multiplicity_codes, process_data

INIT_CHECKPOINT: typing.Optional[str] = None  # set by train.py
USE_HETERO = True  # evaluate*.py --no-hetero sets False
EMPTY_HETERO = False  # evaluate*.py --empty-hetero sets True: absent nuclei get an empty spectrum

HETERO_DROP_PROB = 0.2
EMPTY_SPECTRUM_PROB = 0.5
PLAIN_PER_HETERO = 3  # molecules without heteronuclear spectra per molecule with one, per epoch
HETERO_ELEMENTS = {"F_NMR": "F", "P_NMR": "P", "B_NMR": "B", "Si_NMR": "Si"}
EMPTY = "empty"  # stands for an empty spectrum in dataset items

FIELDS = ("spectrum", "intensity", "multiplicity", "coupling")


def pop_argument(name: str, is_flag: bool = False) -> typing.Any:
    """Remove `name` (and its value) from sys.argv so the main argparse does not see it."""
    if name not in sys.argv:
        return False if is_flag else None
    idx = sys.argv.index(name)
    if is_flag:
        del sys.argv[idx]
        return True
    value = sys.argv[idx + 1]
    del sys.argv[idx : idx + 2]
    return value


def build_model(tokenizer: spm.SentencePieceProcessor, smiles_pad_token: int) -> HeteroTransformer:
    model = HeteroTransformer(
        hetero_types=HETERO_TYPES,
        hetero_spectrum_hidden_dim=56,
        hetero_multiplicity_hidden_dim=16,
        hetero_coupling_slot_dim=6,
        hetero_hidden_dim=96,  # 56 + 16 + 4 * 6
        hetero_num_heads=4,
        hetero_num_layers=2,
        hetero_d_ff=192,
        # Same as training.build_model
        multiplicity_vocab_size=len(NMRDataset.MULTIPLICITY2IDX),
        multiplicity_hidden_dim=48,
        spectrum_d_ff=256,
        spectrum_hidden_dim=104,
        intensity_d_ff=256,
        intensity_hidden_dim=104,
        max_couplings=NMRDataset.MAX_COUPLINGS,
        coupling_d_ff=256,
        coupling_slot_dim=8,
        tgt_vocab_size=tokenizer.vocab_size(),
        d_model=288,
        num_heads=8,
        num_layers=6,
        d_ff=512,
        max_seq_length=400,
        dropout=0.2,
        smiles_pad_token=smiles_pad_token,
    )
    return model


def build_model_from_base(tokenizer: spm.SentencePieceProcessor, smiles_pad_token: int) -> HeteroTransformer:
    if INIT_CHECKPOINT is None:
        raise ValueError("--init-checkpoint (trained 1H/13C weights) is required")
    model = build_model(tokenizer, smiles_pad_token)
    model.init_from_base(torch.load(INIT_CHECKPOINT, map_location="cpu"))
    print(f"Initialized from {INIT_CHECKPOINT}; heteronuclear encoders and cross-attentions are new")
    return model


def _tensorize(spectrum: dict) -> typing.Dict[str, torch.Tensor]:
    return {
        "spectrum": torch.tensor([cur["signal"] for cur in spectrum["spectrum"]], dtype=torch.float32),
        "intensity": torch.tensor([cur["intensity"] for cur in spectrum["spectrum"]], dtype=torch.float32),
        "multiplicity": torch.tensor(
            [NMRDataset.MULTIPLICITY2IDX[cur["multiplicity"]] for cur in spectrum["spectrum"]],
            dtype=torch.int64,
        ),
        "coupling": NMRDataset._coupling_tensor(spectrum["spectrum"]),
    }


def absent_hetero(smiles: str) -> typing.FrozenSet[str]:
    """Heteronuclear spectrum types whose element is not in the molecule."""
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return frozenset()
    elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
    return frozenset(t for t in HETERO_TYPES if HETERO_ELEMENTS[t] not in elements)


class HeteroNMRDataset(NMRDataset):
    """NMRDataset + one random spectrum of each available heteronucleus, or an empty one for absent nuclei."""

    def __init__(
        self,
        data: typing.Dict[str, typing.List[typing.Dict[str, typing.Any]]],
        solvent: str,
        tokenizer: spm.SentencePieceProcessor,
        smiles_bos_id: int,
        smiles_eos_id: int,
    ) -> None:
        super().__init__(data, solvent, tokenizer, smiles_bos_id, smiles_eos_id)
        self.hetero_spectra = {
            spectrum_type: {
                smiles: [
                    _tensorize(spectrum)
                    for spectrum in spectra
                    if spectrum["spectrum_type"] == spectrum_type and spectrum["solvent"] == solvent
                ]
                for smiles, spectra in data.items()
            }
            for spectrum_type in HETERO_TYPES
        }
        self.absent_hetero = {smiles: absent_hetero(smiles) for smiles in tqdm(data.keys(), desc="Elements")}

    def has_hetero(self, idx: int) -> bool:
        smiles = self.idx2smiles[idx]
        return any(self.hetero_spectra[t][smiles] for t in HETERO_TYPES)

    def __getitem__(self, idx: int) -> typing.Dict[str, typing.Any]:
        item = super().__getitem__(idx)
        smiles = self.idx2smiles[idx]
        hetero = {}
        for spectrum_type in HETERO_TYPES:
            spectrum = self._select_spectrum(self.hetero_spectra[spectrum_type][smiles])
            if spectrum is None:
                if spectrum_type in self.absent_hetero[smiles] and np.random.rand() < EMPTY_SPECTRUM_PROB:
                    spectrum = EMPTY
            elif np.random.rand() < HETERO_DROP_PROB:
                spectrum = None
            hetero[spectrum_type] = spectrum
        # Every molecule has a 13C or 1H spectrum and NMRDataset drops at most one of them,
        # so dropping heteronuclear spectra never leaves an example without input
        assert item["carbon_spectrum"] is not None or item["hydrogen_spectrum"] is not None
        item["hetero"] = hetero
        return item


class HeteroEvaluationDataset(EvaluationDataset):
    """Same 13C/1H pairs as EvaluationDataset; pair k of a molecule gets its k-th spectrum of each heteronucleus."""

    def __init__(self, base_dataset: HeteroNMRDataset, use_hetero: bool, empty_hetero: bool) -> None:
        super().__init__(base_dataset)
        combo_idx: typing.Dict[str, int] = {}
        for example, smiles in zip(self.examples, self.example_smiles):
            k = combo_idx.get(smiles, 0)
            combo_idx[smiles] = k + 1
            example["hetero"] = {}
            for spectrum_type in HETERO_TYPES:
                spectra = base_dataset.hetero_spectra[spectrum_type][smiles] if use_hetero else []
                if spectra:
                    example["hetero"][spectrum_type] = spectra[k % len(spectra)]
                elif empty_hetero and spectrum_type in base_dataset.absent_hetero[smiles]:
                    example["hetero"][spectrum_type] = EMPTY
                else:
                    example["hetero"][spectrum_type] = None

    def _build_examples(self, base_dataset: NMRDataset) -> None:
        super()._build_examples(base_dataset)
        self.example_smiles: typing.List[str] = []
        for smiles in base_dataset.smiles2idx.keys():
            n = max(len(base_dataset.carbon_spectra[smiles]), len(base_dataset.hydrogen_spectra[smiles]))
            self.example_smiles.extend([smiles] * n)


def add_hetero(
    collated: typing.Dict[str, typing.Any],
    batch: typing.List[typing.Dict[str, typing.Any]],
    max_spectrum_len: typing.Optional[int] = None,
) -> typing.Dict[str, typing.Any]:
    pad_values = {
        "spectrum": SPECTRUM_PAD,
        "intensity": 0.0,
        "multiplicity": NMRDataset.MULTIPLICITY2IDX["<SYS>"],
        "coupling": NMRDataset.COUPLING_PAD,
    }
    for spectrum_type in HETERO_TYPES:
        columns: typing.Dict[str, typing.List[torch.Tensor]] = {field: [] for field in FIELDS}
        empty = []
        for cur in batch:
            spectrum = cur["hetero"][spectrum_type]
            empty.append(isinstance(spectrum, str))
            if isinstance(spectrum, str):
                # One placeholder peak; the model replaces its encoding by a learned vector
                columns["spectrum"].append(torch.tensor([0.0], dtype=torch.float32))
                columns["intensity"].append(torch.tensor([0.0], dtype=torch.float32))
                columns["multiplicity"].append(
                    torch.tensor([NMRDataset.MULTIPLICITY2IDX["<SYS>"]], dtype=torch.int64)
                )
                columns["coupling"].append(
                    torch.full((1, NMRDataset.MAX_COUPLINGS), NMRDataset.COUPLING_PAD, dtype=torch.float32)
                )
            elif spectrum is not None:
                for field in FIELDS:
                    columns[field].append(spectrum[field][:max_spectrum_len])
            else:
                columns["spectrum"].append(torch.tensor([SPECTRUM_PAD], dtype=torch.float32))
                columns["intensity"].append(torch.tensor([0.0], dtype=torch.float32))
                columns["multiplicity"].append(
                    torch.tensor([NMRDataset.MULTIPLICITY2IDX["<SYS>"]], dtype=torch.int64)
                )
                columns["coupling"].append(
                    torch.full((1, NMRDataset.MAX_COUPLINGS), NMRDataset.COUPLING_PAD, dtype=torch.float32)
                )
        for field in FIELDS:
            collated["C_NMR"][f"{spectrum_type}/{field}"] = torch.nn.utils.rnn.pad_sequence(
                sequences=columns[field],
                batch_first=True,
                padding_value=pad_values[field],
            )
        collated["C_NMR"][f"{spectrum_type}/empty"] = torch.tensor(empty, dtype=torch.bool)
    return collated


class CollateHeteroTrainWrapper:
    """Picklable collate wrapper for DataLoader multiprocessing."""

    def __init__(self, smiles_pad_id: int, spectrum_pad_token: float, max_smiles_len: int) -> None:
        self.smiles_pad_id = smiles_pad_id
        self.spectrum_pad_token = spectrum_pad_token
        self.max_smiles_len = max_smiles_len

    def __call__(self, batch: typing.List[typing.Dict[str, typing.Any]]) -> typing.Dict[str, typing.Any]:
        collated = collate_fn(
            batch=batch,
            smiles_pad_id=self.smiles_pad_id,
            spectrum_pad_token=self.spectrum_pad_token,
            max_smiles_len=self.max_smiles_len,
        )
        return add_hetero(collated, batch)


class CollateHeteroEvalWrapper:
    """Picklable collate wrapper for DataLoader multiprocessing."""

    def __init__(self, smiles_pad_id, spectrum_pad_token, max_smiles_len, max_spectrum_len):
        self.smiles_pad_id = smiles_pad_id
        self.spectrum_pad_token = spectrum_pad_token
        self.max_smiles_len = max_smiles_len
        self.max_spectrum_len = max_spectrum_len

    def __call__(self, batch):
        collated = collate_eval(
            batch=batch,
            smiles_pad_id=self.smiles_pad_id,
            spectrum_pad_token=self.spectrum_pad_token,
            max_smiles_len=self.max_smiles_len,
            max_spectrum_len=self.max_spectrum_len,
        )
        return add_hetero(collated, batch, self.max_spectrum_len)


class BalancedHeteroSampler(torch.utils.data.Sampler):
    """
    Every epoch: all molecules with a heteronuclear spectrum plus PLAIN_PER_HETERO times as many without one,
    drawn anew each epoch, shuffled together. With fixed=True the draw is the same every epoch.
    """

    def __init__(self, dataset: HeteroNMRDataset, seed: int, fixed: bool) -> None:
        flags = np.array([dataset.has_hetero(idx) for idx in range(len(dataset))])
        self.hetero_idx = np.flatnonzero(flags)
        self.plain_idx = np.flatnonzero(~flags)
        self.num_plain = min(PLAIN_PER_HETERO * len(self.hetero_idx), len(self.plain_idx))
        self.seed = seed
        self.fixed = fixed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.hetero_idx) + self.num_plain

    def __iter__(self) -> typing.Iterator[int]:
        rng = np.random.default_rng(self.seed if self.fixed else self.seed + self.epoch)
        self.epoch += 1
        plain = rng.choice(self.plain_idx, size=self.num_plain, replace=False)
        order = np.concatenate([self.hetero_idx, plain])
        rng.shuffle(order)
        return iter(order.tolist())


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

    collate = CollateHeteroTrainWrapper(
        smiles_pad_id=tokenizer.pad_id(),
        spectrum_pad_token=-1000.0,
        max_smiles_len=max_smiles_len,
    )

    loaders = []
    for name, path in (("train", train_path), ("val", val_path)):
        dataset = HeteroNMRDataset(
            data=load_split(path),
            solvent=solvent,
            tokenizer=tokenizer,
            smiles_bos_id=tokenizer.bos_id(),
            smiles_eos_id=tokenizer.eos_id(),
        )
        sampler = BalancedHeteroSampler(dataset, seed=0, fixed=name == "val")
        print(
            f"{name}: {len(sampler.hetero_idx)} molecules with heteronuclear spectra + "
            f"{sampler.num_plain} of {len(sampler.plain_idx)} without, per epoch"
        )
        loaders.append(
            torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                collate_fn=collate,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=num_workers > 0,
            )
        )
    return loaders[0], loaders[1]


def build_eval_dataset(
    tokenizer: spm.SentencePieceProcessor,
    multiplicity_codes_path: str,
    data_path: str,
    solvent: str,
) -> torch.utils.data.Dataset:
    multiplicity_codes = load_multiplicity_codes(multiplicity_codes_path)
    data = load_spectrum_data(data_path, solvent=solvent)
    processed = {
        k: [process_data(v_i, multiplicity_codes) for v_i in v]
        for k, v in tqdm(data.items(), desc="Processing spectra")
    }
    for value in processed.values():
        for cur in value:
            cur["solvent"] = solvent

    base_dataset = HeteroNMRDataset(
        data=processed,
        solvent=solvent,
        tokenizer=tokenizer,
        smiles_bos_id=tokenizer.bos_id(),
        smiles_eos_id=tokenizer.eos_id(),
    )
    print(f"Heteronuclear spectra: {'ON' if USE_HETERO else 'OFF'}; empty spectra for absent nuclei: {'ON' if EMPTY_HETERO else 'OFF'}")
    return HeteroEvaluationDataset(base_dataset, use_hetero=USE_HETERO, empty_hetero=EMPTY_HETERO)


training.build_dataloaders = build_dataloaders
training.build_model = build_model_from_base
for module in (evaluation, evaluation_top_n):
    module.build_model = build_model
    module.build_eval_dataset = build_eval_dataset
    module.CollateEvalWrapper = CollateHeteroEvalWrapper
