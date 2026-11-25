"""
Elemental analysis (CHN combustion) as an extra input to the model with heteronuclear spectra,
without touching the main code.

Importing this module points training and evaluation at EATransformer and at the datasets of
modalities/hetero/enable_hetero.py extended with a simulated elemental analysis: exact composition from
the SMILES plus the empirical noise of published analyses (src/blind/elemental_analysis.py,
data/ea/ea_noise.json). The analysis is added to the 13C batch dict as "EA/<field>" keys, so
the main loops run unchanged.

Training is a fine-tune of a trained model with heteronuclear spectra (INIT_CHECKPOINT) on the
same data and sampling as modalities/hetero/: heteronuclear dropout, empty spectra and PLAIN_PER_HETERO
molecules without heteronuclear spectra per molecule with one. A new analysis is drawn every
time a molecule is seen and is dropped with probability EA_DROP_PROB. In evaluation the
analysis of each example is fixed by a seed derived from its SMILES.

EA_NOISE = "flow" (default) draws Found values of C, H, N, S from the learned generator
(src/blind/ea_noise_flow.py, FLOW_PATH, trained by modalities/ea/train_noise_flow.py on organic compounds
and metal complexes): FLOW_SAMPLES analyses per molecule are generated once when the dataset is
built. Which elements the model sees follows what a user enters: C and H always, N and S by
MEASURE_PROB. A measured element absent from the molecule is given as its Found value (about
0.00); an element not measured has no token, so the model learns both "measured, zero" and
"unknown". EA_NOISE = "resampling" keeps the resampling model of src/blind/elemental_analysis.py
(NOISE_MODEL_PATH) with all reported elements.
"""
import hashlib
import sys
import typing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hetero"))

import numpy as np
import sentencepiece as spm
import torch
from tqdm.auto import tqdm

import enable_hetero
from blind import evaluation, evaluation_top_n, training
from blind.data_io import load_jsonl_data, load_spectrum_data
from blind.ea_noise_flow import TARGETS, ElementalAnalysisFlow
from blind.elemental_analysis import ATOMIC_MASS, ElementalAnalysisNoise, formula_counts
from blind.evaluation import collate_eval
from blind.models.ea_transformer import EA_ELEMENT2IDX, EA_PREFIX, EATransformer
from blind.models.hetero_transformer import HETERO_TYPES
from blind.training import NMRDataset, collate_fn, load_multiplicity_codes, process_data
from enable_hetero import BalancedHeteroSampler, HeteroEvaluationDataset, HeteroNMRDataset, add_hetero, pop_argument

INIT_CHECKPOINT: typing.Optional[str] = None  # set by train.py: a model with heteronuclear spectra
USE_EA = True  # evaluate*.py --no-ea sets False
# evaluate*.py --no-hetero / --empty-hetero set enable_hetero.USE_HETERO / EMPTY_HETERO
NOISE_MODEL_PATH = "data/ea/ea_noise.json"

EA_DROP_PROB = 0.3
EA_NOISE = "flow"
FLOW_PATH = "data/ea/noise_flow.pt"
FLOW_SAMPLES = 16
# Probability that an element is measured: (present in the molecule, absent from it)
MEASURE_PROB = {"C": (1.0, 1.0), "H": (1.0, 1.0), "N": (0.95, 0.5), "S": (0.5, 0.25)}
# Main transformer sizes. "base" is the model of the paper (all checkpoints so far); "large" is for
# training from scratch on all inputs (modalities/joint). A peak token of 1H/13C is the concatenation
# shift + intensity + multiplicity + 4 coupling slots, which must add up to d_model.
MODEL_SIZE = "base"
MODEL_SIZES = {
    "base": dict(spectrum_hidden_dim=104, intensity_hidden_dim=104, multiplicity_hidden_dim=48, coupling_slot_dim=8,
                 d_model=288, num_heads=8, num_layers=6, d_ff=512, dropout=0.2),
    "large": dict(spectrum_hidden_dim=136, intensity_hidden_dim=136, multiplicity_hidden_dim=64, coupling_slot_dim=12,
                  d_model=384, num_heads=8, num_layers=8, d_ff=1536, dropout=0.1),
}


def build_model(tokenizer: spm.SentencePieceProcessor, smiles_pad_token: int) -> EATransformer:
    return EATransformer(**model_kwargs(tokenizer, smiles_pad_token))


def model_kwargs(tokenizer: spm.SentencePieceProcessor, smiles_pad_token: int) -> typing.Dict[str, typing.Any]:
    """Architecture of the model with elemental analysis (also the base of modalities/ms), of size MODEL_SIZE."""
    kwargs = dict(
        ea_value_hidden_dim=40,
        ea_element_hidden_dim=16,
        ea_hidden_dim=96,  # 16 + 40 + 40
        ea_num_heads=4,
        ea_num_layers=2,
        ea_d_ff=192,
        # Same as enable_hetero.build_model
        hetero_types=HETERO_TYPES,
        hetero_spectrum_hidden_dim=56,
        hetero_multiplicity_hidden_dim=16,
        hetero_coupling_slot_dim=6,
        hetero_hidden_dim=96,
        hetero_num_heads=4,
        hetero_num_layers=2,
        hetero_d_ff=192,
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
    kwargs.update(MODEL_SIZES[MODEL_SIZE])
    return kwargs


def build_model_from_hetero(tokenizer: spm.SentencePieceProcessor, smiles_pad_token: int) -> EATransformer:
    if INIT_CHECKPOINT is None:
        raise ValueError("--init-checkpoint (trained model with heteronuclear spectra) is required")
    model = build_model(tokenizer, smiles_pad_token)
    model.init_from_hetero(torch.load(INIT_CHECKPOINT, map_location="cpu"))
    print(f"Initialized from {INIT_CHECKPOINT}; elemental analysis encoder and cross-attentions are new")
    return model


def tensorize(found: typing.Dict[str, float]) -> typing.Dict[str, torch.Tensor]:
    """{element: Found %} -> element ids, mass % and molar % among the reported elements."""
    elements = sorted(found)
    moles = np.array([found[e] / ATOMIC_MASS[e] for e in elements])
    molar = 100.0 * moles / moles.sum() if moles.sum() > 0 else moles
    return {
        "element": torch.tensor([EA_ELEMENT2IDX.get(e, EA_ELEMENT2IDX["<other>"]) for e in elements], dtype=torch.int64),
        "mass": torch.tensor([found[e] for e in elements], dtype=torch.float32),
        "molar": torch.tensor(molar, dtype=torch.float32),
    }


class EANMRDataset(HeteroNMRDataset):
    """HeteroNMRDataset + a freshly simulated elemental analysis (or None) for every item."""

    def __init__(
        self,
        data: typing.Dict[str, typing.List[typing.Dict[str, typing.Any]]],
        solvent: str,
        tokenizer: spm.SentencePieceProcessor,
        smiles_bos_id: int,
        smiles_eos_id: int,
    ) -> None:
        super().__init__(data, solvent, tokenizer, smiles_bos_id, smiles_eos_id)
        self.formulas = {smiles: formula_counts(smiles) for smiles in tqdm(data.keys(), desc="Formulas")}
        if EA_NOISE == "flow":
            smiles_list = list(data.keys())
            print(f"Generating {FLOW_SAMPLES} elemental analyses per molecule with {FLOW_PATH}")
            self.flow_found = dict(zip(smiles_list, ElementalAnalysisFlow(FLOW_PATH).sample(smiles_list, FLOW_SAMPLES)))
        else:
            self.noise = ElementalAnalysisNoise(NOISE_MODEL_PATH)
        self._rng: typing.Optional[np.random.Generator] = None

    @property
    def rng(self) -> np.random.Generator:
        # Seeded per DataLoader worker (torch gives each worker its own seed), so workers draw different noise
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % 2**32)
        return self._rng

    def simulate(self, smiles: str, rng: np.random.Generator) -> typing.Optional[typing.Dict[str, torch.Tensor]]:
        counts = self.formulas[smiles]
        if counts is None:
            return None
        if EA_NOISE != "flow":
            found = self.noise.sample(counts, rng)
            return tensorize(found) if found else None
        values = self.flow_found[smiles][rng.integers(FLOW_SAMPLES)]
        if np.isnan(values).any():
            return None
        found = {
            e: float(values[j])
            for j, e in enumerate(TARGETS)
            if rng.random() < MEASURE_PROB[e][0 if counts.get(e) else 1]
        }
        return tensorize(found) if found else None

    def __getitem__(self, idx: int) -> typing.Dict[str, typing.Any]:
        item = super().__getitem__(idx)
        smiles = self.idx2smiles[idx]
        item["ea"] = None if self.rng.random() < EA_DROP_PROB else self.simulate(smiles, self.rng)
        return item


class EAEvaluationDataset(HeteroEvaluationDataset):
    """Same examples as HeteroEvaluationDataset; pair k of a molecule gets an analysis seeded by (SMILES, k)."""

    def __init__(self, base_dataset: EANMRDataset, use_hetero: bool, empty_hetero: bool, use_ea: bool) -> None:
        super().__init__(base_dataset, use_hetero=use_hetero, empty_hetero=empty_hetero)
        combo_idx: typing.Dict[str, int] = {}
        for example, smiles in zip(self.examples, self.example_smiles):
            k = combo_idx.get(smiles, 0)
            combo_idx[smiles] = k + 1
            seed = int(hashlib.md5(f"{smiles}|{k}".encode()).hexdigest()[:8], 16)
            example["ea"] = base_dataset.simulate(smiles, np.random.default_rng(seed)) if use_ea else None


def add_ea(
    collated: typing.Dict[str, typing.Any],
    batch: typing.List[typing.Dict[str, typing.Any]],
) -> typing.Dict[str, typing.Any]:
    pad_values = {"element": EA_ELEMENT2IDX["<PAD>"], "mass": 0.0, "molar": 0.0}
    for field, pad_value in pad_values.items():
        sequences = [
            cur["ea"][field] if cur["ea"] is not None
            else torch.tensor([pad_value], dtype=torch.int64 if field == "element" else torch.float32)
            for cur in batch
        ]
        collated["C_NMR"][f"{EA_PREFIX}{field}"] = torch.nn.utils.rnn.pad_sequence(
            sequences=sequences,
            batch_first=True,
            padding_value=pad_value,
        )
    return collated


class CollateEATrainWrapper:
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
        return add_ea(add_hetero(collated, batch), batch)


class CollateEAEvalWrapper:
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
        return add_ea(add_hetero(collated, batch, self.max_spectrum_len), batch)


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

    collate = CollateEATrainWrapper(
        smiles_pad_id=tokenizer.pad_id(),
        spectrum_pad_token=-1000.0,
        max_smiles_len=max_smiles_len,
    )

    loaders = []
    for name, path in (("train", train_path), ("val", val_path)):
        dataset = EANMRDataset(
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

    base_dataset = EANMRDataset(
        data=processed,
        solvent=solvent,
        tokenizer=tokenizer,
        smiles_bos_id=tokenizer.bos_id(),
        smiles_eos_id=tokenizer.eos_id(),
    )
    print(
        f"Heteronuclear spectra: {'ON' if enable_hetero.USE_HETERO else 'OFF'}; "
        f"empty spectra for absent nuclei: {'ON' if enable_hetero.EMPTY_HETERO else 'OFF'}; "
        f"elemental analysis: {'ON' if USE_EA else 'OFF'}"
    )
    return EAEvaluationDataset(
        base_dataset, use_hetero=enable_hetero.USE_HETERO, empty_hetero=enable_hetero.EMPTY_HETERO, use_ea=USE_EA
    )


training.build_dataloaders = build_dataloaders
training.build_model = build_model_from_hetero
for module in (evaluation, evaluation_top_n):
    module.build_model = build_model
    module.build_eval_dataset = build_eval_dataset
    module.CollateEvalWrapper = CollateEAEvalWrapper
