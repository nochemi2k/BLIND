import argparse
import os
import random
import json
import typing
from collections import defaultdict
import time

import torch
import numpy as np
import sentencepiece as spm
from tqdm.auto import tqdm

from .models.transformer import Transformer
from .data_io import load_jsonl_data
from .device import (
    resolve_device,
    default_num_workers,
    use_pin_memory,
    supports_cuda_amp,
    autocast_context,
    create_grad_scaler,
    add_device_arg,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def load_multiplicity_codes(path: str) -> typing.DefaultDict[str, str]:
    with open(path, "r") as f:
        multiplicity_codes = json.load(f)
    multiplicity_codes = defaultdict(lambda: "m", multiplicity_codes)
    return multiplicity_codes


CARBON_MULTIPLICITIES = {"s", "d", "t", "q", "dd", "m"}


def process_data(x: dict, multiplicity_codes: typing.Mapping[str, str]) -> dict:
    import copy

    x_copy = copy.deepcopy(x)
    x_copy["spectrum"] = sorted(x_copy["spectrum"], key=lambda s: s["signal"])
    for spectrum in x_copy["spectrum"]:
        spectrum["multiplicity"] = multiplicity_codes[spectrum["multiplicity"]]
        if x_copy["spectrum_type"] == "C_NMR":
            if spectrum["multiplicity"] not in CARBON_MULTIPLICITIES:
                spectrum["multiplicity"] = "m"
    return x_copy


class NMRDataset(torch.utils.data.Dataset):
    MULTIPLICITY2IDX = {
        "<SYS>": 0,
        "s": 1,
        "d": 2,
        "t": 3,
        "q": 4,
        "p": 5,
        "hept": 6,
        "dd": 7,
        "dt": 8,
        "td": 9,
        "tt": 10,
        "dq": 11,
        "qd": 12,
        "ddd": 13,
        "ddt": 14,
        "dddd": 15,
        "brs": 16,
        "br": 17,
        "m": 18,
    }
    MAX_COUPLINGS = 4
    COUPLING_PAD = -1.0

    @classmethod
    def _coupling_tensor(cls, peaks: typing.List[typing.Dict[str, typing.Any]]) -> torch.Tensor:
        """[L, MAX_COUPLINGS] J values in Hz, largest first, right-padded with COUPLING_PAD."""
        couplings = torch.full((len(peaks), cls.MAX_COUPLINGS), cls.COUPLING_PAD, dtype=torch.float32)
        for i, peak in enumerate(peaks):
            values = sorted(peak.get("couplings") or [], reverse=True)[: cls.MAX_COUPLINGS]
            couplings[i, : len(values)] = torch.tensor(values, dtype=torch.float32)
        return couplings

    def __init__(
        self,
        data: typing.Dict[str, typing.List[typing.Dict[str, typing.Any]]],
        solvent: str,
        tokenizer: spm.SentencePieceProcessor,
        smiles_bos_id: int,
        smiles_eos_id: int,
    ) -> None:
        self.smiles2idx = {smiles: idx for idx, smiles in enumerate(data.keys())}
        self.idx2smiles = {v: k for k, v in self.smiles2idx.items()}

        self.carbon_spectra = {
            smiles: [
                {
                    "spectrum": torch.tensor(
                        [cur["signal"] for cur in spectrum["spectrum"]],
                        dtype=torch.float32,
                    ),
                    "intensity": torch.tensor(
                        [cur["intensity"] for cur in spectrum["spectrum"]],
                        dtype=torch.float32,
                    ),
                    "multiplicity": torch.tensor(
                        [
                            self.MULTIPLICITY2IDX[cur["multiplicity"]]
                            for cur in spectrum["spectrum"]
                        ],
                        dtype=torch.int64,
                    ),
                    "coupling": self._coupling_tensor(spectrum["spectrum"]),
                }
                for spectrum in spectra
                if spectrum["spectrum_type"] == "C_NMR"
                and spectrum["solvent"] == solvent
            ]
            for smiles, spectra in data.items()
        }

        self.hydrogen_spectra = {
            smiles: [
                {
                    "spectrum": torch.tensor(
                        [cur["signal"] for cur in spectrum["spectrum"]],
                        dtype=torch.float32,
                    ),
                    "intensity": torch.tensor(
                        [cur["intensity"] for cur in spectrum["spectrum"]],
                        dtype=torch.float32,
                    ),
                    "multiplicity": torch.tensor(
                        [
                            self.MULTIPLICITY2IDX[cur["multiplicity"]]
                            for cur in spectrum["spectrum"]
                        ],
                        dtype=torch.int64,
                    ),
                    "coupling": self._coupling_tensor(spectrum["spectrum"]),
                }
                for spectrum in spectra
                if spectrum["spectrum_type"] == "H_NMR"
                and spectrum["solvent"] == solvent
            ]
            for smiles, spectra in data.items()
        }

        self.encoded_smiles = {
            smiles: torch.tensor(
                [smiles_bos_id] + tokenizer.encode(smiles) + [smiles_eos_id],
                dtype=torch.int64,
            )
            for smiles in data.keys()
        }

    def __len__(self) -> int:
        return len(self.smiles2idx)

    @staticmethod
    def _select_spectrum(
        values: typing.List[typing.Dict[str, torch.Tensor]]
    ) -> typing.Optional[typing.Dict[str, torch.Tensor]]:
        size = len(values)
        if size == 0:
            return None
        return values[np.random.randint(low=0, high=size)]

    def __getitem__(self, idx: int) -> typing.Dict[str, typing.Any]:
        smiles = self.idx2smiles[idx]

        carbon_spectrum = self._select_spectrum(self.carbon_spectra[smiles])
        hydrogen_spectrum = self._select_spectrum(self.hydrogen_spectra[smiles])

        if carbon_spectrum is not None and hydrogen_spectrum is not None:
            if np.random.rand() > 0.8:
                if np.random.rand() > 0.5:
                    carbon_spectrum = None
                else:
                    hydrogen_spectrum = None

        return {
            "smiles": self.encoded_smiles[smiles],
            "carbon_spectrum": carbon_spectrum,
            "hydrogen_spectrum": hydrogen_spectrum,
        }


def collate_fn(
    batch: typing.List[typing.Dict[str, typing.Any]],
    smiles_pad_id: int,
    spectrum_pad_token: float,
    max_smiles_len: int,
) -> typing.Dict[str, typing.Any]:
    smiles = []
    carbon_spectrum = []
    carbon_intensity = []
    carbon_multiplicity = []
    carbon_coupling = []
    hydrogen_spectrum = []
    hydrogen_intensity = []
    hydrogen_multiplicity = []
    hydrogen_coupling = []

    for cur in batch:
        smiles.append(cur["smiles"][:max_smiles_len])
        if cur["carbon_spectrum"] is not None:
            carbon_spectrum.append(cur["carbon_spectrum"]["spectrum"])
            carbon_intensity.append(cur["carbon_spectrum"]["intensity"])
            carbon_multiplicity.append(cur["carbon_spectrum"]["multiplicity"])
            carbon_coupling.append(cur["carbon_spectrum"]["coupling"])
        else:
            carbon_spectrum.append(
                torch.tensor([spectrum_pad_token], dtype=torch.float32)
            )
            carbon_intensity.append(torch.tensor([0.0], dtype=torch.float32))
            carbon_multiplicity.append(
                torch.tensor(
                    [NMRDataset.MULTIPLICITY2IDX["<SYS>"]], dtype=torch.int64
                )
            )
            carbon_coupling.append(
                torch.full((1, NMRDataset.MAX_COUPLINGS), NMRDataset.COUPLING_PAD, dtype=torch.float32)
            )

        if cur["hydrogen_spectrum"] is not None:
            hydrogen_spectrum.append(cur["hydrogen_spectrum"]["spectrum"])
            hydrogen_intensity.append(cur["hydrogen_spectrum"]["intensity"])
            hydrogen_multiplicity.append(cur["hydrogen_spectrum"]["multiplicity"])
            hydrogen_coupling.append(cur["hydrogen_spectrum"]["coupling"])
        else:
            hydrogen_spectrum.append(
                torch.tensor([spectrum_pad_token], dtype=torch.float32)
            )
            hydrogen_intensity.append(torch.tensor([0.0], dtype=torch.float32))
            hydrogen_multiplicity.append(
                torch.tensor(
                    [NMRDataset.MULTIPLICITY2IDX["<SYS>"]], dtype=torch.int64
                )
            )
            hydrogen_coupling.append(
                torch.full((1, NMRDataset.MAX_COUPLINGS), NMRDataset.COUPLING_PAD, dtype=torch.float32)
            )

    return {
        "smiles": torch.nn.utils.rnn.pad_sequence(
            sequences=smiles,
            batch_first=True,
            padding_value=smiles_pad_id,
        ),
        "C_NMR": {
            "spectrum": torch.nn.utils.rnn.pad_sequence(
                sequences=carbon_spectrum,
                batch_first=True,
                padding_value=spectrum_pad_token,
            ),
            "intensity": torch.nn.utils.rnn.pad_sequence(
                sequences=carbon_intensity,
                batch_first=True,
                padding_value=0.0,
            ),
            "multiplicity": torch.nn.utils.rnn.pad_sequence(
                sequences=carbon_multiplicity,
                batch_first=True,
                padding_value=NMRDataset.MULTIPLICITY2IDX["<SYS>"],
            ),
            "coupling": torch.nn.utils.rnn.pad_sequence(
                sequences=carbon_coupling,
                batch_first=True,
                padding_value=NMRDataset.COUPLING_PAD,
            ),
        },
        "H_NMR": {
            "spectrum": torch.nn.utils.rnn.pad_sequence(
                sequences=hydrogen_spectrum,
                batch_first=True,
                padding_value=spectrum_pad_token,
            ),
            "intensity": torch.nn.utils.rnn.pad_sequence(
                sequences=hydrogen_intensity,
                batch_first=True,
                padding_value=0.0,
            ),
            "multiplicity": torch.nn.utils.rnn.pad_sequence(
                sequences=hydrogen_multiplicity,
                batch_first=True,
                padding_value=NMRDataset.MULTIPLICITY2IDX["<SYS>"],
            ),
            "coupling": torch.nn.utils.rnn.pad_sequence(
                sequences=hydrogen_coupling,
                batch_first=True,
                padding_value=NMRDataset.COUPLING_PAD,
            ),
        },
    }


def move_to_device(data: typing.Any, device: torch.device) -> typing.Any:
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=device.type == "cuda")
    if isinstance(data, dict):
        return {k: move_to_device(v, device=device) for k, v in data.items()}
    return data


class CollateTrainWrapper:
    """Picklable collate wrapper for DataLoader multiprocessing."""

    def __init__(
        self,
        smiles_pad_id: int,
        spectrum_pad_token: float,
        max_smiles_len: int,
    ) -> None:
        self.smiles_pad_id = smiles_pad_id
        self.spectrum_pad_token = spectrum_pad_token
        self.max_smiles_len = max_smiles_len

    def __call__(self, batch: typing.List[typing.Dict[str, typing.Any]]) -> typing.Dict[str, typing.Any]:
        return collate_fn(
            batch=batch,
            smiles_pad_id=self.smiles_pad_id,
            spectrum_pad_token=self.spectrum_pad_token,
            max_smiles_len=self.max_smiles_len,
        )


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

    processed_train = load_split(train_path)
    processed_val = load_split(val_path)

    smiles_bos_id = tokenizer.bos_id()
    smiles_eos_id = tokenizer.eos_id()

    train_set = NMRDataset(
        data=processed_train,
        solvent=solvent,
        tokenizer=tokenizer,
        smiles_bos_id=smiles_bos_id,
        smiles_eos_id=smiles_eos_id,
    )

    val_set = NMRDataset(
        data=processed_val,
        solvent=solvent,
        tokenizer=tokenizer,
        smiles_bos_id=smiles_bos_id,
        smiles_eos_id=smiles_eos_id,
    )

    spectrum_pad_token = -1000.0

    collate = CollateTrainWrapper(
        smiles_pad_id=tokenizer.pad_id(),
        spectrum_pad_token=spectrum_pad_token,
        max_smiles_len=max_smiles_len,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader


def build_model(tokenizer: spm.SentencePieceProcessor, smiles_pad_token: int) -> Transformer:
    model = Transformer(
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
        d_model=288,  # 104 + 104 + 48 + 4 * 8
        num_heads=8,
        num_layers=6,
        d_ff=512,
        max_seq_length=400,
        dropout=0.2,
        smiles_pad_token=smiles_pad_token,
    )
    return model


def train(
    args: argparse.Namespace,
) -> None:
    seed_everything(args.seed)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(args.tokenizer_path)

    train_loader, val_loader = build_dataloaders(
        tokenizer=tokenizer,
        multiplicity_codes_path=args.multiplicity_codes_path,
        train_path=args.train_path,
        val_path=args.val_path,
        solvent=args.solvent,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_smiles_len=args.max_smiles_len,
        pin_memory=use_pin_memory(device),
    )

    model = build_model(tokenizer, smiles_pad_token=tokenizer.pad_id()).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    use_amp = args.mixed_precision and supports_cuda_amp(device)
    if args.mixed_precision and not use_amp:
        print("Mixed precision is only supported on CUDA; running in full precision.")

    scaler = create_grad_scaler(device, enabled=use_amp)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_loss_log: typing.List[float] = []
    val_loss_log: typing.List[float] = []
    lr_log: typing.List[float] = []
    epoch_time_log: typing.List[float] = []

    for epoch in range(args.epochs):
        epoch_start = time.time()

        model.train()
        cur_train_loss: typing.List[float] = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
        for batch in pbar:
            batch = move_to_device(batch, device=device)

            with autocast_context(device, enabled=use_amp):
                logits = model(
                    src_c_nmr=batch["C_NMR"],
                    src_h_nmr=batch["H_NMR"],
                    tgt=batch["smiles"][:, :-1],
                )
                loss = criterion(
                    logits.reshape(-1, tokenizer.vocab_size()),
                    batch["smiles"][:, 1:].reshape(-1),
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_val = loss.detach().item()
            cur_train_loss.append(loss_val)
            pbar.set_postfix({"loss": loss_val})

        model.eval()
        cur_val_loss: typing.List[float] = []
        with torch.inference_mode():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch} [val]")
            for batch in pbar_val:
                batch = move_to_device(batch, device=device)
                with autocast_context(device, enabled=use_amp):
                    logits = model(
                        src_c_nmr=batch["C_NMR"],
                        src_h_nmr=batch["H_NMR"],
                        tgt=batch["smiles"][:, :-1],
                    )
                    loss = criterion(
                        logits.reshape(-1, tokenizer.vocab_size()),
                        batch["smiles"][:, 1:].reshape(-1),
                    )
                loss_val = loss.detach().item()
                cur_val_loss.append(loss_val)
                pbar_val.set_postfix({"loss": loss_val})

        epoch_time = time.time() - epoch_start
        epoch_train_loss = float(np.mean(cur_train_loss))
        epoch_val_loss = float(np.mean(cur_val_loss))

        train_loss_log.append(epoch_train_loss)
        val_loss_log.append(epoch_val_loss)
        epoch_time_log.append(epoch_time)

        scheduler.step(epoch_val_loss)
        lr_log.append(float(optimizer.param_groups[0]["lr"]))

        print(
            f"Epoch {epoch}: train_loss={epoch_train_loss:.4f} "
            f"val_loss={epoch_val_loss:.4f} lr={lr_log[-1]:.6f} "
            f"time={epoch_time:.1f}s"
        )

        log_obj = {
            "train_loss": train_loss_log,
            "val_loss": val_loss_log,
            "lr_log": lr_log,
            "epoch_time_sec": epoch_time_log,
        }
        with open(os.path.join(args.checkpoint_dir, f"{args.exp_prefix}_logs.json"), "w") as f:
            json.dump(log_obj, f)

        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            ckpt_path = os.path.join(
                args.checkpoint_dir, f"{args.exp_prefix}_epoch_{epoch}.pt"
            )
            torch.save(model.state_dict(), ckpt_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train bimodal Transformer on NMR data (optimized)."
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="models/smiles_tokenizer.model",
        help="Path to SentencePiece tokenizer model.",
    )
    parser.add_argument(
        "--multiplicity-codes-path",
        type=str,
        default="src/blind/data/multiplicity_codes_nmrexp.json",
        help="Path to multiplicity codes JSON.",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default="data/samples/train.jsonl",
        help="Path to training jsonl file.",
    )
    parser.add_argument(
        "--val-path",
        type=str,
        default="data/samples/val.jsonl",
        help="Path to validation jsonl file.",
    )
    parser.add_argument(
        "--solvent",
        type=str,
        default="CDCl_3",
        help="Solvent name to filter spectra.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--max-smiles-len",
        type=int,
        default=275,
        help="Truncate tokenized SMILES to this length (including BOS/EOS).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=default_num_workers(),
        help="Number of DataLoader workers (0 recommended on Windows).",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision training on CUDA.",
    )
    add_device_arg(parser)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint dir."
    )
    parser.add_argument(
        "--exp-prefix",
        type=str,
        default="blind_256",
        help="Prefix for checkpoints/logs.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())


