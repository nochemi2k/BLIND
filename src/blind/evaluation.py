import argparse
import os
import json
import typing

import torch
import sentencepiece as spm
from tqdm.auto import tqdm
from rdkit import Chem, RDLogger

from .models.transformer import Transformer
from .training import NMRDataset, load_multiplicity_codes, process_data, move_to_device
from .data_io import load_spectrum_data
from .device import resolve_device, default_num_workers, use_pin_memory, add_device_arg


RDLogger.DisableLog("rdApp.*")


class EvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset: NMRDataset) -> None:
        self.examples: typing.List[typing.Dict[str, typing.Any]] = []
        self._build_examples(base_dataset)

    def _build_examples(self, base_dataset: NMRDataset) -> None:
        for smiles in base_dataset.smiles2idx.keys():
            carbon_list = base_dataset.carbon_spectra[smiles]
            hydrogen_list = base_dataset.hydrogen_spectra[smiles]
            max_len = max(len(carbon_list), len(hydrogen_list))
            if max_len == 0:
                continue
            smiles_tensor = base_dataset.encoded_smiles[smiles]
            for combo_idx in range(max_len):
                carbon_entry = (
                    carbon_list[combo_idx] if combo_idx < len(carbon_list) else None
                )
                hydrogen_entry = (
                    hydrogen_list[combo_idx] if combo_idx < len(hydrogen_list) else None
                )
                if carbon_entry is None and hydrogen_entry is None:
                    continue
                self.examples.append(
                    {
                        "smiles": smiles_tensor,
                        "carbon_spectrum": carbon_entry,
                        "hydrogen_spectrum": hydrogen_entry,
                    }
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> typing.Dict[str, typing.Any]:
        return self.examples[idx]


def collate_eval(
    batch: typing.List[typing.Dict[str, typing.Any]],
    smiles_pad_id: int,
    spectrum_pad_token: float,
    max_smiles_len: int,
    max_spectrum_len: int,
) -> typing.Dict[str, typing.Any]:
    smiles = []
    carbon_spectrum = []
    carbon_intensity = []
    carbon_multiplicity = []
    hydrogen_spectrum = []
    hydrogen_intensity = []
    hydrogen_multiplicity = []

    for cur in batch:
        smiles.append(cur["smiles"][:max_smiles_len])
        if cur["carbon_spectrum"] is not None:
            carbon_spectrum.append(
                cur["carbon_spectrum"]["spectrum"][:max_spectrum_len]
            )
            carbon_intensity.append(
                cur["carbon_spectrum"]["intensity"][:max_spectrum_len]
            )
            carbon_multiplicity.append(
                cur["carbon_spectrum"]["multiplicity"][:max_spectrum_len]
            )
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

        if cur["hydrogen_spectrum"] is not None:
            hydrogen_spectrum.append(
                cur["hydrogen_spectrum"]["spectrum"][:max_spectrum_len]
            )
            hydrogen_intensity.append(
                cur["hydrogen_spectrum"]["intensity"][:max_spectrum_len]
            )
            hydrogen_multiplicity.append(
                cur["hydrogen_spectrum"]["multiplicity"][:max_spectrum_len]
            )
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
        },
    }


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

    smiles_bos_id = tokenizer.bos_id()
    smiles_eos_id = tokenizer.eos_id()

    base_dataset = NMRDataset(
        data=processed,
        solvent=solvent,
        tokenizer=tokenizer,
        smiles_bos_id=smiles_bos_id,
        smiles_eos_id=smiles_eos_id,
    )
    return EvaluationDataset(base_dataset)


@torch.inference_mode
def generate_batch(
    model: Transformer,
    src_c_nmr: typing.Dict[str, torch.Tensor],
    src_h_nmr: typing.Dict[str, torch.Tensor],
    tokenizer: spm.SentencePieceProcessor,
    max_tokens: int = 200,
    device: torch.device = torch.device("cpu"),
) -> typing.List[str]:
    model.eval()

    batch_size = src_c_nmr["spectrum"].shape[0]
    idxs = torch.full(
        (batch_size, 1),
        fill_value=tokenizer.bos_id(),
        dtype=torch.int64,
        device=device,
    )
    system_tokens = {
        tokenizer.bos_id(),
        tokenizer.eos_id(),
        tokenizer.pad_id(),
        tokenizer.unk_id(),
    }

    src_c_nmr = move_to_device(src_c_nmr, device)
    src_h_nmr = move_to_device(src_h_nmr, device)

    while idxs.shape[-1] < max_tokens:
        logits = model(
            src_c_nmr=src_c_nmr,
            src_h_nmr=src_h_nmr,
            tgt=idxs,
        )
        new_idxs = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idxs = torch.cat([idxs, new_idxs], dim=-1)

        if torch.all(torch.any(idxs == tokenizer.eos_id(), dim=-1)):
            break

    idxs_list = []
    for seq in idxs.cpu().numpy().tolist():
        filtered = [tok for tok in seq if tok not in system_tokens]
        idxs_list.append(filtered)

    smiles_list = tokenizer.decode(idxs_list)
    return smiles_list


class CollateEvalWrapper:
    """Wrapper class to make collate function picklable for Windows multiprocessing."""
    def __init__(self, smiles_pad_id, spectrum_pad_token, max_smiles_len, max_spectrum_len):
        self.smiles_pad_id = smiles_pad_id
        self.spectrum_pad_token = spectrum_pad_token
        self.max_smiles_len = max_smiles_len
        self.max_spectrum_len = max_spectrum_len
    
    def __call__(self, batch):
        return collate_eval(
            batch=batch,
            smiles_pad_id=self.smiles_pad_id,
            spectrum_pad_token=self.spectrum_pad_token,
            max_smiles_len=self.max_smiles_len,
            max_spectrum_len=self.max_spectrum_len,
        )


def evaluate(
    args: argparse.Namespace,
) -> None:
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(args.tokenizer_path)

    spectrum_pad_token = -1000.0

    collate = CollateEvalWrapper(
        smiles_pad_id=tokenizer.pad_id(),
        spectrum_pad_token=spectrum_pad_token,
        max_smiles_len=args.max_smiles_len,
        max_spectrum_len=args.max_spectrum_len,
    )

    dataset = build_eval_dataset(
        tokenizer=tokenizer,
        multiplicity_codes_path=args.multiplicity_codes_path,
        data_path=args.data_path,
        solvent=args.solvent,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory(device),
        persistent_workers=args.num_workers > 0,
    )

    model = Transformer(
        multiplicity_vocab_size=len(NMRDataset.MULTIPLICITY2IDX),
        multiplicity_hidden_dim=48,
        spectrum_d_ff=256,
        spectrum_hidden_dim=104,
        intensity_d_ff=256,
        intensity_hidden_dim=104,
        tgt_vocab_size=tokenizer.vocab_size(),
        d_model=256,
        num_heads=8,
        num_layers=6,
        d_ff=512,
        max_seq_length=400,
        dropout=0.2,
        smiles_pad_token=tokenizer.pad_id(),
    ).to(device)

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    results: typing.List[typing.Dict[str, typing.Any]] = []
    only_c = only_c_valid = only_c_correct = 0
    only_h = only_h_valid = only_h_correct = 0
    both = both_valid = both_correct = 0

    for batch in tqdm(loader, desc="Evaluating"):
        smiles_tokens = batch["smiles"]
        true_smiles = [
            tokenizer.decode(smiles_tokens[i].cpu().numpy().tolist())
            for i in range(smiles_tokens.shape[0])
        ]

        generated_smiles = generate_batch(
            model=model,
            src_c_nmr=batch["C_NMR"],
            src_h_nmr=batch["H_NMR"],
            tokenizer=tokenizer,
            max_tokens=args.max_tokens,
            device=device,
        )

        valid = [(Chem.MolFromSmiles(cur) is not None) for cur in generated_smiles]

        missed_c_list = torch.all(
            batch["C_NMR"]["spectrum"] == spectrum_pad_token, dim=-1
        ).cpu()
        missed_h_list = torch.all(
            batch["H_NMR"]["spectrum"] == spectrum_pad_token, dim=-1
        ).cpu()

        carbon_lengths = (
            batch["C_NMR"]["spectrum"] != spectrum_pad_token
        ).sum(dim=-1).cpu()
        hydrogen_lengths = (
            batch["H_NMR"]["spectrum"] != spectrum_pad_token
        ).sum(dim=-1).cpu()

        for i in range(len(generated_smiles)):
            missed_c = bool(missed_c_list[i].item())
            missed_h = bool(missed_h_list[i].item())

            if missed_h and not missed_c:
                only_c += 1
                only_c_valid += int(valid[i])
                only_c_correct += int(generated_smiles[i] == true_smiles[i])
            if not missed_h and missed_c:
                only_h += 1
                only_h_valid += int(valid[i])
                only_h_correct += int(generated_smiles[i] == true_smiles[i])
            if not missed_c and not missed_h:
                both += 1
                both_valid += int(valid[i])
                both_correct += int(generated_smiles[i] == true_smiles[i])

            carbon_entry = None
            hydrogen_entry = None
            if not missed_c:
                cl = int(carbon_lengths[i].item())
                carbon_entry = {
                    "spectrum": batch["C_NMR"]["spectrum"][i, :cl].cpu().tolist(),
                    "intensity": batch["C_NMR"]["intensity"][i, :cl].cpu().tolist(),
                    "multiplicity": batch["C_NMR"]["multiplicity"][i, :cl]
                    .cpu()
                    .tolist(),
                }
            if not missed_h:
                hl = int(hydrogen_lengths[i].item())
                hydrogen_entry = {
                    "spectrum": batch["H_NMR"]["spectrum"][i, :hl].cpu().tolist(),
                    "intensity": batch["H_NMR"]["intensity"][i, :hl].cpu().tolist(),
                    "multiplicity": batch["H_NMR"]["multiplicity"][i, :hl]
                    .cpu()
                    .tolist(),
                }

            results.append(
                {
                    "smiles": true_smiles[i],
                    "true_smiles": true_smiles[i],
                    "prediction": generated_smiles[i],
                    "has_carbon": not missed_c,
                    "has_hydrogen": not missed_h,
                    "carbon": carbon_entry,
                    "hydrogen": hydrogen_entry,
                    "valid": bool(valid[i]),
                    "exact_match": generated_smiles[i] == true_smiles[i],
                }
            )

    count = only_c + only_h + both
    count_valid = only_c_valid + only_h_valid + both_valid
    count_correct = only_c_correct + only_h_correct + both_correct

    print(
        f"Only C: {only_c}; Valid Ratio: {only_c_valid / max(only_c, 1):.4f}; "
        f"Correct Ratio: {only_c_correct / max(only_c, 1):.4f}"
    )
    print(
        f"Only H: {only_h}; Valid Ratio: {only_h_valid / max(only_h, 1):.4f}; "
        f"Correct Ratio: {only_h_correct / max(only_h, 1):.4f}"
    )
    print(
        f"Both: {both}; Valid Ratio: {both_valid / max(both, 1):.4f}; "
        f"Correct Ratio: {both_correct / max(both, 1):.4f}"
    )
    print(
        f"Overall: {count}; Valid Ratio: {count_valid / max(count, 1):.4f}; "
        f"Correct Ratio: {count_correct / max(count, 1):.4f}"
    )

    if args.output_json is not None:
        summary = {
            "only_c": {
                "count": only_c,
                "valid_ratio": (only_c_valid / max(only_c, 1)),
                "correct_ratio": (only_c_correct / max(only_c, 1)),
            },
            "only_h": {
                "count": only_h,
                "valid_ratio": (only_h_valid / max(only_h, 1)),
                "correct_ratio": (only_h_correct / max(only_h, 1)),
            },
            "both": {
                "count": both,
                "valid_ratio": (both_valid / max(both, 1)),
                "correct_ratio": (both_correct / max(both, 1)),
            },
            "overall": {
                "count": count,
                "valid_ratio": (count_valid / max(count, 1)),
                "correct_ratio": (count_correct / max(count, 1)),
            },
        }
        with open(args.output_json, "w") as fp:
            json.dump(
                {
                    "predictions": results,
                    "summary": summary,
                },
                fp,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate bimodal Transformer on NMR data."
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
        default="src/blind/data/multiplicity_codes.json",
        help="Path to multiplicity codes JSON.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/samples/test.jsonl",
        help="Path to evaluation data (JSONL or external validation CSV).",
    )
    parser.add_argument(
        "--solvent",
        type=str,
        default="CDCl_3",
        help="Solvent name to filter spectra.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt).",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=default_num_workers(),
        help="Number of DataLoader workers (0 recommended on Windows).",
    )
    add_device_arg(parser)
    parser.add_argument("--max-tokens", type=int, default=275)
    parser.add_argument(
        "--max-smiles-len",
        type=int,
        default=275,
        help="Truncate tokenized SMILES to this length (including BOS/EOS).",
    )
    parser.add_argument(
        "--max-spectrum-len",
        type=int,
        default=400,
        help="Truncate NMR spectra length to this many points to match model PE.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save predictions as JSON.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())


