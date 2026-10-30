import argparse
import os
import json
import typing
from pathlib import Path

import torch
import torch.nn.functional as F
import sentencepiece as spm
from tqdm.auto import tqdm
from rdkit import Chem, RDLogger

from .models.transformer import Transformer
from .training import NMRDataset, build_model, load_multiplicity_codes, process_data, move_to_device
from .evaluation import (
    EvaluationDataset,
    collate_eval,
    build_eval_dataset,
    CollateEvalWrapper,
)
from .device import resolve_device, default_num_workers, use_pin_memory, add_device_arg

RDLogger.DisableLog("rdApp.*")


@torch.inference_mode()
def generate_with_sampling(
    model: Transformer,
    src_c_nmr: typing.Dict[str, torch.Tensor],
    src_h_nmr: typing.Dict[str, torch.Tensor],
    tokenizer: spm.SentencePieceProcessor,
    max_tokens: int = 275,
    num_samples: int = 10,
    top_k: int = 50,
    top_p: float = 0.95,
    temperature: float = 1.0,
    device: torch.device = torch.device("cpu"),
    generation_batch_size: int = None,
    batch_size: int = None,
) -> typing.List[typing.List[typing.Tuple[str, float]]]:
    """
    Generate multiple SMILES candidates with scores using top-k/top-p sampling.
    
    Returns:
        List of lists, where each inner list contains (SMILES, score) tuples
        sorted by score (highest first). Score is the sum of log probabilities.
    """
    model.eval()

    # Use provided batch_size or infer from tensor shape
    if batch_size is None:
        batch_size = src_c_nmr["spectrum"].shape[0]
    else:
        # Limit to actual batch size if provided batch_size is larger
        actual_batch_size = src_c_nmr["spectrum"].shape[0]
        batch_size = min(batch_size, actual_batch_size)
    
    system_tokens = {
        tokenizer.bos_id(),
        tokenizer.eos_id(),
        tokenizer.pad_id(),
        tokenizer.unk_id(),
    }

    src_c_nmr = move_to_device(src_c_nmr, device)
    src_h_nmr = move_to_device(src_h_nmr, device)

    # If generation_batch_size is set, process in chunks to control memory usage
    # If None, process all samples at once (original behavior)
    if generation_batch_size is None or generation_batch_size <= 0:
        generation_batch_size = batch_size * num_samples
    
    # Generate num_samples candidates for each item in the batch
    # Process in chunks if generation_batch_size is smaller than total
    all_results: typing.List[typing.List[typing.Tuple[str, float]]] = []
    
    # Calculate how many samples to process per chunk for each item
    # If generation_batch_size allows processing all items together, do that
    if generation_batch_size >= batch_size * num_samples:
        # Process all items together (original behavior)
        expanded_src_c = {
            k: v.repeat_interleave(num_samples, dim=0) for k, v in src_c_nmr.items()
        }
        expanded_src_h = {
            k: v.repeat_interleave(num_samples, dim=0) for k, v in src_h_nmr.items()
        }
        
        total_samples = batch_size * num_samples
        idxs = torch.full(
            (total_samples, 1),
            fill_value=tokenizer.bos_id(),
            dtype=torch.int64,
            device=device,
        )
        
        log_probs_sum = torch.zeros(total_samples, device=device)
        finished = torch.zeros(total_samples, dtype=torch.bool, device=device)

        for step in range(max_tokens):
            logits = model(
                src_c_nmr=expanded_src_c,
                src_h_nmr=expanded_src_h,
                tgt=idxs,
            )
            
            # Get log probabilities for the last token
            log_probs = F.log_softmax(logits[:, -1, :] / temperature, dim=-1)
            
            # Apply top-k filtering
            if top_k > 0 and top_k < log_probs.size(-1):
                top_k_log_probs, top_k_indices = torch.topk(log_probs, top_k, dim=-1)
                # Create a mask and set non-top-k values to very negative
                mask = torch.full_like(log_probs, float('-inf'))
                mask.scatter_(-1, top_k_indices, top_k_log_probs)
                log_probs = mask
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_log_probs, sorted_indices = torch.sort(log_probs, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_log_probs, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Create mask for tokens to remove (keep tokens where cumulative <= top_p)
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift to keep at least one token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                
                # Scatter sorted values back
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                log_probs = log_probs.masked_fill(indices_to_remove, float('-inf'))
            
            # Sample from the filtered distribution
            probs = F.softmax(log_probs, dim=-1)
            new_idxs = torch.multinomial(probs, num_samples=1)
            
            # Accumulate log probabilities (only for non-finished sequences)
            step_log_probs = torch.gather(log_probs, dim=-1, index=new_idxs).squeeze(-1)
            log_probs_sum = log_probs_sum + step_log_probs * (~finished).float()
            
            idxs = torch.cat([idxs, new_idxs], dim=-1)
            
            # Check for EOS tokens
            eos_mask = (new_idxs.squeeze(-1) == tokenizer.eos_id())
            finished = finished | eos_mask
            
            # Early stopping if all sequences are finished
            if finished.all():
                break

        # Decode and collect unique SMILES with scores for each input
        idxs_list = idxs.cpu().numpy().tolist()
        log_probs_list = log_probs_sum.cpu().numpy().tolist()
        
        # Group results by original batch item
        for batch_idx in range(batch_size):
            candidates: typing.Dict[str, float] = {}
            start_idx = batch_idx * num_samples
            end_idx = start_idx + num_samples
            
            for seq, score in zip(idxs_list[start_idx:end_idx], log_probs_list[start_idx:end_idx]):
                filtered = [tok for tok in seq if tok not in system_tokens]
                if not filtered:
                    continue
                decoded = tokenizer.decode([filtered])
                # tokenizer.decode returns a list when given a list of lists, extract the string
                smiles = decoded[0] if isinstance(decoded, list) else decoded
                # Keep the highest score for duplicate SMILES
                if smiles not in candidates or score > candidates[smiles]:
                    candidates[smiles] = score
            
            # Sort by score (descending) and return as list of tuples
            sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
            all_results.append(sorted_candidates)
    else:
        # Process items one at a time with chunking
        samples_per_chunk = max(1, generation_batch_size)
        
        for batch_idx in range(batch_size):
            # Process each input item separately
            item_src_c = {k: v[batch_idx:batch_idx+1] for k, v in src_c_nmr.items()}
            item_src_h = {k: v[batch_idx:batch_idx+1] for k, v in src_h_nmr.items()}
            
            # Process num_samples in chunks
            item_candidates: typing.Dict[str, float] = {}
            
            for chunk_start in range(0, num_samples, samples_per_chunk):
                chunk_end = min(chunk_start + samples_per_chunk, num_samples)
                chunk_num_samples = chunk_end - chunk_start
                
                # Expand for this chunk
                chunk_src_c = {
                    k: v.repeat_interleave(chunk_num_samples, dim=0) for k, v in item_src_c.items()
                }
                chunk_src_h = {
                    k: v.repeat_interleave(chunk_num_samples, dim=0) for k, v in item_src_h.items()
                }
                
                total_chunk_samples = chunk_num_samples
                
                idxs = torch.full(
                    (total_chunk_samples, 1),
                    fill_value=tokenizer.bos_id(),
                    dtype=torch.int64,
                    device=device,
                )
                
                log_probs_sum = torch.zeros(total_chunk_samples, device=device)
                finished = torch.zeros(total_chunk_samples, dtype=torch.bool, device=device)

                for step in range(max_tokens):
                    logits = model(
                        src_c_nmr=chunk_src_c,
                        src_h_nmr=chunk_src_h,
                        tgt=idxs,
                    )
                    
                    # Get log probabilities for the last token
                    log_probs = F.log_softmax(logits[:, -1, :] / temperature, dim=-1)
                    
                    # Apply top-k filtering
                    if top_k > 0 and top_k < log_probs.size(-1):
                        top_k_log_probs, top_k_indices = torch.topk(log_probs, top_k, dim=-1)
                        # Create a mask and set non-top-k values to very negative
                        mask = torch.full_like(log_probs, float('-inf'))
                        mask.scatter_(-1, top_k_indices, top_k_log_probs)
                        log_probs = mask
                    
                    # Apply top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_log_probs, sorted_indices = torch.sort(log_probs, descending=True, dim=-1)
                        sorted_probs = F.softmax(sorted_log_probs, dim=-1)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                        
                        # Create mask for tokens to remove (keep tokens where cumulative <= top_p)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Shift to keep at least one token
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = False
                        
                        # Scatter sorted values back
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        log_probs = log_probs.masked_fill(indices_to_remove, float('-inf'))
                    
                    # Sample from the filtered distribution
                    probs = F.softmax(log_probs, dim=-1)
                    new_idxs = torch.multinomial(probs, num_samples=1)
                    
                    # Accumulate log probabilities (only for non-finished sequences)
                    step_log_probs = torch.gather(log_probs, dim=-1, index=new_idxs).squeeze(-1)
                    log_probs_sum = log_probs_sum + step_log_probs * (~finished).float()
                    
                    idxs = torch.cat([idxs, new_idxs], dim=-1)
                    
                    # Check for EOS tokens
                    eos_mask = (new_idxs.squeeze(-1) == tokenizer.eos_id())
                    finished = finished | eos_mask
                    
                    # Early stopping if all sequences are finished
                    if finished.all():
                        break

                # Decode and collect unique SMILES with scores for this chunk
                idxs_list = idxs.cpu().numpy().tolist()
                log_probs_list = log_probs_sum.cpu().numpy().tolist()
                
                for seq, score in zip(idxs_list, log_probs_list):
                    filtered = [tok for tok in seq if tok not in system_tokens]
                    if not filtered:
                        continue
                    decoded = tokenizer.decode([filtered])
                    # tokenizer.decode returns a list when given a list of lists, extract the string
                    smiles = decoded[0] if isinstance(decoded, list) else decoded
                    # Keep the highest score for duplicate SMILES
                    if smiles not in item_candidates or score > item_candidates[smiles]:
                        item_candidates[smiles] = score
            
            # Sort by score (descending) and add to results
            sorted_candidates = sorted(item_candidates.items(), key=lambda x: x[1], reverse=True)
            all_results.append(sorted_candidates)

    return all_results


def evaluate_top_n(
    args: argparse.Namespace,
) -> None:
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    optimized = _load_optimized_sampling_defaults()
    if optimized:
        print(
            "Sampling defaults from models/optimized_params.json: "
            f"top_k={int(optimized.get('top_k', 50))}, "
            f"top_p={float(optimized.get('top_p', 0.95)):.3f}, "
            f"temperature={float(optimized.get('temperature', 1.0)):.3f}"
        )

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

    model = build_model(tokenizer, smiles_pad_token=tokenizer.pad_id()).to(device)

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    results: typing.List[typing.Dict[str, typing.Any]] = []
    
    # Track metrics for different categories
    categories = {
        "only_c": {"count": 0, "top10": 0, "top10_valid": 0, "unique_samples": []},
        "only_h": {"count": 0, "top10": 0, "top10_valid": 0, "unique_samples": []},
        "both": {"count": 0, "top10": 0, "top10_valid": 0, "unique_samples": []},
    }

    for batch in tqdm(loader, desc="Evaluating"):
        smiles_tokens = batch["smiles"]
        true_smiles = [
            tokenizer.decode(smiles_tokens[i].cpu().numpy().tolist())
            for i in range(smiles_tokens.shape[0])
        ]

        # Generate candidates with scores
        candidates_list = generate_with_sampling(
            model=model,
            src_c_nmr=batch["C_NMR"],
            src_h_nmr=batch["H_NMR"],
            tokenizer=tokenizer,
            max_tokens=args.max_tokens,
            num_samples=args.num_samples,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            device=device,
            generation_batch_size=args.generation_batch_size,
            batch_size=args.batch_size,
        )

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

        for i in range(len(true_smiles)):
            missed_c = bool(missed_c_list[i].item())
            missed_h = bool(missed_h_list[i].item())
            
            # Determine category
            if missed_h and not missed_c:
                category = "only_c"
            elif not missed_h and missed_c:
                category = "only_h"
            elif not missed_c and not missed_h:
                category = "both"
            else:
                continue  # Skip if both are missing
            
            categories[category]["count"] += 1
            
            candidates = candidates_list[i]
            true_smiles_i = true_smiles[i]
            
            # Extract top predictions (using unique SMILES only)
            # Note: The code deduplicates SMILES, so we work with unique candidates only.
            # If fewer than N unique candidates exist, we use all available unique ones.
            num_unique = len(candidates)
            categories[category]["unique_samples"].append(num_unique)
            
            top10_smiles = [c[0] for c in candidates[:10]]  # Up to 10 unique SMILES
            
            # Check validity
            top10_valid = [Chem.MolFromSmiles(s) is not None for s in top10_smiles]
            
            # Check exact matches
            top10_correct = true_smiles_i in top10_smiles
            
            # Update metrics
            if top10_correct:
                categories[category]["top10"] += 1
            
            # Track validity (how many predictions are valid SMILES)
            if any(top10_valid):
                categories[category]["top10_valid"] += 1

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
                    "smiles": true_smiles_i,
                    "true_smiles": true_smiles_i,
                    "top10_predictions": top10_smiles,
                    "all_candidates": [(s, float(score)) for s, score in candidates],  # score = sum of log probabilities (log-likelihood)
                    "has_carbon": not missed_c,
                    "has_hydrogen": not missed_h,
                    "carbon": carbon_entry,
                    "hydrogen": hydrogen_entry,
                    "top10_exact_match": top10_correct,
                }
            )

    # Print results
    for cat_name, metrics in categories.items():
        count = metrics["count"]
        if count == 0:
            continue
        unique_samples = metrics["unique_samples"]
        avg_unique = sum(unique_samples) / len(unique_samples) if unique_samples else 0
        min_unique = min(unique_samples) if unique_samples else 0
        max_unique = max(unique_samples) if unique_samples else 0
        
        print(f"\n{cat_name.upper()}:")
        print(f"  Count: {count}")
        print(f"  Unique candidates per sample: avg={avg_unique:.2f}, min={min_unique}, max={max_unique}")
        print(f"  Top-10 Accuracy: {metrics['top10'] / count:.4f}")
        print(f"  Top-10 Validity: {metrics['top10_valid'] / count:.4f}")

    # Overall metrics
    all_unique_samples = []
    for m in categories.values():
        all_unique_samples.extend(m["unique_samples"])
    
    overall = {
        "count": sum(m["count"] for m in categories.values()),
        "top10": sum(m["top10"] for m in categories.values()),
        "top10_valid": sum(m["top10_valid"] for m in categories.values()),
    }
    
    if overall["count"] > 0:
        avg_unique_overall = sum(all_unique_samples) / len(all_unique_samples) if all_unique_samples else 0
        min_unique_overall = min(all_unique_samples) if all_unique_samples else 0
        max_unique_overall = max(all_unique_samples) if all_unique_samples else 0
        
        print(f"\nOVERALL:")
        print(f"  Count: {overall['count']}")
        print(f"  Unique candidates per sample: avg={avg_unique_overall:.2f}, min={min_unique_overall}, max={max_unique_overall}")
        print(f"  Top-10 Accuracy: {overall['top10'] / overall['count']:.4f}")
        print(f"  Top-10 Validity: {overall['top10_valid'] / overall['count']:.4f}")

    if args.output_json is not None:
        summary = {}
        for cat_name, metrics in categories.items():
            count = metrics["count"]
            if count > 0:
                summary[cat_name] = {
                    "count": count,
                    "top10_accuracy": metrics["top10"] / count,
                    "top10_validity": metrics["top10_valid"] / count,
                }
        
        if overall["count"] > 0:
            summary["overall"] = {
                "count": overall["count"],
                "top10_accuracy": overall["top10"] / overall["count"],
                "top10_validity": overall["top10_valid"] / overall["count"],
            }
        
        with open(args.output_json, "w") as fp:
            json.dump(
                {
                    "predictions": results,
                    "summary": summary,
                },
                fp,
                indent=2,
            )


def _load_optimized_sampling_defaults() -> dict:
    params_path = Path(__file__).resolve().parents[2] / "models" / "optimized_params.json"
    if not params_path.exists():
        return {}
    with open(params_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("best_params", {})


def parse_args() -> argparse.Namespace:
    optimized = _load_optimized_sampling_defaults()
    parser = argparse.ArgumentParser(
        description="Evaluate bimodal Transformer on NMR data with top-N metrics."
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
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=None,
        help="Batch size for generation (controls memory usage). If None, processes all samples at once. Smaller values use less memory but may be slower.",
    )
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
        "--num-samples",
        type=int,
        default=10,
        help="Number of candidate SMILES to generate per input. Note: After deduplication, you may have fewer unique candidates for top-N evaluation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(optimized.get("top_k", 50)),
        help="Top-k sampling parameter (0 to disable).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=float(optimized.get("top_p", 0.95)),
        help="Top-p (nucleus) sampling parameter (1.0 to disable).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(optimized.get("temperature", 1.0)),
        help="Temperature for sampling.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save predictions as JSON.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_top_n(parse_args())

