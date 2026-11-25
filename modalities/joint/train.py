#!/usr/bin/env python3
"""
Trains one model from scratch on 1H, 13C, heteronuclear spectra, elemental analysis and MS2 at once
(see enable_joint.py). Run from the repository root after prepare_joint.py:

    python modalities/joint/train.py --device cuda --mixed-precision --exp-prefix joint

Takes the arguments of train.py (--batch-size, --epochs, --lr, --num-workers, --checkpoint-dir, ...;
--train-path / --val-path are not used) plus
    --joint-dir      data of prepare_joint.py (default data/joint; its tokenizer is the default --tokenizer-path)
    --nmr-dir        NMR splits (default data/nmrexp/processed_hetero)
    --epoch-size     examples per training epoch (default 1000000), --val-epoch-size (default 100000)
    --scenario-size  validation examples per input combination (default 512), --val-every N epochs (default 5)
    --flow-path, --ea-noise   elemental analysis generator, as in modalities/ea
    --warmup-steps   linear learning-rate warmup at the start (default 4000 steps)
    --model-size     large (default: d_model 384, 8 layers, d_ff 1536, dropout 0.1) or base (the earlier models)
    --resume         continue from <checkpoint-dir>/<exp-prefix>_last.pt (model, optimizer, scheduler, logs)

The loss ignores padding tokens. Written to --checkpoint-dir: <prefix>_epoch_N.pt every --save-every
epochs, <prefix>_best.pt (highest mean scenario accuracy), <prefix>_last.pt (to resume), <prefix>_logs.json.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import enable_joint
from enable_joint import enable_ms

import numpy as np
import sentencepiece as spm
import torch
from tqdm.auto import tqdm

from blind.device import autocast_context, create_grad_scaler, resolve_device, supports_cuda_amp, use_pin_memory
from blind.training import move_to_device, parse_args, seed_everything

pop = enable_ms.pop_argument


def main() -> None:
    enable_joint.JOINT_DIR = pop("--joint-dir") or enable_joint.JOINT_DIR
    enable_joint.NMR_DIR = pop("--nmr-dir") or enable_joint.NMR_DIR
    enable_joint.EPOCH_SIZE = int(pop("--epoch-size") or enable_joint.EPOCH_SIZE)
    enable_joint.VAL_EPOCH_SIZE = int(pop("--val-epoch-size") or enable_joint.VAL_EPOCH_SIZE)
    enable_joint.SCENARIO_SIZE = int(pop("--scenario-size") or enable_joint.SCENARIO_SIZE)
    val_every = int(pop("--val-every") or 5)
    warmup_steps = int(pop("--warmup-steps") or 4000)
    enable_ms.enable_ea.FLOW_PATH = pop("--flow-path") or enable_ms.enable_ea.FLOW_PATH
    enable_ms.enable_ea.EA_NOISE = pop("--ea-noise") or enable_ms.enable_ea.EA_NOISE
    enable_ms.enable_ea.MODEL_SIZE = pop("--model-size") or "large"
    resume = pop("--resume", is_flag=True)
    if "--tokenizer-path" not in sys.argv:
        sys.argv += ["--tokenizer-path", str(Path(enable_joint.JOINT_DIR) / "smiles_tokenizer.model")]
    args = parse_args()

    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}; tokenizer {args.tokenizer_path}")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(args.tokenizer_path)
    args.pin_memory = use_pin_memory(device)

    train_loader, val_loader, (val_nmr, val_ms_only, val_store) = enable_joint.build_loaders(tokenizer, args)
    scenarios = enable_joint.scenario_examples(val_nmr, val_ms_only, val_store, seed=args.seed)
    print("Validation scenarios: " + ", ".join(f"{k} {len(v)}" for k, v in scenarios.items()))

    model = enable_ms.build_model(tokenizer, smiles_pad_token=tokenizer.pad_id()).to(device)
    print(f"Model from scratch ({enable_ms.enable_ea.MODEL_SIZE}): {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters")
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    use_amp = args.mixed_precision and supports_cuda_amp(device)
    scaler = create_grad_scaler(device, enabled=use_amp)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    prefix = os.path.join(args.checkpoint_dir, args.exp_prefix)
    logs = {"train_loss": [], "val_loss": [], "lr_log": [], "epoch_time_sec": [], "scenarios": {}, "args": vars(args)}
    start_epoch, best = 0, -1.0
    if resume:
        state = torch.load(f"{prefix}_last.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        logs, start_epoch, best = state["logs"], state["epoch"] + 1, state["best"]
        train_loader.sampler.epoch = start_epoch
        print(f"Resumed from {prefix}_last.pt at epoch {start_epoch}")

    vocab = tokenizer.vocab_size()
    step = start_epoch * len(train_loader)
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
        for batch in pbar:
            batch = move_to_device(batch, device=device)
            step += 1
            if step <= warmup_steps:
                for group in optimizer.param_groups:
                    group["lr"] = args.lr * step / warmup_steps
            with autocast_context(device, enabled=use_amp):
                logits = model(src_c_nmr=batch["C_NMR"], src_h_nmr=batch["H_NMR"], tgt=batch["smiles"][:, :-1])
                loss = criterion(logits.reshape(-1, vocab), batch["smiles"][:, 1:].reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
            pbar.set_postfix({"loss": losses[-1]})

        model.eval()
        val_losses = []
        with torch.inference_mode():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                batch = move_to_device(batch, device=device)
                with autocast_context(device, enabled=use_amp):
                    logits = model(src_c_nmr=batch["C_NMR"], src_h_nmr=batch["H_NMR"], tgt=batch["smiles"][:, :-1])
                    val_losses.append(criterion(logits.reshape(-1, vocab), batch["smiles"][:, 1:].reshape(-1)).item())

        train_loss, val_loss = float(np.mean(losses)), float(np.mean(val_losses))
        scheduler.step(val_loss)
        logs["train_loss"].append(train_loss)
        logs["val_loss"].append(val_loss)
        logs["lr_log"].append(float(optimizer.param_groups[0]["lr"]))
        message = f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={logs['lr_log'][-1]:.6f}"

        if (epoch + 1) % val_every == 0 or epoch + 1 == args.epochs:
            accuracy = enable_joint.scenario_accuracy(model, scenarios, tokenizer, device, args.batch_size)
            logs["scenarios"][str(epoch)] = accuracy
            mean = float(np.mean(list(accuracy.values())))
            message += f" scenarios mean={mean:.4f} | " + " ".join(f"{k}={v:.3f}" for k, v in accuracy.items())
            if mean > best:
                best = mean
                torch.save(model.state_dict(), f"{prefix}_best.pt")
                message += " (best)"
        logs["epoch_time_sec"].append(time.time() - epoch_start)
        print(message + f" time={logs['epoch_time_sec'][-1]:.0f}s", flush=True)

        with open(f"{prefix}_logs.json", "w") as f:
            json.dump(logs, f)
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            torch.save(model.state_dict(), f"{prefix}_epoch_{epoch}.pt")
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "logs": logs,
        }, f"{prefix}_last.pt")


if __name__ == "__main__":
    main()
