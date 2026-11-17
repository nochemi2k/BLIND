#!/usr/bin/env python3
"""
Trains the generator of realistic elemental analyses (src/blind/ea_noise_flow.py) on the experimental
analyses of modalities/ea/build_dataset.py and compares it with simpler noise models on held-out articles.

    python modalities/ea/train_noise_flow.py --dataset data/ea/dataset/ea_dataset.jsonl.gz --output data/ea/noise_flow.pt
    python modalities/ea/train_noise_flow.py --output data/ea/noise_flow_v2.pt --compare data/ea/noise_flow.pt

Analyses with a SMILES are used, organic and metal compounds together. Train / val / test are split by
article (PMCID), as one lab's analyses share their errors. Written next to the checkpoint:
<output>.report.json with, on the test articles,
  - NLL per element (nats, Found values to 0.01) of: independent Gaussians per element (one spread for all
    molecules), the flow given only the calculated % of C, H, N, S, and the flow given the whole formula;
  - coverage of the 50 % and 90 % central intervals of the flow samples;
  - classifier two-sample test: AUC of an MLP that tells real Found - Calcd vectors from generated ones for
    the same molecules (0.5 = indistinguishable), also for the resampling model of
    src/blind/elemental_analysis.py (data/ea/ea_noise_all.json) and for the --compare checkpoint;
  - Other = 100 - C - H - N - S of fully reported analyses: share below 0 (the four exceed 100 %) and
    its quantiles, real and generated.
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
import sys
import typing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch

from blind.elemental_analysis import ElementalAnalysisNoise, mass_percent
from blind.ea_noise_flow import NONMETALS, NUM_FEATURES, TARGETS, NoiseFlow, compound, features, target_arrays

# Larger deviations come from a wrong structure or formula, not from the measurement
MAX_ABS_DELTA = 5.0
MAX_ZERO_FOUND = 0.5


def split_of(pmcid: str) -> str:
    h = int(hashlib.md5(str(pmcid).encode()).hexdigest()[:8], 16) % 10
    return "test" if h == 0 else "val" if h == 1 else "train"


def load(dataset_path: str) -> dict:
    rows, seen, zero_found = [], set(), []
    dropped = collections.Counter()
    for line in gzip.open(dataset_path, "rt"):
        r = json.loads(line)
        if not r["smiles"]:
            continue
        counts = compound(r["smiles"], r["solvates"])
        if counts is None:
            dropped["no_composition"] += 1
            continue
        key = (r["smiles"], json.dumps(r["found"], sort_keys=True))
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        calc, present = target_arrays(counts)
        observed = np.array([float(present[j] > 0 and e in r["found"]) for j, e in enumerate(TARGETS)], dtype=np.float32)
        delta = np.array([r["found"].get(e, 0.0) - calc[j] for j, e in enumerate(TARGETS)], dtype=np.float32) * observed
        for j, e in enumerate(TARGETS):
            if not present[j] and e in r["found"] and r["found"][e] <= MAX_ZERO_FOUND:
                zero_found.append(r["found"][e])
        if not observed.any():
            dropped["no_target_reported"] += 1
            continue
        if np.abs(delta).max() > MAX_ABS_DELTA:
            dropped[f"abs_delta_over_{MAX_ABS_DELTA}"] += 1
            continue
        rows.append({
            "smiles": r["smiles"], "counts": counts, "calc": calc, "present": present, "observed": observed,
            "delta": delta, "split": split_of(r["pmcid"]), "metal": any(e not in NONMETALS for e in counts),
            "pmcid": r["pmcid"],
        })
    data = {"rows": rows, "zero_found": zero_found, "dropped": dict(dropped)}
    for name in ("calc", "present", "observed", "delta"):
        data[name] = np.stack([r[name] for r in rows])
    data["feats"] = np.stack([features(r["counts"]) for r in rows])
    data["split"] = np.array([r["split"] for r in rows])
    data["metal"] = np.array([r["metal"] for r in rows])
    return data


def tensors(data: dict, mask: np.ndarray, device: str) -> typing.List[torch.Tensor]:
    return [torch.tensor(data[k][mask], device=device) for k in ("feats", "calc", "present", "delta", "observed")]


def train_flow(data: dict, args, calc_only: bool) -> NoiseFlow:
    torch.manual_seed(0)
    model = NoiseFlow().to(args.device)
    train = tensors(data, data["split"] == "train", args.device)
    val = tensors(data, data["split"] == "val", args.device)
    if calc_only:
        train[0].zero_()
        val[0].zero_()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    best, best_state, n = float("inf"), None, len(train[0])
    for epoch in range(args.epochs):
        model.train()
        for idx in torch.randperm(n, device=args.device).split(args.batch_size):
            feats, calc, present, delta, observed = (t[idx] for t in train)
            # Dequantization: Found values are rounded to 0.01
            delta = delta + (torch.rand_like(delta) - 0.5) * 0.01 * observed
            loss = -model.log_prob(feats, calc, present, delta, observed).sum() / observed.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_nll = -evaluate_nll(model, val).sum() / val[4].sum().item()
        if val_nll < best:
            best, best_state = val_nll, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  {'calc-only' if calc_only else 'formula'} epoch {epoch}: val NLL {val_nll:.4f} (best {best:.4f})", flush=True)
    model.load_state_dict(best_state)
    return model.eval()


@torch.no_grad()
def evaluate_nll(model: NoiseFlow, t: typing.List[torch.Tensor]) -> torch.Tensor:
    """[len(TARGETS)] summed log likelihood of the Found values rounded to 0.01 (density x 0.01)."""
    model.eval()
    feats, calc, present, delta, observed = t
    lp = torch.cat([model.log_prob(*[x[i] for x in (feats, calc, present, delta, observed)]) for i in torch.arange(len(feats), device=feats.device).split(8192)])
    return ((lp + np.log(0.01)) * observed).sum(0).cpu()


def gaussian_nll(data: dict, test: np.ndarray) -> np.ndarray:
    train = data["split"] == "train"
    out = []
    for j in range(len(TARGETS)):
        tr = data["delta"][train][data["observed"][train][:, j] > 0, j]
        te = data["delta"][test][data["observed"][test][:, j] > 0, j]
        mu, sd = tr.mean(), tr.std()
        out.append(-(-0.5 * ((te - mu) / sd) ** 2 - np.log(sd * np.sqrt(2 * np.pi)) + np.log(0.01)).sum())
    return np.array(out)


def resampling_deltas(data: dict, mask: np.ndarray, path: str, rng: np.random.Generator) -> np.ndarray:
    """Found - Calcd of the resampling noise model (solvate + resampled vector) for the observed targets."""
    noise = ElementalAnalysisNoise(path)
    rows = [r for r, m in zip(data["rows"], mask) if m]
    out = np.zeros((len(rows), len(TARGETS)), dtype=np.float32)
    for i, r in enumerate(rows):
        counts = collections.Counter(r["counts"])
        if rng.random() < noise.solvate_prob:
            k = rng.choice(len(noise.solvates), p=noise.solvate_weights)
            for e, n in noise.solvate_counts[k].items():
                counts[e] += noise.solvates[k][1] * n
        exact = mass_percent(counts)
        elements = frozenset(e for j, e in enumerate(TARGETS) if r["observed"][j])
        delta = noise._delta(elements, rng)
        for j, e in enumerate(TARGETS):
            if r["observed"][j]:
                out[i, j] = exact.get(e, 0.0) + delta.get(e, 0.0) - r["calc"][j]
    return out


def two_sample_auc(data: dict, mask: np.ndarray, fake: np.ndarray, device: str, seed: int = 0) -> float:
    """AUC of an MLP telling real from generated (formula features, delta, observed) on held-out molecules."""
    feats = data["feats"][mask]
    observed = data["observed"][mask]
    real = data["delta"][mask]
    x = np.concatenate([
        np.concatenate([feats, real * observed, observed], 1),
        np.concatenate([feats, fake * observed, observed], 1),
    ]).astype(np.float32)
    y = np.concatenate([np.ones(len(real)), np.zeros(len(real))]).astype(np.float32)
    # Molecules, not rows, are split: a molecule's real and fake vectors land on the same side
    side = np.random.default_rng(seed).random(len(real)) < 0.5
    side = np.concatenate([side, side])
    torch.manual_seed(seed)
    clf = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 128), torch.nn.GELU(), torch.nn.Linear(128, 128), torch.nn.GELU(), torch.nn.Linear(128, 1)).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    xt, yt = torch.tensor(x[side], device=device), torch.tensor(y[side], device=device)
    for _ in range(60):
        for idx in torch.randperm(len(xt), device=device).split(256):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(clf(xt[idx])[:, 0], yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        score = clf(torch.tensor(x[~side], device=device))[:, 0].cpu().numpy()
    label = y[~side]
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = label == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


@torch.no_grad()
def flow_deltas(model: NoiseFlow, data: dict, mask: np.ndarray, device: str, num_samples: int, seed: int) -> np.ndarray:
    feats, calc, present, _, _ = tensors(data, mask, device)
    generator = torch.Generator(device=device).manual_seed(seed)
    rep = lambda t: t.repeat_interleave(num_samples, 0)
    out = []
    for i in torch.arange(len(feats), device=device).split(max(1, 8192 // num_samples)):
        out.append(model.sample(rep(feats[i]), rep(calc[i]), rep(present[i]), generator).reshape(len(i), num_samples, -1).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dataset", default="data/ea/dataset/ea_dataset.jsonl.gz")
    parser.add_argument("--output", default="data/ea/noise_flow.pt")
    parser.add_argument("--resampling-model", default="data/ea/ea_noise_all.json")
    parser.add_argument("--compare", help="Checkpoint of another generator to compare with on the same test articles")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    data = load(args.dataset)
    splits = collections.Counter(data["split"])
    print(f"{len(data['rows'])} analyses ({dict(splits)}), dropped {data['dropped']}, "
          f"observed per element {dict(zip(TARGETS, data['observed'].sum(0).astype(int).tolist()))}, "
          f"{len(data['zero_found'])} measured absent elements", flush=True)

    flow = train_flow(data, args, calc_only=False)
    calc_flow = train_flow(data, args, calc_only=True)
    other_flow = None
    if args.compare:
        checkpoint = torch.load(args.compare, map_location=args.device, weights_only=False)
        other_flow = NoiseFlow(**checkpoint["config"]).to(args.device).eval()
        other_flow.load_state_dict(checkpoint["state_dict"])

    report = {"num_analyses": dict(splits), "dropped": data["dropped"], "max_abs_delta": MAX_ABS_DELTA, "test": {}}
    test = data["split"] == "test"
    for group, mask in (("all", test), ("organic", test & ~data["metal"]), ("metal", test & data["metal"])):
        t = tensors(data, mask, args.device)
        num = data["observed"][mask].sum(0)
        comp = tensors(data, mask, args.device)
        comp[0].zero_()
        nll = {
            "gaussian_per_element": gaussian_nll(data, mask) / num,
            "flow_calc_only": -evaluate_nll(calc_flow, comp).numpy() / num,
            "flow": -evaluate_nll(flow, t).numpy() / num,
        }
        if other_flow is not None:
            nll["compare"] = -evaluate_nll(other_flow, t).numpy() / num
        samples = flow_deltas(flow, data, mask, args.device, 200, seed=1)
        real, observed = data["delta"][mask], data["observed"][mask] > 0
        coverage = {}
        for level in (0.5, 0.9):
            lo, hi = np.quantile(samples, [(1 - level) / 2, (1 + level) / 2], axis=1)
            inside = (real >= lo - 0.005) & (real <= hi + 0.005)
            coverage[str(level)] = {e: float(inside[observed[:, j], j].mean()) for j, e in enumerate(TARGETS)}
        rng = np.random.default_rng(2)
        auc = {
            "flow": two_sample_auc(data, mask, samples[:, 0], args.device),
            "resampling_model": two_sample_auc(data, mask, resampling_deltas(data, mask, args.resampling_model, rng), args.device),
        }
        if other_flow is not None:
            auc["compare"] = two_sample_auc(data, mask, flow_deltas(other_flow, data, mask, args.device, 1, seed=1)[:, 0], args.device)
        full = (data["observed"][mask] == data["present"][mask]).all(1)
        calc = data["calc"][mask][full]
        other_real = 100 - (calc + real[full]).sum(1)
        other_fake = 100 - (calc[:, None] + samples[full]).sum(-1).ravel()
        other = {
            name: {"share_below_0": float((v < 0).mean()), "quantiles_1_50_99": np.round(np.quantile(v - np.repeat(100 - calc.sum(1), len(v) // len(calc)), [0.01, 0.5, 0.99]), 3).tolist()}
            for name, v in (("real", other_real), ("flow", other_fake))
        }
        report["test"][group] = {
            "num_analyses": int(mask.sum()),
            "num_observed": dict(zip(TARGETS, num.astype(int).tolist())),
            "nll_per_value": {k: dict(zip(TARGETS, np.round(v, 4).tolist())) for k, v in nll.items()},
            "coverage": coverage,
            "two_sample_auc": auc,
            "other_minus_calc": other,
        }
        print(group, json.dumps(report["test"][group]), flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        "config": {"num_features": NUM_FEATURES},
        "state_dict": {k: v.cpu() for k, v in flow.state_dict().items()},
        "zero_found": data["zero_found"],
        "targets": TARGETS,
        "max_abs_delta": MAX_ABS_DELTA,
    }, args.output)
    json.dump(report, open(os.path.splitext(args.output)[0] + ".report.json", "w"), indent=2)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
