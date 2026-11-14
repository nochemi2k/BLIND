"""
Learned generator of realistic elemental analyses: p(Found - Calcd | composition) for C, H, N, S.

Trained by ea/train_noise_flow.py on the experimental analyses of ea/build_dataset.py (organic
compounds and metal complexes together; no compound class is given).

    Delta_e = Found_e - Calcd_e(compound),  e in C, H, N, S present in the compound,
where the compound is the SMILES plus the acids and counterions of the reported formula (a salt is
part of the substance, as in the NMR data), but not the neutral solvates (H2O, CH2Cl2, ...): their
shift of the Found values is part of what is learned, as a real sample also holds them.

The only input is the molecular formula (mass %, metal content, molecular weight, atom ratios):
what an elemental analysis tells anyway. No structural features, so the generated noise cannot
carry a hint about the structure that real analyses do not have.

The density is autoregressive over C -> H -> N -> S, every factor a mixture of NUM_COMPONENTS
Gaussians whose parameters come from an MLP over the formula features, the previous deltas and
their observed flags (an autoregressive flow with mixture-CDF transforms: exact likelihood, one
pass to sample). An element of the compound that the article did not report is masked out of the
likelihood, so the 10k analyses with sulfur train the S factor and all 90k train C, H, N.

Elements absent from the compound but measured (mostly "N, 0.00") follow the empirical Found
values of such reports (zero_found in the checkpoint), the same for N and S.
"""
import collections
import typing

import numpy as np
import torch

from .elemental_analysis import ATOMIC_MASS, formula_counts, formula_counts_plain, mass_percent

TARGETS = ("C", "H", "N", "S")
COMPOSITION_ELEMENTS = ("C", "H", "N", "O", "S", "F", "Cl", "Br", "I", "P", "Si", "B", "Se")
NONMETALS = {"H", "D", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I", "As", "Te"}
NUM_COMPONENTS = 8
MIN_SIGMA = 0.005  # Found values have two decimals

# Neutral solvates whose shift is learned as noise; anything else after "·" (HCl, TFA, PF6, ...) is
# part of the compound. Names as written by ea/build_dataset.py.
NEUTRAL_SOLVATES = {
    "H2O", "CH2Cl2", "CHCl3", "CH4O", "C2H6O", "C3H8O", "C2H3N", "C6H14", "C5H12", "C4H8O", "C4H8O2",
    "C7H8", "C6H6", "C4H10O", "C3H7NO", "C3H6O", "C2H6OS", "C2H4O2", "CH3NO2", "C2H4Cl2", "C5H5N", "C6H12",
    "C8H10", "CCl4", "C2H5OH", "CH3OH", "CH3CN",
}


def compound(smiles: str, solvates: typing.Sequence = ()) -> typing.Optional[typing.Dict[str, float]]:
    """Atom counts of the substance: SMILES + non-neutral solvates of the formula."""
    counts = formula_counts(smiles)
    if counts is None:
        return None
    counts = collections.Counter(counts)
    for name, multiplier in solvates:
        name = name.strip("()")
        if name not in NEUTRAL_SOLVATES:
            for e, n in formula_counts_plain(name).items():
                counts[e] += multiplier * n
    if any(e not in ATOMIC_MASS for e in counts):
        return None
    return dict(counts)


def features(counts: typing.Mapping[str, float]) -> np.ndarray:
    """Formula only: mass % of the common elements and of metals, weight, size and atom ratios."""
    percent = mass_percent(counts)
    metal = [e for e in counts if e not in NONMETALS]
    weight = sum(ATOMIC_MASS[e] * n for e, n in counts.items())
    carbon = max(counts.get("C", 0.0), 1.0)
    return np.array(
        [percent.get(e, 0.0) / 10.0 for e in COMPOSITION_ELEMENTS]
        + [
            sum(percent[e] for e in metal) / 10.0,
            sum(percent[e] for e in counts if e not in metal and e not in COMPOSITION_ELEMENTS) / 10.0,
            float(sum(counts[e] for e in metal)),
            np.log(weight) - 6.0,
            np.log(sum(counts.values())) - 3.5,
            counts.get("H", 0.0) / carbon,
            counts.get("N", 0.0) / carbon,
            counts.get("O", 0.0) / carbon,
        ],
        dtype=np.float32,
    )


NUM_FEATURES = len(COMPOSITION_ELEMENTS) + 8


class NoiseFlow(torch.nn.Module):
    """Autoregressive Gaussian-mixture density of the deltas over TARGETS."""

    def __init__(self, num_features: int = NUM_FEATURES, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.context = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout),
        )
        n = len(TARGETS)
        # Input of factor j: context, calculated % of the target, which targets are present, previous
        # deltas and their observed flags
        self.heads = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Linear(hidden + 1 + n + 2 * j, hidden // 2), torch.nn.GELU(),
                torch.nn.Linear(hidden // 2, 3 * NUM_COMPONENTS),
            )
            for j in range(n)
        )

    def _mixture(self, h, calc, present, delta, observed, j):
        x = torch.cat([h, calc[:, j:j + 1] / 10.0, present, delta[:, :j] * observed[:, :j], observed[:, :j]], dim=-1)
        logits, mu, log_sigma = self.heads[j](x).chunk(3, dim=-1)
        return torch.log_softmax(logits, -1), mu, torch.nn.functional.softplus(log_sigma) + MIN_SIGMA

    def log_prob(self, feats, calc, present, delta, observed) -> torch.Tensor:
        """[batch, len(TARGETS)] log density of every delta; 0 where not observed."""
        h = self.context(feats)
        out = []
        for j in range(len(TARGETS)):
            log_w, mu, sigma = self._mixture(h, calc, present, delta, observed, j)
            lp = torch.distributions.Normal(mu, sigma).log_prob(delta[:, j:j + 1])
            out.append(torch.logsumexp(log_w + lp, -1) * observed[:, j])
        return torch.stack(out, -1)

    @torch.no_grad()
    def sample(self, feats, calc, present, generator=None, temperature: float = 1.0) -> torch.Tensor:
        """[batch, len(TARGETS)] deltas of the present targets (0 for the absent ones)."""
        h = self.context(feats)
        delta = torch.zeros_like(calc)
        for j in range(len(TARGETS)):
            log_w, mu, sigma = self._mixture(h, calc, present, delta, present, j)
            k = torch.multinomial(log_w.exp(), 1, generator=generator)
            eps = torch.randn(k.shape, generator=generator, device=k.device)
            value = mu.gather(1, k) + temperature * sigma.gather(1, k) * eps
            delta[:, j] = value[:, 0] * present[:, j]
        return delta


def target_arrays(counts: typing.Mapping[str, float]) -> typing.Tuple[np.ndarray, np.ndarray]:
    percent = mass_percent(counts)
    calc = np.array([percent.get(e, 0.0) for e in TARGETS], dtype=np.float32)
    present = np.array([float(counts.get(e, 0.0) > 0) for e in TARGETS], dtype=np.float32)
    return calc, present


class ElementalAnalysisFlow:
    """Samples Found % of C, H, N, S for SMILES with the model saved by ea/train_noise_flow.py."""

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model = NoiseFlow(**checkpoint["config"]).to(device).eval()
        self.model.load_state_dict(checkpoint["state_dict"])
        self.zero_found = np.asarray(checkpoint["zero_found"], dtype=np.float32)
        self.device = device

    def sample(
        self,
        smiles: typing.Sequence[str],
        num_samples: int,
        seed: int = 0,
        temperature: float = 1.0,
        batch_size: int = 65536,
    ) -> np.ndarray:
        """[len(smiles), num_samples, len(TARGETS)] Found % (two decimals). Elements absent from a
        molecule get the Found value of a measured absent element ("N, 0.00"); NaN if the SMILES
        cannot be parsed. Which elements count as measured is up to the caller."""
        rng = np.random.default_rng(seed)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        out = np.full((len(smiles), num_samples, len(TARGETS)), np.nan, dtype=np.float32)
        rows = []
        for i, s in enumerate(smiles):
            counts = formula_counts(s)
            if counts is not None:
                rows.append((i, features(counts), *target_arrays(counts)))
        step = max(1, batch_size // num_samples)
        for start in range(0, len(rows), step):
            chunk = rows[start:start + step]
            idx = [r[0] for r in chunk]
            feats, calc, present = (torch.tensor(np.stack([r[k] for r in chunk]), device=self.device) for k in (1, 2, 3))
            rep = lambda t: t.repeat_interleave(num_samples, 0)
            delta = self.model.sample(rep(feats), rep(calc), rep(present), generator, temperature)
            found = (rep(calc) + delta).cpu().numpy().reshape(len(chunk), num_samples, -1)
            absent = present.cpu().numpy()[:, None, :] == 0
            zero = rng.choice(self.zero_found, size=found.shape)
            out[idx] = np.round(np.clip(np.where(absent, zero, found), 0.0, 100.0), 2)
        return out
