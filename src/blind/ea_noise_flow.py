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


SPLINE_BINS = 16
SPLINE_BOUND = 6.0
# Deltas enter the spline as asinh(delta / ASINH_SCALE): linear near 0 (the sharp peak of fitted and
# precise analyses), logarithmic in the tails (solvates, up to 5 %), so equal bins cover both
ASINH_SCALE = 0.1


def rational_quadratic_spline(
    x: torch.Tensor, params: torch.Tensor, inverse: bool = False, bound: float = SPLINE_BOUND,
    min_width: float = 1e-3, min_derivative: float = 1e-3,
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """Monotone rational-quadratic spline on [-bound, bound], identity outside (Durkan et al. 2019,
    Neural Spline Flows). params [..., 3 K - 1]: bin widths, heights, interior derivatives.
    Returns the output and the log |Jacobian| of the map applied (of the inverse map when inverse)."""
    k = (params.shape[-1] + 1) // 3
    w, h, d = params[..., :k], params[..., k:2 * k], params[..., 2 * k:]
    widths = min_width + (1 - min_width * k) * torch.softmax(w, -1)
    heights = min_width + (1 - min_width * k) * torch.softmax(h, -1)
    pad = torch.ones(*d.shape[:-1], 1, device=d.device, dtype=d.dtype)
    derivs = torch.cat([pad, min_derivative + torch.nn.functional.softplus(d) / np.log(2), pad], -1)
    cw = torch.nn.functional.pad(torch.cumsum(widths, -1), (1, 0)) * 2 * bound - bound
    ch = torch.nn.functional.pad(torch.cumsum(heights, -1), (1, 0)) * 2 * bound - bound
    cw[..., -1], ch[..., -1] = bound, bound
    inside = (x > -bound) & (x < bound)
    xc = x.clamp(-bound + 1e-6, bound - 1e-6)
    knots = ch if inverse else cw
    idx = (torch.searchsorted(knots.contiguous(), xc[..., None].contiguous()) - 1).clamp(0, k - 1)
    take = lambda t: t.gather(-1, idx)[..., 0]
    x0, wk, y0 = take(cw), take(cw[..., 1:] - cw[..., :-1]), take(ch)
    hk, d0, d1 = take(ch[..., 1:] - ch[..., :-1]), take(derivs[..., :-1]), take(derivs[..., 1:])
    s = hk / wk
    if inverse:
        dy = xc - y0
        a = hk * (s - d0) + dy * (d0 + d1 - 2 * s)
        b = hk * d0 - dy * (d0 + d1 - 2 * s)
        c = -s * dy
        xi = (2 * c / (-b - torch.sqrt((b ** 2 - 4 * a * c).clamp(min=0)))).clamp(0, 1)
        out = x0 + xi * wk
    else:
        xi = (xc - x0) / wk
        out = y0 + hk * (s * xi ** 2 + d0 * xi * (1 - xi)) / (s + (d0 + d1 - 2 * s) * xi * (1 - xi))
    denom = s + (d0 + d1 - 2 * s) * xi * (1 - xi)
    logdet = 2 * torch.log(s) + torch.log(d1 * xi ** 2 + 2 * s * xi * (1 - xi) + d0 * (1 - xi) ** 2) - 2 * torch.log(denom)
    if inverse:
        logdet = -logdet
    return torch.where(inside, out, x), torch.where(inside, logdet, torch.zeros_like(logdet))


class SplineNoiseFlow(NoiseFlow):
    """Autoregressive neural spline flow of the deltas over TARGETS: factor j maps asinh(delta_j / ASINH_SCALE)
    to a standard normal by a rational-quadratic spline whose knots come from the same inputs as in
    NoiseFlow (formula context, calculated %, present flags, previous deltas and observed flags)."""

    def __init__(self, num_features: int = NUM_FEATURES, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__(num_features, hidden, dropout)
        n = len(TARGETS)
        self.heads = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Linear(hidden + 1 + n + 2 * j, hidden // 2), torch.nn.GELU(),
                torch.nn.Linear(hidden // 2, hidden // 2), torch.nn.GELU(),
                torch.nn.Linear(hidden // 2, 3 * SPLINE_BINS - 1),
            )
            for j in range(n)
        )

    def _params(self, h, calc, present, delta, observed, j):
        x = torch.cat([h, calc[:, j:j + 1] / 10.0, present, delta[:, :j] * observed[:, :j], observed[:, :j]], dim=-1)
        return self.heads[j](x)

    def log_prob(self, feats, calc, present, delta, observed) -> torch.Tensor:
        h = self.context(feats)
        out = []
        for j in range(len(TARGETS)):
            params = self._params(h, calc, present, delta, observed, j)
            u = torch.asinh(delta[:, j] / ASINH_SCALE)
            z, logdet = rational_quadratic_spline(u, params)
            log_du = -np.log(ASINH_SCALE) - 0.5 * torch.log1p((delta[:, j] / ASINH_SCALE) ** 2)
            lp = -0.5 * z ** 2 - 0.5 * np.log(2 * np.pi) + logdet + log_du
            out.append(lp * observed[:, j])
        return torch.stack(out, -1)

    @torch.no_grad()
    def sample(self, feats, calc, present, generator=None, temperature: float = 1.0) -> torch.Tensor:
        h = self.context(feats)
        delta = torch.zeros_like(calc)
        for j in range(len(TARGETS)):
            params = self._params(h, calc, present, delta, present, j)
            z = temperature * torch.randn(len(calc), generator=generator, device=calc.device)
            u, _ = rational_quadratic_spline(z, params, inverse=True)
            delta[:, j] = ASINH_SCALE * torch.sinh(u) * present[:, j]
        return delta


MODELS = {"mixture": NoiseFlow, "spline": SplineNoiseFlow}


def target_arrays(counts: typing.Mapping[str, float]) -> typing.Tuple[np.ndarray, np.ndarray]:
    percent = mass_percent(counts)
    calc = np.array([percent.get(e, 0.0) for e in TARGETS], dtype=np.float32)
    present = np.array([float(counts.get(e, 0.0) > 0) for e in TARGETS], dtype=np.float32)
    return calc, present


class ElementalAnalysisFlow:
    """Samples Found % of C, H, N, S for SMILES with the model saved by ea/train_noise_flow.py."""

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model = MODELS[checkpoint.get("model", "mixture")](**checkpoint["config"]).to(device).eval()
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
        with_other: bool = False,
    ) -> np.ndarray:
        """[len(smiles), num_samples, len(TARGETS)] Found % (two decimals). Elements absent from a
        molecule get the Found value of a measured absent element ("N, 0.00"); NaN if the SMILES
        cannot be parsed. Which elements count as measured is up to the caller. with_other adds a
        last column Other = 100 - C - H - N - S, so the five sum to 100 %; it is slightly negative
        when the four exceed 100 %, as in about 1 % of real analyses of compounds made only of C, H, N, S."""
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
        if with_other:
            out = np.concatenate([out, np.round(100.0 - out.sum(-1, keepdims=True), 2)], -1)
        return out
