"""
Helpers for test_analysis.ipynb.

Every run (a model, or a model with a different set of input modalities) is loaded into one
DataFrame with a row per test case, so all tables and plots are grouped by `run` and
`modality` and new runs are compared by just adding them to RUNS in the notebook.
"""
import gzip
import json
import math
import multiprocessing
import typing
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from scipy import stats

RDLogger.DisableLog("rdApp.*")

MODALITIES = ("Only C", "Only H", "Both")
COMMON_ELEMENTS = {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I", "Si", "B"}
ORGANIC_ELEMENTS = COMMON_ELEMENTS | {"Se", "Te", "Ge", "As", "Sb", "Bi", "Sn"}
CATEGORY_ELEMENTS = ("N", "O", "S", "P", "F", "Cl", "Br", "I", "Si", "B", "Se")
# Paper (SI, section C): Morgan fingerprints, radius 4, 4096 bits.
FINGERPRINT = rdFingerprintGenerator.GetMorganGenerator(radius=4, fpSize=4096)


# --------------------------------------------------------------------------- SMILES


def canonical_pair(smiles: str) -> typing.Tuple[typing.Optional[str], typing.Optional[str]]:
    """(canonical isomeric SMILES, canonical SMILES without stereo) or (None, None) if invalid."""
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None, None
    return Chem.MolToSmiles(mol), Chem.MolToSmiles(mol, isomericSmiles=False)


def canonicalize_all(
    smiles: typing.Iterable[str], workers: int = 0
) -> typing.Dict[str, typing.Tuple[typing.Optional[str], typing.Optional[str]]]:
    unique = sorted(set(smiles))
    workers = workers or multiprocessing.cpu_count()
    if len(unique) < 20_000 or workers <= 1:
        return {s: canonical_pair(s) for s in unique}
    with multiprocessing.Pool(workers) as pool:
        return dict(zip(unique, pool.map(canonical_pair, unique, chunksize=2000)))


# --------------------------------------------------------------------------- loading


def modality_of(record: dict) -> str:
    if record["has_carbon"] and record["has_hydrogen"]:
        return "Both"
    return "Only C" if record["has_carbon"] else "Only H"


def _spectrum_columns(record: dict) -> dict:
    carbon, hydrogen = record.get("carbon"), record.get("hydrogen")
    return {
        "c_signals": len(carbon["spectrum"]) if carbon else np.nan,
        "h_signals": len(hydrogen["spectrum"]) if hydrogen else np.nan,
        "h_integral": float(sum(hydrogen["intensity"])) if hydrogen else np.nan,
    }


def _merge_candidates(
    candidates: typing.List[typing.Tuple[str, float]], canon: dict
) -> typing.List[typing.Tuple[str, typing.Optional[str], float]]:
    """Merge candidates equal after canonicalization (max score); sorted by score."""
    merged: typing.Dict[str, typing.Tuple[typing.Optional[str], float]] = {}
    for smiles, score in candidates:
        iso, flat = canon[smiles]
        key = iso if iso is not None else smiles
        if key not in merged or score > merged[key][1]:
            merged[key] = (flat, score)
    return sorted(((k, f, s) for k, (f, s) in merged.items()), key=lambda c: -c[2])


def load_run(
    name: str,
    top1_path: typing.Optional[str] = None,
    topn_path: typing.Optional[str] = None,
    top_n: int = 10,
    workers: int = 0,
) -> pd.DataFrame:
    """
    Load evaluate.py (greedy) and/or evaluate_top_n.py output of one run.

    Columns: top1_* from greedy decoding; topn_* from sampling, where candidates are
    re-canonicalized, merged, and ranked by score (sum of token log-probabilities).
    """
    top1 = json.load(open(top1_path))["predictions"] if top1_path else None
    topn = json.load(open(topn_path))["predictions"] if topn_path else None
    base = top1 if top1 is not None else topn
    if base is None:
        raise ValueError("Give at least one of top1_path / topn_path")
    if top1 is not None and topn is not None:
        if len(top1) != len(topn) or any(
            a["true_smiles"] != b["true_smiles"] for a, b in zip(top1, topn)
        ):
            raise ValueError(f"{name}: greedy and top-N files are not aligned (different data?)")

    to_canon = [r["true_smiles"] for r in base]
    if top1 is not None:
        to_canon += [r["prediction"] for r in top1]
    if topn is not None:
        to_canon += [s for r in topn for s, _ in r["all_candidates"]]
    canon = canonicalize_all(to_canon, workers)

    rows = []
    for i, record in enumerate(base):
        true_iso, true_flat = canon[record["true_smiles"]]
        row = {
            "run": name,
            "case": i,
            "true_smiles": true_iso or record["true_smiles"],
            "true_flat": true_flat,
            "modality": modality_of(record),
            **_spectrum_columns(record),
        }
        if top1 is not None:
            iso, flat = canon[top1[i]["prediction"]]
            row.update(
                top1_pred=iso or top1[i]["prediction"],
                top1_valid=iso is not None,
                top1_exact=iso is not None and iso == true_iso,
                top1_topo=flat is not None and flat == true_flat,
            )
        if topn is not None:
            cands = _merge_candidates(topn[i]["all_candidates"], canon)
            top = cands[:top_n]
            scores = np.array([c[2] for c in cands], dtype=float)
            row.update(
                topn_candidates=cands,
                topn_unique=len(cands),
                topn_valid=any(c[1] is not None for c in top),
                topn_exact=any(c[0] == true_iso for c in top),
                topn_topo=any(c[1] is not None and c[1] == true_flat for c in top),
                topn_rank=next((k + 1 for k, c in enumerate(cands) if c[0] == true_iso), np.nan),
                first_pred=cands[0][0] if cands else None,
                first_exact=bool(cands) and cands[0][0] == true_iso,
                first_topo=bool(cands) and cands[0][1] is not None and cands[0][1] == true_flat,
                first_valid=bool(cands) and cands[0][1] is not None,
                score1=scores[0] if len(scores) else np.nan,
                score2=scores[1] if len(scores) > 1 else np.nan,
                # Gap between the best and the second-best candidate; inf when only one exists.
                margin=(scores[0] - scores[1]) if len(scores) > 1 else (np.inf if len(scores) else np.nan),
                # Share of the best candidate in the probability mass of all sampled candidates.
                p_first=float(np.exp(scores[0] - _logsumexp(scores))) if len(scores) else np.nan,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _logsumexp(x: np.ndarray) -> float:
    m = np.max(x)
    return float(m + np.log(np.sum(np.exp(x - m))))


def add_token_lengths(df: pd.DataFrame, tokenizer_path: str) -> pd.DataFrame:
    """Length-normalized score of the first candidate (score per SMILES token incl. EOS)."""
    try:
        import sentencepiece as spm
    except ImportError:
        print("sentencepiece is not installed: score1_per_token is skipped")
        return df.assign(score1_per_token=np.nan)

    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    lengths = df["first_pred"].map(lambda s: len(sp.encode(s)) + 1 if isinstance(s, str) else np.nan)
    return df.assign(score1_per_token=df["score1"] / lengths)


# --------------------------------------------------------------------------- molecules


def molecule_descriptors(smiles: str) -> dict:
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"smiles": smiles}
    mol_h = Chem.AddHs(mol)
    elements = Counter(a.GetSymbol() for a in mol_h.GetAtoms())
    carbons = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C"]
    # Symmetry classes: atoms with equal rank (no tie breaking) are topologically equivalent.
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    exchangeable = sum(
        a.GetTotalNumHs() for a in mol.GetAtoms() if a.GetSymbol() in ("O", "N", "S")
    )
    stereocenters = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    return {
        "smiles": smiles,
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "n_carbon": len(carbons),
        "n_unique_carbon": len({ranks[i] for i in carbons}),
        "n_hydrogen": elements.get("H", 0),
        "n_exchangeable_h": exchangeable,
        "mol_weight": Descriptors.MolWt(mol),
        "n_rings": rdMolDescriptors.CalcNumRings(mol),
        "n_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "n_stereocenters": len(stereocenters),
        "n_specified_stereo": sum(1 for _, label in stereocenters if label != "?"),
        "has_double_bond_stereo": any(
            b.GetStereo() != Chem.BondStereo.STEREONONE for b in mol.GetBonds()
        ),
        "n_species": len(Chem.GetMolFrags(mol)),
        "charged": any(a.GetFormalCharge() != 0 for a in mol.GetAtoms()),
        "elements": " ".join(sorted(elements)),
        "n_heteroatoms": sum(v for k, v in elements.items() if k not in ("C", "H")),
        **{f"n_{e}": elements.get(e, 0) for e in CATEGORY_ELEMENTS},
        "has_metal": any(_is_metal(e) for e in elements),
        "only_common_elements": set(elements) <= COMMON_ELEMENTS,
        "has_rare_element": not set(elements) <= ORGANIC_ELEMENTS,
    }


def _is_metal(symbol: str) -> bool:
    return symbol not in ORGANIC_ELEMENTS | {"He", "Ne", "Ar", "Kr", "Xe"}


def descriptor_table(smiles: typing.Iterable[str], workers: int = 0) -> pd.DataFrame:
    unique = sorted(set(smiles))
    workers = workers or multiprocessing.cpu_count()
    if len(unique) < 5_000 or workers <= 1:
        rows = [molecule_descriptors(s) for s in unique]
    else:
        with multiprocessing.Pool(workers) as pool:
            rows = pool.map(molecule_descriptors, unique, chunksize=500)
    return pd.DataFrame(rows).set_index("smiles")


def add_descriptors(df: pd.DataFrame, descriptors: pd.DataFrame) -> pd.DataFrame:
    out = df.join(descriptors, on="true_smiles")
    # Paper (SI, section E): observed - expected 13C signals, expected = number of carbons.
    out["c_signal_dev"] = out["c_signals"] - out["n_carbon"]
    out["c_signal_dev_sym"] = out["c_signals"] - out["n_unique_carbon"]
    out["h_integral_dev"] = out["h_integral"] - out["n_hydrogen"]
    c_ok = out["c_signals"].isna() | (out["c_signals"] <= out["n_carbon"])
    h_ok = out["h_integral"].isna() | (
        (out["h_integral"] >= out["n_hydrogen"] - out["n_exchangeable_h"] - 0.5)
        & (out["h_integral"] <= out["n_hydrogen"] + 0.5)
    )
    # "Ideal" subset of the paper: <=30 heavy atoms, consistent signal counts,
    # common heteroatoms only, <=1 stereocentre, a single species.
    out["ideal"] = (
        (out["heavy_atoms"] <= 30)
        & c_ok
        & h_ok
        & out["only_common_elements"].astype(bool)
        & (out["n_stereocenters"] <= 1)
        & (out["n_species"] == 1)
    )
    return out


def element_categories(df: pd.DataFrame) -> typing.Dict[str, pd.Series]:
    """Overlapping subsets used for accuracy by elemental composition (paper, Fig. 3C)."""
    elements = df["elements"].fillna("").str.split()
    cats = {
        "C,H only": elements.map(lambda e: set(e) <= {"C", "H"}),
        "C,H,O only": elements.map(lambda e: set(e) <= {"C", "H", "O"} and "O" in e),
        "C,H,N,O only": elements.map(lambda e: set(e) <= {"C", "H", "N", "O"} and "N" in e),
    }
    for e in CATEGORY_ELEMENTS:
        cats[f"contains {e}"] = df[f"n_{e}"] > 0
    cats["contains halogen"] = df[["n_F", "n_Cl", "n_Br", "n_I"]].sum(axis=1) > 0
    cats["rare element"] = df["has_rare_element"].astype(bool)
    cats["metal"] = df["has_metal"].astype(bool)
    cats["charged"] = df["charged"].astype(bool)
    cats["multiple species"] = df["n_species"] > 1
    return cats


# --------------------------------------------------------------------------- test set file


def read_test_records(path: str) -> pd.DataFrame:
    """One row per spectrum of the processed JSONL split."""
    rows = []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            peaks = r["spectrum"]
            rows.append(
                {
                    "smiles": r["smiles"],
                    "spectrum_type": r["spectrum_type"],
                    "n_signals": len(peaks),
                    "integral": sum(p["intensity"] for p in peaks),
                    "shift_min": min(p["signal"] for p in peaks),
                    "shift_max": max(p["signal"] for p in peaks),
                    "multiplicities": [p["multiplicity"] for p in peaks],
                    "n_couplings": [len(p.get("couplings", [])) for p in peaks],
                    "couplings": [j for p in peaks for j in p.get("couplings", [])],
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- statistics


def wilson(k: float, n: float, z: float = 1.96) -> typing.Tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def rate_table(
    df: pd.DataFrame, target: str, by: typing.Union[str, typing.List[str]], min_count: int = 1
) -> pd.DataFrame:
    """Mean of a boolean column per group, with Wilson 95% CI."""
    by = [by] if isinstance(by, str) else list(by)
    g = df.groupby(by, observed=True)[target].agg(["sum", "count"]).reset_index()
    g = g[g["count"] >= min_count]
    g["rate"] = g["sum"] / g["count"]
    ci = [wilson(k, n) for k, n in zip(g["sum"], g["count"])]
    g["ci_low"] = [c[0] for c in ci]
    g["ci_high"] = [c[1] for c in ci]
    return g


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Paper Tables S5/S6-style metrics per run and modality (+ Overall)."""
    metrics = [
        c
        for c in (
            "top1_valid", "top1_exact", "top1_topo",
            "topn_valid", "topn_exact", "topn_topo", "first_exact", "topn_unique",
        )
        if c in df
    ]
    overall = df.assign(modality="Overall")
    g = pd.concat([df, overall]).groupby(["run", "modality"])
    table = g[metrics].mean()
    table.insert(0, "count", g.size())
    order = [m for m in (*MODALITIES, "Overall") if m in table.index.get_level_values(1)]
    return table.reindex(order, level=1)


def mcnemar(a: pd.Series, b: pd.Series) -> dict:
    """Paired McNemar test (continuity corrected) for two boolean Series on the same cases."""
    only_a = int((a & ~b).sum())
    only_b = int((b & ~a).sum())
    n = only_a + only_b
    chi2 = (abs(only_a - only_b) - 1) ** 2 / n if n else 0.0
    return {
        "acc_a": a.mean(),
        "acc_b": b.mean(),
        "diff": a.mean() - b.mean(),
        "only_a": only_a,
        "only_b": only_b,
        "p_value": stats.chi2.sf(chi2, 1) if n else 1.0,
    }


def tanimoto(a: str, b: str) -> float:
    ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
    if ma is None or mb is None:
        return np.nan
    return DataStructs.TanimotoSimilarity(FINGERPRINT.GetFingerprint(ma), FINGERPRINT.GetFingerprint(mb))


def stereo_variants(row: pd.Series, top_n: int = 10) -> dict:
    """Among top-N candidates with the true skeleton: how many stereoisomers, is the right one first."""
    same = [c for c in row["topn_candidates"][:top_n] if c[1] is not None and c[1] == row["true_flat"]]
    return {
        "n_stereo_variants": len(same),
        "correct_stereo_first": bool(same) and same[0][0] == row["true_smiles"],
    }


# --------------------------------------------------------------------------- uncertainty


def roc_auc(score: pd.Series, label: pd.Series) -> float:
    """AUC = P(score of a correct case > score of a wrong one); NaN scores are dropped."""
    mask = score.notna() & label.notna()
    s, y = score[mask].to_numpy(dtype=float), label[mask].to_numpy(dtype=bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = stats.rankdata(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def expected_calibration_error(prob: pd.Series, label: pd.Series, bins: int = 10) -> float:
    mask = prob.notna()
    p, y = prob[mask].to_numpy(), label[mask].to_numpy(dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def risk_coverage(confidence: pd.Series, correct: pd.Series) -> pd.DataFrame:
    """Accuracy of the most confident fraction of cases (selective prediction)."""
    order = np.argsort(-confidence.to_numpy(dtype=float), kind="stable")
    hits = correct.to_numpy(dtype=float)[order]
    n = np.arange(1, len(hits) + 1)
    return pd.DataFrame({"coverage": n / len(hits), "accuracy": np.cumsum(hits) / n})


def confidence_report(df: pd.DataFrame, signals: typing.Dict[str, str], target: str = "first_exact") -> pd.DataFrame:
    """AUC and Spearman correlation of each confidence signal with correctness, per run/modality."""
    rows = []
    for (run, modality), g in pd.concat([df, df.assign(modality="Overall")]).groupby(["run", "modality"]):
        for name, column in signals.items():
            s = g[column].replace(np.inf, np.nan)
            if column == "margin":  # a single unique candidate is the most confident case
                s = g[column].replace(np.inf, g[column][np.isfinite(g[column])].max() + 1)
            mask = s.notna()
            rho = stats.spearmanr(s[mask], g.loc[mask, target]).statistic if mask.sum() > 2 else np.nan
            rows.append({"run": run, "modality": modality, "signal": name, "auc": roc_auc(s, g[target]), "spearman": rho, "n": int(mask.sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- plotting

# Human-readable axis labels for DataFrame columns.
LABELS = {
    "top1_exact": "доля точных совпадений, top-1 (жадно)",
    "top1_topo": "доля совпадений без стереохимии, top-1",
    "top1_valid": "доля валидных SMILES, top-1",
    "topn_exact": "доля точных совпадений, top-10",
    "topn_topo": "доля совпадений без стереохимии, top-10",
    "topn_valid": "доля примеров с валидным кандидатом, top-10",
    "first_exact": "доля верных лучших по log-likelihood кандидатов",
    "first_topo": "доля лучших кандидатов с верным скелетом",
    "topn_unique": "уникальных кандидатов из 10 сэмплов",
    "n_carbon": "число атомов углерода в молекуле",
    "heavy_atoms": "число тяжёлых (не H) атомов",
    "mw_bin": "молекулярная масса, Да (интервалы)",
    "mol_weight": "молекулярная масса, Да",
    "n_heteroatoms": "число гетероатомов (не C и не H)",
    "n_rings": "число циклов",
    "n_aromatic_rings": "число ароматических циклов",
    "n_stereocenters": "число стереоцентров",
    "n_species": "число частиц в SMILES (через '.')",
    "c_signal_dev": "сигналов 13C − атомов C",
    "c_signal_dev_sym": "сигналов 13C − неэквивалентных атомов C",
    "h_dev_bin": "сумма интегралов 1H − атомов H (округлено)",
    "h_integral_bin": "сумма интегралов 1H (интервалы по 5)",
    "c_signals": "число сигналов в спектре 13C",
    "h_signals": "число сигналов в спектре 1H",
    "margin_bin": "score₁ − score₂: разница log-likelihood 1-го и 2-го кандидатов",
}

MODALITY_COLORS = {"Only C": "#1f77b4", "Only H": "#d62728", "Both": "#2ca02c", "Overall": "#444444"}
RUN_STYLES = ("-", "--", ":", "-.")


def plot_rate(
    df: pd.DataFrame,
    target: str,
    x: str,
    ax,
    min_count: int = 30,
    title: str = "",
    ylabel: typing.Optional[str] = None,
    by_modality: bool = True,
) -> pd.DataFrame:
    """Line per run (line style) and modality (color) with Wilson 95% CI band."""
    keys = ["run", "modality", x] if by_modality else ["run", x]
    table = rate_table(df, target, keys, min_count=min_count)
    for r_idx, run in enumerate(table["run"].unique()):
        groups = table[table["run"] == run].groupby("modality") if by_modality else [("Overall", table[table["run"] == run])]
        for modality, g in groups:
            xs = g[x].astype(str) if isinstance(g[x].dtype, pd.CategoricalDtype) or g[x].dtype == object else g[x]
            color = MODALITY_COLORS.get(modality)
            ax.plot(xs, g["rate"], RUN_STYLES[r_idx % len(RUN_STYLES)], marker="o", ms=3, color=color, label=f"{run} | {modality}")
            ax.fill_between(xs, g["ci_low"], g["ci_high"], color=color, alpha=0.15)
    ax.set_xlabel(LABELS.get(x, x))
    ax.set_ylabel(ylabel or LABELS.get(target, target))
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return table


def plot_counts(df: pd.DataFrame, x: str, ax, title: str = "", logy: bool = False) -> None:
    for modality, g in df.groupby("modality"):
        counts = g[x].value_counts().sort_index()
        xs = counts.index.astype(str) if isinstance(counts.index, pd.CategoricalIndex) else counts.index
        ax.plot(xs, counts.values, marker=".", color=MODALITY_COLORS.get(modality), label=modality)
    ax.set_title(title or f"count by {x}")
    ax.set_xlabel(x)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
