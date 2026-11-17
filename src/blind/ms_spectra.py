"""
MS2 spectra as model input: normalization of peaks and of the acquisition metadata.

Shared by prepare_ms.py (which writes data/ms/processed/<split>.npz) and modalities/ms (which
turns stored spectra into tensors). The vocabularies are fixed here, so the model knows their sizes.
"""
import re
import typing

import numpy as np

MAX_PEAKS = 128  # most intense peaks kept per spectrum (95 % of MS2 spectra have fewer than 141)
PRECURSOR_MARGIN = 2.0  # peaks above precursor m/z + this are noise or another ion

ADDUCTS = (
    "<unk>", "<other>", "M+H", "M-H", "M+Na", "M+NH4", "M+H-H2O", "M+HCOO", "M+Cl", "M", "M+H-2H2O",
    "2M-H", "M+K", "2M+H", "M+H+K", "2M-2H+Na", "2M+Na", "M+CH3COO", "M+H+Na", "M+2H", "M-H2O-H",
    "M+Li", "M+H-NH3", "M-2H", "M-H-CO2", "M+2Na-H", "M+ACN+H", "M-CH3", "M+Br",
)
ADDUCT_SYNONYMS = {
    "M-H2O+H": "M+H-H2O", "M-2H2O+H": "M+H-2H2O", "M+FA": "M+HCOO", "M+FA-H": "M+HCOO",
    "M+formate": "M+HCOO", "M+HCOO-": "M+HCOO", "M+HAc-H": "M+CH3COO", "M+Hac-H": "M+CH3COO",
    "M+OAc": "M+CH3COO", "M+H2O-H": "M-H2O-H", "M+NH3": "M+NH4", "M+": "M", "M-": "M",
}
POLARITIES = ("<unk>", "Positive", "Negative")
IONIZATIONS = ("<unk>", "<other>", "ESI", "APCI", "APPI", "EI", "CI", "MALDI", "FAB")
# Instrument classes by keyword, first match wins ("itft" is an Orbitrap/FT hybrid, "ittof" a TOF)
INSTRUMENT_RULES = (
    ("orbitrap_ft", ("orbitrap", "exactive", "qehf", "itft", "qft", "hybrid ft", "-ft", " ft", "fticr", "ft-icr")),
    ("tof", ("tof", "maxis", "impact", "synapt", "tripletof")),
    ("qqq", ("qqq", "-qq", "quattro", "triple", "qtrap")),
    ("ion_trap", ("qit", "-it", "ion trap", "lcq", "ltq", "iontrap")),
    ("sector", ("-b", "sector")),
)
INSTRUMENTS = ("<unk>", "<other>") + tuple(name for name, _ in INSTRUMENT_RULES)
CHARGES = ("<unk>", "1", "2", "3+")
META_FIELDS = ("adduct", "polarity", "ionization", "instrument", "charge")
META_VOCABS = {"adduct": ADDUCTS, "polarity": POLARITIES, "ionization": IONIZATIONS, "instrument": INSTRUMENTS, "charge": CHARGES}


def is_simulated(record: dict) -> bool:
    return "silico" in str(record.get("instrument") or "").lower() or record.get("spectrum_kind") == "in_silico"


def normalize_adduct(adduct: typing.Optional[str]) -> int:
    if not adduct or adduct.strip().lower() in ("unknown", "n/a", "none"):
        return 0
    text = adduct.strip().replace(" ", "")
    match = re.fullmatch(r"\[?([^\[\]]+?)\]?(\d*[+-])?", text)
    name = match.group(1) if match else text
    name = ADDUCT_SYNONYMS.get(name, name)
    return ADDUCTS.index(name) if name in ADDUCTS else 1


def normalize_polarity(record: dict) -> int:
    polarity = record.get("polarity")
    if polarity in POLARITIES:
        return POLARITIES.index(polarity)
    adduct = (record.get("adduct") or "").strip()
    if adduct.endswith("+"):
        return 1
    if adduct.endswith("-"):
        return 2
    return 0


def normalize_ionization(record: dict) -> int:
    method = record.get("ionization_method")
    if method in IONIZATIONS:
        return IONIZATIONS.index(method)
    instrument = str(record.get("instrument") or "").upper()
    for name in ("APCI", "APPI", "ESI", "MALDI", "FAB", "EI", "CI"):
        if re.search(rf"(^|[^A-Z]){name}([^A-Z]|$)", instrument):
            return IONIZATIONS.index(name)
    return 1 if method else 0


def normalize_instrument(instrument: typing.Optional[str]) -> int:
    text = str(instrument or "").strip().lower()
    if not text or text in ("n/a", "none", "unknown"):
        return 0
    for name, keywords in INSTRUMENT_RULES:
        if any(k in text for k in keywords):
            return INSTRUMENTS.index(name)
    return 1


def normalize_charge(charge: typing.Any) -> int:
    try:
        value = abs(int(charge))
    except (TypeError, ValueError):
        return 0
    if value == 0:
        return 0
    return min(value, 3)


def clean_peaks(peaks: typing.Sequence, precursor_mz: typing.Optional[float]) -> typing.Optional[np.ndarray]:
    """[n, 2] m/z and intensity (max 1), the MAX_PEAKS most intense, sorted by m/z; None if empty."""
    arr = np.asarray(peaks, dtype=np.float64).reshape(-1, 2) if len(peaks) else np.zeros((0, 2))
    keep = np.isfinite(arr).all(1) & (arr[:, 0] > 0) & (arr[:, 1] > 0)
    if precursor_mz:
        keep &= arr[:, 0] <= precursor_mz + PRECURSOR_MARGIN
    arr = arr[keep]
    if not len(arr):
        return None
    if len(arr) > MAX_PEAKS:
        arr = arr[np.argsort(-arr[:, 1], kind="stable")[:MAX_PEAKS]]
    arr = arr[np.argsort(arr[:, 0], kind="stable")]
    arr[:, 1] /= arr[:, 1].max()
    return arr.astype(np.float32)
