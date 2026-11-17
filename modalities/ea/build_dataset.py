#!/usr/bin/env python3
"""
Dataset of experimental elemental analyses with structures: SMILES, molecular formula, Calcd and Found.

Reads the PubMed Central full texts downloaded by modalities/ea/download_corpus.py (any number of
<dir>/raw/*.xml.gz folders) and the supporting information texts from modalities/ea/download_si.py (<dir>/si/*.txt.gz), parses every "Anal. Calcd for <formula>: ... Found: ..." statement
and looks for the name of the compound in the text before it (start of the paragraph, bold text,
section title, bracket formulas such as [Cu(phen)2Cl]ClO4). Names are turned into SMILES by
  - modalities/ea/ligands.py for coordination formulas and additive names of complexes,
  - OPSIN (systematic names; needs java and the OPSIN jar),
  - PubChem (optional, trivial names),
  - LLM translations of the names left (llm_smiles.json in the output dir, see modalities/ea/llm_smiles.py),
and a SMILES is kept only if its element composition equals the formula of the statement.

Run from the repository root:
    python modalities/ea/build_dataset.py --input-dirs data/ea data/ea/metals data/ea/complexes data/ea/general \
        --opsin-jar data/ea/tools/opsin.jar --output-dir data/ea/dataset
"""
import argparse
import collections
import csv
import glob
import gzip
import hashlib
import itertools
import json
import multiprocessing
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import typing
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rdkit import Chem, RDLogger
from tqdm.auto import tqdm

from blind.elemental_analysis import formula_counts, formula_counts_plain
from blind.ea_statements import find_statements, matches_formula
from ligands import LIGAND_NAMES, LIGANDS, METAL_NAMES, METALS

RDLogger.DisableLog("rdApp.*")

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/IsomericSMILES,SMILES/JSON"
COUNTER_ANION_NAMES = {
    "hexafluorophosphate": "PF6", "tetrafluoroborate": "BF4", "perchlorate": "ClO4", "chloride": "Cl", "bromide": "Br",
    "iodide": "I", "nitrate": "NO3", "triflate": "OTf", "trifluoromethanesulfonate": "OTf", "tetraphenylborate": "BPh4",
    "hexafluoroantimonate": "SbF6", "acetate": "OAc", "sulfate": "SO4",
}
MULTIPLIERS = {"di": 2, "tri": 3, "tetra": 4, "penta": 5, "hexa": 6, "bis": 2, "tris": 3, "tetrakis": 4, "pentakis": 5, "hexakis": 6}
LABEL = r"(?:\d+\s?[a-z]{0,2}|[IVX]+[a-z]?|[a-z]|\d+[a-z]?[-–,]\d*[a-z]?|H?\d?L\d*[a-z]?|[A-Z]{1,2}\d+[a-z]?)"
# A label follows a space: "... ethanone (3a)", unlike the locants of "benzo[d]imidazole"
LABEL_RE = re.compile(rf"\s+[\(\[]\s*(?:compound\s*|ligand\s*|complex\s*)?({LABEL})\s*[\)\]]", re.I)
TRAILING_LABEL_RE = re.compile(rf"\s+(?:compound\s+|ligand\s+|complex\s+)?({LABEL})$")
LEAD_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+|(?:general\s+)?(?:synthesis|preparation|procedure|synthetic procedure)s?\s+(?:of|for)\s+"
    r"(?:the\s+)?(?:compounds?\s+|complex(?:es)?\s+|ligands?\s+)?)+",
    re.I,
)
SPLIT_RE = re.compile(r":|\.\s+(?=[A-Z])|;\s|\s(?:was|were|is)\s|,\s*(?:yield|obtained|prepared|synthesi[sz]ed|following)", re.I)


# ----------------------------------------------------------------------------- statements and names
def normalize_name(text: str) -> str:
    """Undo the spacing that italics and sub/superscripts leave in names: '1 H -pyrazol', '[3,4- b ]'."""
    text = re.sub(r"[­-]", "", text)
    for dash in "−–‐‑":
        text = text.replace(dash, "-")
    text = text.replace("′", "'").replace("″", "''")
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"(?<=\d)\s+(?=[A-Z]\b)", "", text)
    text = re.sub(r"([\[(])\s+", r"\1", text)
    text = re.sub(r"\s+([\])])", r"\1", text)
    text = re.sub(r"(?<=[\])])\s+(?=[a-z])", "", text)  # "piperazin-1-yl) quinoline"
    text = re.sub(r"\s*,\s*(?=\S)", ",", text) if not re.search(r"[a-z]{3,},\s[a-z]{3,}", text) else text
    return re.sub(r"\s+", " ", text).strip()


def split_head(text: str) -> str:
    """Text up to the first separator outside brackets."""
    depth = 0
    for i, ch in enumerate(text):
        depth += ch in "([{"
        depth -= ch in ")]}"
        if depth == 0 and SPLIT_RE.match(text, i):
            return text[:i]
    return text


def _clean_name(text: str) -> typing.Tuple[typing.Optional[str], typing.Optional[str]]:
    """(name, label of the compound such as 3a or L1)."""
    text = LEAD_RE.sub("", normalize_name(text))
    label = None
    match = LABEL_RE.search(text)
    if match:
        label, text = match.group(1), text[: match.start()]
    text = re.sub(r"\s*\(\s*(?:yield|\d+\s*%)[^)]*\)\s*$", "", text, flags=re.I).strip(" .,;:-")
    match = TRAILING_LABEL_RE.search(text)
    if match:
        label, text = label or match.group(1), text[: match.start()]
    text = text.strip(" .,;:-")
    if re.fullmatch(LABEL, text):
        return None, label or text
    if not 3 <= len(text) <= 250 or len(text.split()) > 12:
        return None, label
    if re.match(r"(?i)^(yield|mp|m\.p\.|anal|elemental|ir|1h|13c|hrms|ms|esi|white|yellow|colorless|colourless|"
                r"orange|red|green|brown|black|pale|the|a|an|to|in|after|then|this|it|general|materials)\b", text):
        return None, label
    return text, label


def name_candidates(segment: str, bold: typing.List[str], title: typing.Optional[str]) -> typing.Tuple[typing.List[str], typing.List[str]]:
    head = segment.strip()
    candidates = [split_head(head), head[:250]] + bold + ([split_head(title)] if title else [])
    names, labels = [], []
    for c in candidates:
        name, label = _clean_name(c)
        if name and name not in names:
            names.append(name)
        if label and label not in labels:
            labels.append(label)
    return names, labels


def bracket_formulas(segment: str, title: typing.Optional[str]) -> typing.List[str]:
    text = (title or "") + " " + segment
    found = re.findall(r"(?:\([A-Za-z0-9]+\)\d*|[A-Z][a-z]?\d*)?\[[^\[\]\s]*(?:\[[^\[\]\s]*\][^\[\]\s]*)*\][A-Za-z0-9()]*", text)
    return list(dict.fromkeys(f.strip(".,;:") for f in found if len(f) <= 120))


CLASS_WORDS = r"(?:acid|ester|amide|anhydride|salt|hydrochloride|dihydrochloride|hydrobromide|trifluoroacetate|oxide|hydrate|monohydrate|dihydrate|chloride|bromide|iodide|ether|ketone|alcohol|lactone|oxime|hydrazone|methiodide|mesylate|tosylate|oxalate|fumarate|maleate|citrate|tartrate|sulfate|nitrate|perchlorate|acetate|formate)"
INVERT_RE = re.compile(rf"^((?:mono|di|tri)?(?:methyl|ethyl|propyl|butyl|benzyl|tert-butyl)?\s*{CLASS_WORDS}(?:\s+{CLASS_WORDS})?)\s+of\s+(.+)$", re.I)


def name_variants(name: str) -> typing.List[str]:
    """The name and cleaned forms of it that OPSIN may parse; the formula check decides."""
    out = [name]
    def add(v):
        v = v.strip(" .,;:-")
        if len(v) >= 3 and v not in out:
            out.append(v)
    v = re.sub(r"^(?:\d+\.){1,4}\d*\.?\s*", "", name)  # 2.2.4.N-[...]: section numbers
    add(v)
    for base in list(out):
        add(re.sub(r"\s*[\(\[\{][^()\[\]{}]*(?:[\(\[\{][^()\[\]{}]*[\)\]\}][^()\[\]{}]*)*[\)\]\}]\s*$", "", base))  # trailing (6bc'), (9{6,16}), ((E)-4c)
        add(re.sub(r",\s*[^,\s]{1,8}(?:\s*\([^()]*\))?$", "", base))  # ",20d (R = Ph)"
        m = INVERT_RE.match(base)
        if m:
            add(f"{m.group(2)} {m.group(1)}")  # Hydrochloride of X -> X hydrochloride
    for base in list(out):
        # "benzene acetic acid" -> "benzeneacetic acid": join words except before class words
        add(re.sub(rf"(?<=[A-Za-z0-9\)\]])\s+(?!{CLASS_WORDS}\b)(?=[a-z0-9\(\[])", "", base))
    return out[:8]


LABEL_REF_RE = re.compile(r"\((\d{1,3}[a-z]{0,2}|[IVX]{1,4}[a-z]?|H?\d?L\d{0,2}[a-z]?)\)")


def article_label_names(paragraphs: typing.Iterable[str]) -> typing.Dict[str, typing.List[str]]:
    """label -> names written right before "(label)" anywhere in the article ("... benzaldehyde (11)")."""
    out: typing.Dict[str, typing.List[str]] = collections.defaultdict(list)
    for text in paragraphs:
        for m in LABEL_REF_RE.finditer(text):
            before = text[max(0, m.start() - 220):m.start()].rstrip()
            words = before.split(" ")
            for k in (1, 2, 3):
                if len(words) < k:
                    break
                cand = " ".join(words[-k:]).strip(" ,;:")
                cand = re.sub(r"^.*?\b(?:of|gave|give|afforded|yielded|obtained|compound|furnished|with|and|to|from|as)\s+", "", cand)
                if len(cand) >= 6 and re.search(r"[a-z]{3}", cand) and re.search(r"[\d\-\(\[]|yl|ine|one|ol|ate|ide|ene|ane", cand):
                    if cand not in out[m.group(1)] and len(out[m.group(1)]) < 6:
                        out[m.group(1)].append(cand)
    return out


def _article_text(fragment: str) -> typing.Tuple[str, typing.List[typing.Tuple[int, str]]]:
    """Plain text of an XML fragment and its bold spans as (position in the plain text, text)."""
    text = re.sub(r"<sub>\s*([^<]*?)\s*</sub>", r"\1", fragment)
    text = re.sub(r"<sup>\s*([^<]*?)\s*</sup>", r"\1", text)
    text = re.sub(r"<(?:bold|b)>", "\x01", text)
    text = re.sub(r"</(?:bold|b)>", "\x02", text)
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()
    plain, bold, start = [], [], None
    for ch in text:
        if ch == "\x01":
            start = len(plain)
        elif ch == "\x02":
            if start is not None:
                bold.append((start, "".join(plain[start:])))
            start = None
        else:
            plain.append(ch)
    return "".join(plain), bold


def parse_file(path: str) -> typing.List[dict]:
    rows = []
    xml = gzip.open(path, "rt", errors="ignore").read()
    for article in re.split(r"(?=<article[ >])", xml)[1:]:
        pmcid = re.search(r'<article-id pub-id-type="pmc[^"]*">(?:PMC)?(\d+)<', article)
        doi = re.search(r'<article-id pub-id-type="doi">([^<]+)<', article)
        year = re.search(r"<pub-date[^>]*>.*?<year>(\d{4})</year>", article, re.S)
        journal = re.search(r"<journal-title>([^<]+)<", article)
        title = None
        first_row = len(rows)
        paragraphs = []
        for block in re.finditer(r"<(title|p)(?:\s[^>]*)?>(.*?)</\1>", article, re.S):
            text, bold = _article_text(block.group(2))
            paragraphs.append(text)
            if block.group(1) == "title":
                title = text
                continue
            prev = 0
            for st in find_statements(text):
                segment = text[prev : st.start]
                segment_bold = [b for pos, b in bold if prev <= pos < st.start]
                prev = st.end
                rows.append(
                    {
                        "text": "body",
                        "file": path,
                        "pmcid": pmcid.group(1) if pmcid else None,
                        "doi": doi.group(1) if doi else None,
                        "year": int(year.group(1)) if year else None,
                        "journal": journal.group(1) if journal else None,
                        **_statement_fields(st),
                        **dict(zip(("names", "labels"), name_candidates(segment, segment_bold, title))),
                        "bracket": bracket_formulas(segment, title),
                    }
                )
        _attach_label_names(rows[first_row:], paragraphs)
    return rows


def _attach_label_names(rows: typing.List[dict], paragraphs: typing.List[str]) -> None:
    if not rows:
        return
    table = article_label_names(paragraphs)
    for r in rows:
        r["label_names"] = [n for label in r["labels"] for n in table.get(label, [])][:12]


SI_HEADING_RE = re.compile(r"[A-Za-z]{3}")


def _statement_fields(st) -> dict:
    return {"formula_text": st.formula_text, "formula": st.formula, "solvates": st.solvates, "calc": st.calc, "found": st.found}


def _is_heading(text: str) -> bool:
    # Heading-like paragraph: short, with words, not a sentence (a compound name may end with its label)
    return len(text) <= 250 and bool(SI_HEADING_RE.search(text)) and (not text.endswith((".", ",", ";", ":")) or bool(re.search(r"\)\.?$", text)))


def parse_text(text: str, meta: dict) -> typing.List[dict]:
    """Plain text with paragraphs separated by blank lines (supporting information, patents).
    Statements are searched in the whole text (a statement may run over a paragraph break); the
    compound name is at the start of its paragraph or in the nearest heading-like paragraph above."""
    rows = []
    starts = [0] + [m.end() for m in re.finditer(r"\n\n", text)]
    statements = find_statements(text)
    if not statements:
        return rows
    import bisect
    for st in statements:
        k = bisect.bisect_right(starts, st.start) - 1
        segment = text[starts[k]:st.start]
        title = None
        for j in range(k - 1, max(-1, k - 40), -1):
            para = text[starts[j]:starts[j + 1]].strip()
            if find_statements(para):
                break  # the previous compound
            para = normalize_name(para) if len(para) < 300 else para
            if _is_heading(para):
                title = para
                break
        rows.append({**meta, **_statement_fields(st), **dict(zip(("names", "labels"), name_candidates(segment, [], title))), "bracket": bracket_formulas(segment, title)})
    _attach_label_names(rows, text.split("\n\n"))
    return rows


def parse_si_file(path: str) -> typing.List[dict]:
    meta_path = os.path.join(os.path.dirname(path), "meta.jsonl")
    pmcid = os.path.basename(path).split(".")[0]
    meta = _si_meta(meta_path).get(pmcid, {})
    text = gzip.open(path, "rt", errors="ignore").read()
    return parse_text(text, {"text": "si", "file": path, "pmcid": pmcid.replace("PMC", ""), "doi": meta.get("doi"), "year": meta.get("year"), "journal": meta.get("journal")})


_SI_META: typing.Dict[str, dict] = {}


def _si_meta(path: str) -> dict:
    if path not in _SI_META:
        _SI_META[path] = {}
        if os.path.exists(path):
            for line in open(path):
                a = json.loads(line)
                _SI_META[path][a["pmcid"]] = a
    return _SI_META[path]


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


PARSE_TIMEOUT = 60  # s per file: a pathological text (regex backtracking) must not stall the whole pool


def parse_any(path: str) -> typing.List[dict]:
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(PARSE_TIMEOUT)
    try:
        return parse_si_file(path) if path.endswith(".txt.gz") else parse_file(path)
    except _Timeout:
        print(f"skipped (over {PARSE_TIMEOUT} s): {path}", file=sys.stderr, flush=True)
        return []
    finally:
        signal.alarm(0)


# ----------------------------------------------------------------------------- complexes
# Longest first, ligands and metals together: Ir before I, Ni before N
_KEYS = sorted(set(LIGANDS) | METALS, key=len, reverse=True)
_ALKALI = {"Li", "Na", "K", "Cs", "Rb"}  # counter cations, listed in LIGANDS


def _number(text: str, pos: int) -> typing.Tuple[int, int]:
    match = re.match(r"\d+", text[pos:])
    return (int(match.group()), pos + match.end()) if match else (1, pos)


def _parse_sequence(
    text: str, pos: int, close: typing.Optional[str], labels: typing.Optional[dict] = None, keys: typing.Optional[list] = None
) -> typing.Tuple[typing.Optional[list], int]:
    """[(kind, value, count)] with kind 'metal', 'ligand' or 'label' (a ligand of the same article); stops at `close`."""
    labels = labels or {}
    keys = keys or _KEYS
    items = []
    while pos < len(text):
        ch = text[pos]
        if close and ch == close:
            return items, pos + 1
        if ch in "([":
            end_char = ")" if ch == "(" else "]"
            depth, j = 0, pos
            while j < len(text):
                depth += text[j] in "(["
                depth -= text[j] in ")]"
                if depth == 0:
                    break
                j += 1
            inner = text[pos + 1 : j]
            n, nxt = _number(text, j + 1)
            if inner in labels:
                items.append(("label", inner, n))
            elif inner in LIGANDS:
                items.append(("ligand", inner, n))
            else:
                sub, _ = _parse_sequence(inner, 0, None, labels, keys)
                if sub is None:
                    return None, pos
                items += [(k, v, c * n) for k, v, c in sub]
            pos = nxt
            continue
        for key in keys:
            if text.startswith(key, pos):
                n, pos = _number(text, pos + len(key))
                if key in labels:
                    items.append(("label", key, n))
                else:
                    items.append(("ligand" if key in LIGANDS and (key not in METALS or key in _ALKALI) else "metal", key, n))
                break
        else:
            return None, pos
    return items, pos


def _assemble(items: list, extra_charge: int = 0) -> typing.Optional[str]:
    metals = [(v, c) for k, v, c in items if k == "metal"]
    ligands = [(v, c) for k, v, c in items if k == "ligand"]
    if not metals:
        return None
    charge = sum(LIGANDS[v][1] * c for v, c in ligands) + extra_charge
    n_metal = sum(c for _, c in metals)
    if n_metal <= 0 or charge % n_metal:
        return None
    metal_charge = -charge // n_metal
    parts = []
    for v, c in ligands:
        parts += [LIGANDS[v][0]] * c
    sign = "" if metal_charge == 0 else ("+" if metal_charge > 0 else "-") + (str(abs(metal_charge)) if abs(metal_charge) > 1 else "")
    for v, c in metals:
        parts += [f"[{v}{sign}]"] * c
    return ".".join(parts)


def complex_formula_to_smiles(text: str, labels: typing.Optional[dict] = None) -> typing.List[str]:
    """SMILES options for [Cu(phen)2Cl]ClO4; `labels` maps ligand labels of the article (L1, HL) to
    [(SMILES, charge)] options (neutral and deprotonated)."""
    text = re.sub(r"η\d-|κ\d?-?", "", text.replace("{", "(").replace("}", ")"))
    text = re.split(r"[·•∙⋅]", text)[0].strip()
    labels = labels or {}
    keys = sorted(set(_KEYS) | set(labels), key=len, reverse=True) if labels else _KEYS
    items, pos = _parse_sequence(text, 0, None, labels, keys)
    if items is None or pos < len(text):
        return []
    used = sorted({v for k, v, _ in items if k == "label"})
    options = []
    for choice in itertools.product(*[labels[u] for u in used]):
        chosen = dict(zip(used, choice))
        base = [(k, v, c) for k, v, c in items if k != "label"]
        extra = [chosen[v] for k, v, c in items if k == "label" for _ in range(c)]
        core = _assemble(base, sum(q for _, q in extra))
        if core:
            options.append(".".join([core] + [s for s, _ in extra]))
        if len(options) >= 30:
            break
    return options


def deprotonate(smiles: str, k: int) -> typing.Optional[str]:
    """Remove k acidic hydrogens (O-H first, then S-H, then N-H)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.RWMol(mol)
    for _ in range(k):
        for symbol in ("O", "S", "N"):
            atom = next((a for a in mol.GetAtoms() if a.GetSymbol() == symbol and a.GetTotalNumHs() > 0 and a.GetFormalCharge() == 0), None)
            if atom is not None:
                atom.SetNumExplicitHs(atom.GetTotalNumHs() - 1)
                atom.SetNoImplicit(True)
                atom.SetFormalCharge(-1)
                break
        else:
            return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol)


def article_ligands(resolved: typing.Dict[str, str]) -> typing.Dict[str, typing.List[typing.Tuple[str, int]]]:
    """Label -> SMILES options: as is, deprotonated once or twice; HnL also gives L deprotonated n times."""
    out: typing.Dict[str, typing.List[typing.Tuple[str, int]]] = {}
    for label, smiles in resolved.items():
        variants = [(smiles, 0)] + [(d, -k) for k in (1, 2) if (d := deprotonate(smiles, k))]
        out.setdefault(label, []).extend(variants)
        match = re.fullmatch(r"H(\d?)(L.*)", label)
        if match:
            k = int(match.group(1) or 1)
            d = deprotonate(smiles, k)
            if d:
                out.setdefault(match.group(2), []).append((d, -k))
    return out


def complex_name_to_smiles(name: str, organic: typing.Callable[[str], typing.Optional[str]]) -> typing.List[str]:
    """Additive names: dichlorobis(triphenylphosphine)palladium(II) [hexafluorophosphate].
    Names rarely give the number of counterions, so SMILES with 1-3 of them are returned."""
    text = name.lower().replace("′", "'").replace("″", "''")
    text = re.sub(r"\((?:[ivx]+|0|[+-]?\d)\)", "", text)
    counter = None
    for word, key in COUNTER_ANION_NAMES.items():
        if text.endswith(" " + word) or text.endswith(" di" + word) or text.endswith(" bis(" + word + ")"):
            counter = key
            text = text[: text.rfind(" ")].strip()
            break
    metal = None
    for word, symbol in sorted(METAL_NAMES.items(), key=lambda t: -len(t[0])):
        if text.endswith(word):
            metal, text = symbol, text[: -len(word)]
            break
    if metal is None:
        return []
    text = text.strip(" -")
    items = [("metal", metal, 1)]
    smiles_extra = []
    pos = 0
    names = sorted(LIGAND_NAMES, key=len, reverse=True)
    while pos < len(text):
        if text[pos] in " -":
            pos += 1
            continue
        count = 1
        for mult in sorted(MULTIPLIERS, key=len, reverse=True):
            if text.startswith(mult, pos) and not any(text.startswith(n, pos) for n in names if n.startswith(mult)):
                count, pos = MULTIPLIERS[mult], pos + len(mult)
                break
        if pos < len(text) and text[pos] in "([":
            depth, j = 0, pos
            while j < len(text):
                depth += text[j] in "(["
                depth -= text[j] in ")]"
                if depth == 0:
                    break
                j += 1
            inner = text[pos + 1 : j].strip()
            pos = j + 1
            if inner in LIGAND_NAMES:
                items.append(("ligand", LIGAND_NAMES[inner], count))
                continue
            smiles = organic(inner)
            if smiles is None or Chem.MolFromSmiles(smiles) is None:
                return []
            smiles_extra += [smiles] * count
            continue
        for n in names:
            if text.startswith(n, pos):
                items.append(("ligand", LIGAND_NAMES[n], count))
                pos += len(n)
                break
        else:
            return []
    extra_charge = sum(Chem.GetFormalCharge(Chem.MolFromSmiles(s)) for s in smiles_extra)
    options = []
    for n in (1, 2, 3) if counter else (0,):
        core = _assemble(items + ([("ligand", counter, n)] if counter else []), extra_charge)
        if core:
            options.append(".".join([core] + smiles_extra))
    return options


# ----------------------------------------------------------------------------- name resolvers
OPSIN_PROCESSES = 8


def run_opsin(names: typing.Iterable[str], jar: str) -> typing.Dict[str, str]:
    names = sorted({n for n in names if "\n" not in n})
    java = os.path.join(sys.prefix, "bin", "java")
    java = java if os.path.exists(java) else "java"

    def one(chunk):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("\n".join(chunk) + "\n")
            path = f.name
        out = subprocess.run([java, "-Xmx2g", "-jar", jar, "-osmi", path], capture_output=True, text=True).stdout.split("\n")
        os.unlink(path)
        return {n: s.strip() for n, s in zip(chunk, out) if s.strip()}

    # Several JVMs in parallel: OPSIN itself is single-threaded
    import concurrent.futures
    size = max(1, -(-len(names) // OPSIN_PROCESSES))
    chunks = [names[i:i + size] for i in range(0, len(names), size)]
    result = {}
    with concurrent.futures.ThreadPoolExecutor(OPSIN_PROCESSES) as pool:
        for part in pool.map(one, chunks):
            result.update(part)
    return result


def run_pubchem(names: typing.Iterable[str], cache_path: str, limit: int) -> typing.Dict[str, str]:
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    todo = [n for n in names if n not in cache][:limit]
    for i, name in enumerate(tqdm(todo, desc="PubChem")):
        url = PUBCHEM.format(urllib.parse.quote(name, safe=""))
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "blind-ea/0.1"}), timeout=30) as r:
                props = json.loads(r.read())["PropertyTable"]["Properties"][0]
                cache[name] = props.get("IsomericSMILES") or props.get("SMILES")
        except Exception:
            cache[name] = None
        time.sleep(0.22)  # PubChem allows 5 requests per second
        if i % 200 == 0:
            json.dump(cache, open(cache_path, "w"))
    json.dump(cache, open(cache_path, "w"))
    return {n: s for n, s in cache.items() if s}


# ----------------------------------------------------------------------------- matching
def _same(a: typing.Mapping[str, float], b: typing.Mapping[str, float]) -> bool:
    keys = {k for k, v in a.items() if v} | {k for k, v in b.items() if v}
    return all(abs(a.get(k, 0) - b.get(k, 0)) < 1e-6 for k in keys)


def formula_targets(row: dict) -> typing.List[typing.Dict[str, float]]:
    """Formula with solvates, and without them (the SMILES is usually of the compound alone)."""
    total = row["formula"]
    targets = [total]
    if row["solvates"]:
        main = collections.Counter(total)
        for formula, multiplier in row["solvates"]:
            for e, n in formula_counts_plain(formula).items():
                main[e] -= multiplier * n
        targets.append(dict(main))
    return targets


def canonical(smiles: str) -> typing.Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def hill(counts: typing.Mapping[str, float]) -> str:
    def part(e):
        n = counts[e]
        n = int(n) if float(n).is_integer() else round(n, 2)
        return e + ("" if n == 1 else str(n))
    keys = [e for e, n in counts.items() if n]
    if "C" in keys:
        order = ["C"] + (["H"] if "H" in keys else []) + sorted(e for e in keys if e not in ("C", "H"))
    else:
        order = sorted(keys)
    return "".join(part(e) for e in order)


def llm_key(row: dict) -> str:
    # One analysis: its article, candidate names and formula (the formula last, for modalities/ea/llm_smiles.py check)
    names = json.dumps(row["names"] + row["bracket"] or row["labels"], ensure_ascii=False)
    digest = hashlib.md5(f"{row['pmcid']}|{names}".encode()).hexdigest()[:16]
    return f"{digest}\t{hill(formula_targets(row)[-1])}"


# Salt formers and solvents that a name and a formula often disagree on ("X hydrochloride" with the
# formula of the base, or the formula of the hydrochloride with the name of the base)
SALT_FRAGMENTS = {"Cl": "HCl", "Br": "HBr", "O": "H2O", "OC(=O)C(F)(F)F": "C2HF3O2", "CS(=O)(=O)O": "CH4O3S", "I": "HI"}


def adjust_salt(smiles: str, counts: dict, targets: list) -> typing.Optional[str]:
    """SMILES with salt/solvent fragments removed or added so that its composition is a target."""
    fragments = smiles.split(".")
    if len(fragments) > 1:
        for keep in itertools.product([True, False], repeat=min(len(fragments), 4)):
            kept = [f for f, k in zip(fragments, keep) if k] + fragments[4:]
            if kept and len(kept) < len(fragments):
                c = formula_counts(".".join(kept))
                if c and any(_same(c, t) for t in targets):
                    return canonical(".".join(kept))
    for t in targets:
        diff = {e: t.get(e, 0) - counts.get(e, 0) for e in set(t) | set(counts)}
        diff = {e: d for e, d in diff.items() if abs(d) > 1e-6}
        if not diff or any(d < 0 for d in diff.values()):
            continue
        for frag, formula in SALT_FRAGMENTS.items():
            unit = formula_counts_plain(formula)
            k = diff.get(next(iter(unit)), 0) / unit[next(iter(unit))]
            for kk in (k,):
                if kk in (1, 2, 3) and all(abs(diff.get(e, 0) - kk * unit.get(e, 0)) < 1e-6 for e in set(diff) | set(unit)):
                    return canonical(smiles + ("." + frag) * int(kk))
    return None


def resolve(
    row: dict, opsin: dict, pubchem: dict, labels: typing.Optional[dict] = None, llm: typing.Optional[dict] = None
) -> typing.Tuple[typing.Optional[str], typing.Optional[str], typing.Optional[str]]:
    targets = formula_targets(row)

    def check(smiles):
        if not smiles:
            return None
        smiles = canonical(smiles)
        counts = formula_counts(smiles) if smiles else None
        if not counts:
            return None
        if any(_same(counts, t) for t in targets):
            return smiles
        return adjust_salt(smiles, counts, targets)

    for formula in row["bracket"]:
        for option in complex_formula_to_smiles(formula, labels):
            s = check(option)
            if s:
                return s, "ligand_formula" if not labels else "ligand_formula_with_article_ligands", formula
    for name in row["names"]:
        for variant in name_variants(name):
            s = check(opsin.get(variant))
            if s:
                return s, "opsin", variant
        for option in complex_name_to_smiles(name, lambda n: opsin.get(n)):
            s = check(option)
            if s:
                return s, "ligand_name", name
    for name in row.get("label_names", ()):
        for variant in name_variants(name):
            s = check(opsin.get(variant))
            if s:
                return s, "opsin_label", variant
    for name in row["names"]:
        s = check(pubchem.get(name))
        if s:
            return s, "pubchem", name
    for option in llm.get(llm_key(row), ()) if llm else ():
        s = check(option)
        if s:
            return s, "llm", None
    return None, None, None


GENERIC_NAME_RE = re.compile(r"^(?:compounds?|complex(?:es)?|ligands?|products?|experimental|general|synthesis|under|yield|the|a|an|to|in)\b", re.I)


def translatable(name: str) -> bool:
    # Names worth asking an LLM: some chemistry in them, not only a compound label
    return len(name) >= 4 and len(re.findall(r"[A-Za-z]", name)) >= 3 and not GENERIC_NAME_RE.match(name) and not re.fullmatch(LABEL, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--input-dirs", nargs="+", default=["data/ea"])
    parser.add_argument("--opsin-jar", default="data/ea/tools/opsin.jar")
    parser.add_argument("--output-dir", default="data/ea/dataset")
    parser.add_argument("--pubchem-limit", type=int, default=0, help="Max PubChem lookups (0: no PubChem).")
    parser.add_argument("--num-workers", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    paths = sorted(p for d in args.input_dirs for p in glob.glob(os.path.join(d, "raw", "*.xml.gz")))
    paths += sorted(p for d in args.input_dirs for p in glob.glob(os.path.join(d, "si", "PMC*.txt.gz")))
    with multiprocessing.Pool(args.num_workers) as pool:
        rows = [r for part in tqdm(pool.imap_unordered(parse_any, paths, chunksize=4), total=len(paths), desc="Parsing") for r in part]
    seen, unique = set(), []
    for r in rows:
        key = (r["pmcid"], json.dumps([r["formula"], r["calc"], r["found"]], sort_keys=True))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    # find_statements keeps only statements whose calculated values match the formula
    rows = [r for r in unique if r["formula"] and matches_formula(r["calc"], r["formula"])]
    print(f"statements {len(unique)} (unique), with Calcd matching the formula {len(rows)}")

    # OPSIN for names and for the organic ligands of additive names
    names = {v for r in rows for n in r["names"] + r.get("label_names", []) for v in name_variants(n)}
    ligand_names = set()
    for n in names:
        complex_name_to_smiles(n, lambda inner: ligand_names.add(inner) or "C")
    opsin_cache = os.path.join(args.output_dir, "opsin_cache.json")
    opsin = json.load(open(opsin_cache)) if os.path.exists(opsin_cache) else {}
    tried_path = os.path.join(args.output_dir, "opsin_tried.json")
    tried = set(json.load(open(tried_path))) if os.path.exists(tried_path) else set()
    todo = (names | ligand_names) - tried
    if todo and os.path.exists(args.opsin_jar):
        opsin.update(run_opsin(todo, args.opsin_jar))
        json.dump(opsin, open(opsin_cache, "w"))
        json.dump(sorted(tried | todo), open(tried_path, "w"))
    print(f"OPSIN parsed {len(opsin)} of {len(names | ligand_names)} names")

    pubchem = {}
    if args.pubchem_limit:
        unresolved = [r for r in rows if resolve(r, opsin, {})[0] is None]
        short = [n for r in unresolved for n in r["names"][:2] if len(n) <= 60 and not re.search(r"\d[a-z]?$", n)]
        pubchem = run_pubchem(list(dict.fromkeys(short)), os.path.join(args.output_dir, "pubchem_cache.json"), args.pubchem_limit)

    llm_path = os.path.join(args.output_dir, "llm_smiles.json")
    llm = json.load(open(llm_path)) if os.path.exists(llm_path) else {}
    # A key shared by several analyses (isomers of one article with the same scraped names) cannot
    # tell them apart: its LLM answer is used for none of them
    key_counts = collections.Counter(llm_key(r) for r in rows)
    llm = {k: v for k, v in llm.items() if key_counts.get(k, 0) <= 1}
    results = [resolve(r, opsin, pubchem, llm=llm) for r in tqdm(rows, desc="Matching")]
    # Second pass for complexes: ligands labelled in the same article (L1, HL, 2a) whose structure was found
    by_article: typing.Dict[str, typing.Dict[str, str]] = collections.defaultdict(dict)
    for r, (smiles, _, _) in zip(rows, results):
        if smiles and not set(r["formula"]) & METALS:
            for label in r["labels"]:
                by_article[r["pmcid"]].setdefault(label, smiles)
    for i, r in enumerate(tqdm(rows, desc="Complexes with article ligands")):
        if results[i][0] is None and set(r["formula"]) & METALS and by_article.get(r["pmcid"]):
            results[i] = resolve(r, {}, {}, article_ligands(by_article[r["pmcid"]]))

    # Names still unresolved, for modalities/ea/llm_smiles.py
    todo = {}
    for r, (smiles, _, _) in zip(rows, results):
        # Labels (3a, L1, HL) count too: the LLM looks them up in the article text (modalities/ea/llm_smiles.py context)
        names = [n for n in r["names"] + r["bracket"] if translatable(n)]
        key = llm_key(r)
        if smiles is None and (names or r["labels"]) and key not in llm and key not in todo:
            todo[key] = {
                "key": key, "names": names, "labels": r["labels"], "formula": hill(formula_targets(r)[-1]),
                "formula_text": r["formula_text"], "metal": bool(set(r["formula"]) & METALS),
                "pmcid": "PMC" + str(r["pmcid"]), "file": r["file"],
            }
    with open(os.path.join(args.output_dir, "llm_todo.jsonl"), "w") as f:
        for item in todo.values():
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"analyses left for the LLM: {len(todo)} ({sum(t['metal'] for t in todo.values())} with metals)")
    out_rows = []
    for r, (smiles, source, name) in zip(rows, results):
        out_rows.append(
            {
                "smiles": smiles,
                "formula": hill(r["formula"]),
                "formula_text": r["formula_text"],
                "solvates": r["solvates"],
                "calc": r["calc"],
                "found": r["found"],
                "name": name or (r["names"][0] if r["names"] else None),
                "smiles_source": source,
                "text": r["text"],
                "pmcid": r["pmcid"],
                "doi": r["doi"],
                "year": r["year"],
                "journal": r["journal"],
            }
        )
    with gzip.open(os.path.join(args.output_dir, "ea_dataset.jsonl.gz"), "wt") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    elements = sorted({e for r in out_rows for e in r["found"]}, key=lambda e: ("CHNOS".find(e) if e in "CHNOS" else 9, e))
    with open(os.path.join(args.output_dir, "ea_dataset_with_smiles.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "formula", "solvates"] + [f"found_{e}" for e in elements] + [f"calc_{e}" for e in elements] + ["smiles_source", "name", "text", "pmcid", "doi", "year"])
        for r in out_rows:
            if r["smiles"]:
                writer.writerow(
                    [r["smiles"], r["formula"], ";".join(f"{m}{s}" for s, m in r["solvates"])]
                    + [r["found"].get(e, "") for e in elements] + [r["calc"].get(e, "") for e in elements]
                    + [r["smiles_source"], r["name"], r["text"], r["pmcid"], r["doi"], r["year"]]
                )
    stats = collections.Counter(r["smiles_source"] for r in out_rows)
    summary = {
        "articles": len({r["pmcid"] for r in out_rows}),
        "analyses": len(out_rows),
        "with_smiles": sum(bool(r["smiles"]) for r in out_rows),
        "smiles_source": dict(stats),
        "unique_smiles": len({r["smiles"] for r in out_rows if r["smiles"]}),
    }
    json.dump(summary, open(os.path.join(args.output_dir, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
