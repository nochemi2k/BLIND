"""
Elemental analysis statements in free text: calculated and found values and the molecular formula.

Papers write them in many ways:
    Anal. Calcd for C15H17NO3: C, 69.48; H, 6.61; N, 5.40. Found: C, 69.23; H, 6.62; N, 5.51.
    (Found: C, 69.2; H, 6.6. C15H17NO3 requires C, 69.5; H, 6.6%)
    Anal. Found: C, 74.30; H, 5.65. Calcd. for C15H14O3: C, 74.36; H, 5.82%.
    C26H21NO4 (411.46): calcd. C 75.90, H 5.14, N 3.40; found C 75.91, H 5.12, N 3.41.
    Elemental analysis: calculated for C17H12FNO4 (313.28) %C 65.18, %H 3.86; found %C 65.18, %H 3.85
    Anal. found (calcd) for C20H18O2: C, 82.65 (82.73); H, 6.31 (6.25).
    Calculated for C27H25N5O10, %: C, 55.96; ... Found, %: C, 56.02; ...
    Anal. Calcd: C 72.47%, H 6.08%. Found: ...    (formula only in the HRMS line before)
Instead of one pattern per style: blocks of element-value pairs are found, two adjacent blocks
(or one block of "value (value)" pairs) near the words calcd/found/requires form a statement,
and the formula is any candidate formula in the text around them whose composition gives the
calculated values (absolute %, within CALC_TOLERANCE). That check also tells which block is the
calculated one, so a wrong formula, wrong order or a stray number block cannot pass.
"""
import collections
import re
import typing

from .elemental_analysis import ATOMIC_MASS, mass_percent

CALC_TOLERANCE = 0.06
NUM = r"\d{1,2}[.,]\d{1,3}"
ELEMENT = r"[A-Z][a-z]?"
PAIR = rf"%?\s*({ELEMENT})\s*%?\s*[,:=]?\s*({NUM})\s*%?"
SEP = r"\s*[;,]?\s*(?:and\s+)?"
BLOCK_RE = re.compile(rf"(?:{PAIR}{SEP}){{2,12}}")
PAIR_RE = re.compile(PAIR)
PAREN_PAIR = rf"%?\s*({ELEMENT})\s*%?\s*[,:=]?\s*({NUM})\s*%?\s*[\(\[]\s*({NUM})\s*%?\s*[\)\]]"
PAREN_BLOCK_RE = re.compile(rf"(?:{PAREN_PAIR}{SEP}){{2,12}}")
PAREN_PAIR_RE = re.compile(PAREN_PAIR)
CALC_WORD_RE = re.compile(r"(?i)\b(?:calc(?:ulated|d)?|theor(?:etical|y)?|requires?|required|expected)\b")
FOUND_WORD_RE = re.compile(r"(?i)\b(?:found|obsd|observed|measured|exp(?:erimental|tl)?|anal(?:ysis|ytical)?|elemental|microanaly\w*)\b")
KEYWORD_RE = re.compile(r"(?i)\b(?:anal(?:ysis|ytical|yses)?|elemental|microanaly\w*|calc(?:ulated|d)?|theor\w*|found|requires?)\b")
MAX_GAP = 160
CONTEXT = 450

# Solvent and acid names in formulas ("C16H16BrN3O·0.5EtOAc")
ALIASES = {
    "EtOAc": "C4H8O2", "AcOEt": "C4H8O2", "EtOH": "C2H6O", "MeOH": "CH4O", "Et2O": "C4H10O", "THF": "C4H8O",
    "DMF": "C3H7NO", "DMSO": "C2H6OS", "MeCN": "C2H3N", "CH3CN": "C2H3N", "acetone": "C3H6O", "hexane": "C6H14",
    "hexanes": "C6H14", "pentane": "C5H12", "toluene": "C7H8", "benzene": "C6H6", "dioxane": "C4H8O2", "DCM": "CH2Cl2",
    "TFA": "C2HF3O2", "iPrOH": "C3H8O", "i-PrOH": "C3H8O", "AcOH": "C2H4O2", "py": "C5H5N", "MeNO2": "CH3NO2",
    "CH3OH": "CH4O", "C2H5OH": "C2H6O", "H2O": "H2O", "HCl": "HCl", "HBr": "HBr", "CH2Cl2": "CH2Cl2", "CHCl3": "CHCl3",
}
# A fractional count needs digits before the point: "S.0.65CH2Cl2" is S + 0.65 CH2Cl2
COUNT = r"(?:\d+(?:\.\d+)?)?"
TOKEN = rf"(?:{ELEMENT}{COUNT}|[\(\[](?:{ELEMENT}{COUNT})+[\)\]]{COUNT})"
MAIN_RE = re.compile(rf"(?<![A-Za-z0-9])\(?(?:{TOKEN}){{2,}}")
_ALIAS_ALT = "|".join(sorted(map(re.escape, ALIASES), key=len, reverse=True))
TAIL_RE = re.compile(
    rf"\)?\s*(?:[·•∙⋅×*+]|\.(?=\s*\d*\.?\d*\s*[A-Z(])|[–-](?=\s*\d)|(?=\s+\d*\.?\d+\s*(?:{_ALIAS_ALT}|[A-Z])))\s*"
    rf"(\d*\.?\d+)?\s*({_ALIAS_ALT}|(?:{TOKEN})+)(?![a-z])"
)


class Statement(typing.NamedTuple):
    start: int
    end: int
    formula_text: str
    formula: typing.Dict[str, float]
    solvates: typing.List[typing.Tuple[str, float]]
    calc: typing.Dict[str, float]
    found: typing.Dict[str, float]


def _expand(body: str) -> typing.Optional[typing.Counter]:
    body = body.replace("[", "(").replace("]", ")")
    while "(" in body:  # (CH3)2 -> CH3CH3
        match = re.search(r"\(([^()]*)\)(\d*\.?\d*)", body)
        if match is None:
            return None
        inner = _expand(match.group(1))
        if inner is None:
            return None
        k = float(match.group(2)) if match.group(2) else 1.0
        body = body[: match.start()] + "".join(f"{e}{n * k:g}" for e, n in inner.items()) + body[match.end():]
    if not re.fullmatch(r"(?:[A-Z][a-z]?\d*\.?\d*)+", body):
        return None
    counts: typing.Counter = collections.Counter()
    for element, number in re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", body):
        if element not in ATOMIC_MASS:
            return None
        counts[element] += float(number) if number else 1.0
    return counts


def formula_candidates(text: str) -> typing.List[typing.Tuple[int, str, typing.Dict[str, float], list]]:
    """(position, text, {element: count} incl. solvates, [(solvate formula, multiplier)]) of every
    formula-like token, with and without each solvate part that follows it."""
    out = []
    for match in MAIN_RE.finditer(text):
        main = _expand(match.group(0).lstrip("("))
        if not main or len(main) < 2 or sum(main.values()) < 3:
            continue
        options = [(match.group(0).lstrip("("), dict(main), [])]
        for variant in (match.group(0).lstrip("(").replace("CI", "Cl"),) if "CI" in match.group(0) else ():
            counts = _expand(variant)  # "CIF" for ClF: an I read for an l
            if counts:
                options.append((variant, dict(counts), []))
        total, solvates, pos, shown = collections.Counter(main), [], match.end(), match.group(0).lstrip("(")
        for _ in range(3):
            tail = TAIL_RE.match(text, pos)
            if not tail:
                break
            multiplier = float(tail.group(1)) if tail.group(1) else 1.0
            name = tail.group(2)
            counts = _expand(ALIASES.get(name, name))
            if not counts or multiplier <= 0 or multiplier > 20:
                break
            for e, n in counts.items():
                total[e] += multiplier * n
            solvates = solvates + [(ALIASES.get(name, name), multiplier)]
            shown = shown + f"·{tail.group(1) or ''}{name}"
            options.append((shown, dict(total), list(solvates)))
            pos = tail.end()
        out += [(match.start(), s, f, sv) for s, f, sv in options]
    return out


def matches_formula(calc: typing.Mapping[str, float], formula: typing.Mapping[str, float]) -> bool:
    expected = mass_percent(formula)
    return all(abs(expected.get(e, 0.0) - v) < CALC_TOLERANCE for e, v in calc.items()) and all(e in formula or v == 0 for e, v in calc.items())


def _values(block: str) -> typing.Optional[typing.Dict[str, float]]:
    values = {}
    for element, number in PAIR_RE.findall(block):
        if element not in ATOMIC_MASS or element in values:
            return None
        values[element] = float(number.replace(",", "."))
    return values if len(values) >= 2 else None


def _choose(text: str, lo: int, hi: int, calc_pos: int, calc: dict) -> typing.Optional[tuple]:
    """Formula candidate in text[lo:hi] that gives the calculated values, nearest to calc_pos."""
    best = None
    for pos, shown, formula, solvates in formula_candidates(text[lo:hi]):
        if matches_formula(calc, formula):
            distance = abs(lo + pos - calc_pos)
            if best is None or distance < best[0]:
                best = (distance, lo + pos, shown, formula, solvates)
    return best


def find_statements(text: str) -> typing.List[Statement]:
    out, used = [], []
    blocks = [(m.start(), m.end(), _values(m.group(0))) for m in BLOCK_RE.finditer(text)]
    blocks = [b for b in blocks if b[2]]
    i = 0
    while i + 1 < len(blocks):
        (s1, e1, v1), (s2, e2, v2) = blocks[i], blocks[i + 1]
        gap = text[e1:s2]
        before = text[max(0, s1 - 200):s1]
        if len(gap) > MAX_GAP or not (FOUND_WORD_RE.search(gap) or CALC_WORD_RE.search(gap)) or not KEYWORD_RE.search(before + gap):
            i += 1
            continue
        # Calculated first unless the gap names the calculated values ("Found: ... Calcd for X: ...")
        orders = [(v1, v2, s1), (v2, v1, s2)]
        if CALC_WORD_RE.search(gap) and not re.search(r"(?i)\bfound\b", gap):
            orders.reverse()
        for calc, found, calc_pos in orders:
            if not set(found) <= set(calc) and not set(calc) <= set(found):
                continue
            pick = _choose(text, max(0, s1 - CONTEXT), s2, calc_pos, calc)
            if pick:
                start = min(pick[1], s1)
                keyword = list(KEYWORD_RE.finditer(text, max(0, s1 - 200), s1))
                if keyword:
                    start = min(start, keyword[0].start())
                out.append(Statement(start, e2, pick[2], pick[3], pick[4], calc, found))
                used.append((s1, e2))
                break
        i += 2 if out and out[-1].end == e2 else 1
    for m in PAREN_BLOCK_RE.finditer(text):
        if any(a <= m.start() < b for a, b in used):
            continue
        pairs = PAREN_PAIR_RE.findall(m.group(0))
        first = {e: float(x.replace(",", ".")) for e, x, _ in pairs}
        second = {e: float(y.replace(",", ".")) for e, _, y in pairs}
        if len(first) < 2 or len(first) != len(pairs) or any(e not in ATOMIC_MASS for e in first):
            continue
        before = text[max(0, m.start() - 200):m.start()]
        if not KEYWORD_RE.search(before):
            continue
        # "found (calcd)" is the usual order; the formula check decides
        for calc, found in ((second, first), (first, second)):
            pick = _choose(text, max(0, m.start() - CONTEXT), m.end(), m.start(), calc)
            if pick:
                keyword = list(KEYWORD_RE.finditer(text, max(0, m.start() - 200), m.start()))
                start = min([pick[1], m.start()] + ([keyword[0].start()] if keyword else []))
                out.append(Statement(start, m.end(), pick[2], pick[3], pick[4], calc, found))
                break
    return sorted(out)
