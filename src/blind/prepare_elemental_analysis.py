"""
Build an empirical noise model for elemental analysis (CHN combustion) from published data.

Open-access articles in PubMed Central that report "Anal. Calcd for <formula>: C, x; H, y.
Found: C, x; H, y" are downloaded (PMCIDs from the Europe PMC search, full-text XML from NCBI
E-utilities), the statements are parsed, and only those whose calculated values match the
formula are kept. The output noise model (ea_noise.json) holds, per element, how often it is
reported when present, the solvates added to formulas, and the reported Found - Calcd vectors,
which src/blind/elemental_analysis.py resamples to imitate real data.
"""
import argparse
import collections
import glob
import gzip
import json
import os
import re
import time
import typing
import urllib.parse
import urllib.request

from tqdm.auto import tqdm

from .elemental_analysis import ATOMIC_MASS, mass_percent

SEARCH_QUERY = (
    '("Anal. Calcd" OR "Anal. Calc" OR "Elemental analysis calcd" OR "Anal. calcd") '
    "AND OPEN_ACCESS:y AND IN_PMC:y"
)
EUROPE_PMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id="
EFETCH_BATCH = 20
USER_AGENT = {"User-Agent": "blind-elemental-analysis/0.1"}

# Acids of salts: in the NMR data the salt is part of the SMILES, so only neutral solvates are modelled
SALT_FORMERS = {"HCl", "HBr", "HI", "HF", "C2H2O4", "C2HF3O2", "H2SO4", "H3PO4", "CH4O3S", "HNO3", "HClO4", "C4H4O4"}
# Calculated values must match the formula this closely (abs. %), otherwise the statement
# has a typo in the formula or in the values and its Found - Calcd is meaningless.
CALC_TOLERANCE = 0.06

NUM = r"\d{1,2}[.,]\d{1,3}"
PAIR = rf"(?:[A-Z][a-z]?\s*[,:=]?\s*{NUM}\s*%?\s*[;,]?\s*(?:and\s*)?)"
STATEMENT_RE = re.compile(
    rf"(?i:anal(?:ysis|ytical)?\.?|elemental\s+analysis|microanalysis)\s*(?:\((?:%|[^)]{{0,40}})\))?\s*[:.]?\s*"
    rf"(?i:calc(?:ulated|d)?\.?|theor(?:etical|\.)?)\s*(?:\(%\))?\s*(?i:for)?\s*(?P<formula>[^:;]{{2,160}}?)\s*[:;]?\s*(?:\(%\))?\s*"
    rf"(?P<calc>{PAIR}{{2,8}})[^A-Za-z]{{0,12}}(?i:found|obsd\.?|observed|measured|exp(?:erimental|\.)?)\s*(?:\(%\))?\s*[:.]?\s*"
    rf"(?P<found>{PAIR}{{2,8}})"
)
VALUE_RE = re.compile(rf"([A-Z][a-z]?)\s*[,:=]?\s*({NUM})")
SOLVENT_NAMES = {
    "H2O": "H2O", "CH2Cl2": "CH2Cl2", "CHCl3": "CHCl3", "CH3OH": "CH4O", "CH4O": "CH4O", "C2H5OH": "C2H6O",
    "C2H6O": "C2H6O", "C4H8O2": "C4H8O2", "C4H8O": "C4H8O", "C6H14": "C6H14", "CH3CN": "C2H3N", "C2H3N": "C2H3N",
    "C7H8": "C7H8", "C6H6": "C6H6", "C3H6O": "C3H6O", "C4H10O": "C4H10O", "HCl": "HCl", "HBr": "HBr",
    "H2C2O4": "C2H2O4", "C2H2O4": "C2H2O4", "CF3COOH": "C2HF3O2", "C2HF3O2": "C2HF3O2", "C3H7NO": "C3H7NO", "C2H6OS": "C2H6OS",
}


def fetch(url: str, tries: int = 5) -> bytes:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=USER_AGENT), timeout=120) as response:
                return response.read()
        except OSError as error:
            print(f"retry {attempt}: {error}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}")


def download(output_dir: str, num_articles: int) -> None:
    """Full-text XML of up to num_articles articles into <output_dir>/raw; finished batches are skipped."""
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    ids_path = os.path.join(output_dir, "pmcids.json")
    if os.path.exists(ids_path):
        ids = json.load(open(ids_path))
    else:
        ids, cursor = [], "*"
        while len(ids) < num_articles:
            params = {"query": SEARCH_QUERY, "format": "json", "pageSize": 1000, "resultType": "lite", "cursorMark": cursor}
            page = json.loads(fetch(EUROPE_PMC_SEARCH + urllib.parse.urlencode(params)))
            ids += [r["pmcid"] for r in page["resultList"]["result"] if r.get("pmcid")]
            if page.get("nextCursorMark") in (None, cursor):
                break
            cursor = page["nextCursorMark"]
        ids = list(dict.fromkeys(ids))[:num_articles]
        json.dump(ids, open(ids_path, "w"))
    for start in tqdm(range(0, len(ids), EFETCH_BATCH), desc="Downloading articles"):
        path = os.path.join(raw_dir, f"batch_{start // EFETCH_BATCH:04d}.xml.gz")
        if os.path.exists(path):
            continue
        batch = ",".join(i.replace("PMC", "") for i in ids[start : start + EFETCH_BATCH])
        data = fetch(NCBI_EFETCH + batch)
        with gzip.open(path, "wb") as f:
            f.write(data)
        time.sleep(0.4)  # NCBI allows 3 requests per second without an API key


def _expand(body: str) -> typing.Optional[typing.Counter]:
    while "(" in body:  # (CH3)2 -> CH3CH3
        match = re.search(r"\(([^()]*)\)(\d*)", body)
        if match is None:
            return None
        body = body[: match.start()] + match.group(1) * (int(match.group(2)) if match.group(2) else 1) + body[match.end() :]
    if not re.fullmatch(r"(?:[A-Z][a-z]?\d*\.?\d*)+", body):
        return None
    counts: typing.Counter = collections.Counter()
    for element, number in re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", body):
        if element not in ATOMIC_MASS:
            return None
        counts[element] += float(number) if number else 1.0
    return counts


def parse_formula(text: str) -> typing.Tuple[typing.Optional[typing.Dict[str, float]], typing.List[typing.Tuple[str, float]]]:
    """'C10H12N2O·0.5H2O (MW 186.2)' -> ({element: count} incl. solvate, [(solvate formula, multiplier)])."""
    text = re.sub(r"\([^)]*\d{2,}\.\d+[^)]*\)", "", text)  # (MW 369.21)
    text = re.sub(r"(?i)\b(mw|m\.w\.|fw)\b.*$", "", text)
    for dot in ("∙", "•", "⋅", "×", "*"):
        text = text.replace(dot, "·")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"(?<=[A-Za-z)])\.(?=\d*\.?\d*[A-Z(])", "·", text)  # C10H12N2O.0.5H2O, not the decimal point
    total: typing.Counter = collections.Counter()
    solvates = []
    for i, part in enumerate(text.strip(".,;: ").split("·")):
        match = re.match(r"^(\d*\.?\d+)?(.*)$", part)
        multiplier = float(match.group(1)) if match.group(1) else 1.0
        counts = _expand(match.group(2))
        if not counts:
            return None, solvates
        if i > 0:
            name = match.group(2).strip("()")
            solvates.append((SOLVENT_NAMES.get(name, name), multiplier))
        for element, number in counts.items():
            total[element] += multiplier * number
    return dict(total), solvates


def _values(text: str) -> typing.Dict[str, float]:
    return {element: float(value.replace(",", ".")) for element, value in VALUE_RE.findall(text)}


def parse_articles(output_dir: str) -> typing.List[dict]:
    rows = []
    for path in tqdm(sorted(glob.glob(os.path.join(output_dir, "raw", "*.xml.gz"))), desc="Parsing"):
        xml = gzip.open(path, "rt", errors="ignore").read()
        for article in re.split(r"(?=<article[ >])", xml):
            pmcid = re.search(r'<article-id pub-id-type="pmc[^"]*">(?:PMC)?(\d+)<', article)
            year = re.search(r"<pub-date[^>]*>.*?<year>(\d{4})</year>", article, re.S)
            journal = re.search(r"<journal-title>([^<]+)<", article)
            text = re.sub(r"<sub>\s*([^<]*?)\s*</sub>", r"\1", article)  # keep subscripts glued to symbols
            text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text))
            for match in STATEMENT_RE.finditer(text):
                formula, solvates = parse_formula(match.group("formula"))
                rows.append(
                    {
                        "pmcid": pmcid.group(1) if pmcid else None,
                        "year": int(year.group(1)) if year else None,
                        "journal": journal.group(1) if journal else None,
                        "formula_text": match.group("formula").strip(),
                        "formula": formula,
                        "solvates": solvates,
                        "calc": _values(match.group("calc")),
                        "found": _values(match.group("found")),
                    }
                )
    return rows


def is_consistent(row: dict) -> bool:
    # Found may list fewer elements than Calcd (at least two)
    if not row["formula"] or not row["calc"] or len(row["found"]) < 2 or not set(row["found"]) <= set(row["calc"]):
        return False
    expected = mass_percent(row["formula"])
    return all(abs(expected.get(e, 0.0) - v) < CALC_TOLERANCE for e, v in row["calc"].items())



def noise_models_from_dataset(dataset_path: str, output_dir: str, max_abs_delta: float) -> None:
    """Noise models from the experimental dataset of ea/build_dataset.py (all analyses, with or without
    SMILES): ea_noise.json from compounds without metals (the molecules of the NMR data),
    ea_noise_complexes.json from metal compounds, ea_noise_all.json from all."""
    from .elemental_analysis import formula_counts_plain
    nonmetals = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I", "As", "Te"}
    rows = []
    for line in gzip.open(dataset_path, "rt"):
        r = json.loads(line)
        r["formula"] = formula_counts_plain(r["formula"])
        r["solvates"] = [tuple(x) for x in r["solvates"]]
        rows.append(r)
    groups = {
        "ea_noise.json": [r for r in rows if set(r["formula"]) <= nonmetals],
        "ea_noise_complexes.json": [r for r in rows if not set(r["formula"]) <= nonmetals],
        "ea_noise_all.json": rows,
    }
    for name, group in groups.items():
        model = build_noise_model(group, max_abs_delta, source=f"Experimental elemental analyses ({os.path.basename(dataset_path)}): {name}")
        json.dump(model, open(os.path.join(output_dir, name), "w"))
        summary = {k: v for k, v in model.items() if k not in ("deltas", "solvates")}
        summary["num_deltas"] = len(model["deltas"])
        print(name, json.dumps(summary))


def build_noise_model(rows: typing.List[dict], max_abs_delta: float, source: str = None) -> dict:
    unique = list({json.dumps([r["formula"], r["calc"], r["found"]], sort_keys=True): r for r in rows}.values())
    kept = [r for r in unique if is_consistent(r)]
    # The elements reported are those with a found value (a few papers list more calculated ones)

    present: typing.Counter = collections.Counter()
    reported: typing.Counter = collections.Counter()
    reported_absent: typing.Counter = collections.Counter()
    for r in kept:
        for element in ATOMIC_MASS:
            if r["formula"].get(element):
                present[element] += 1
                reported[element] += element in r["found"]
            elif element in r["found"]:
                reported_absent[element] += 1
    report_prob = {e: reported[e] / present[e] for e in present if present[e] >= 100 and reported[e]}
    # Mostly "N, 0.00" for compounds without nitrogen
    report_absent_prob = {e: reported_absent[e] / (len(kept) - present[e]) for e in report_prob if reported_absent[e]}

    neutral = [r for r in kept if not any(s in SALT_FORMERS for s, _ in r["solvates"])]
    solvated = [r for r in neutral if r["solvates"]]
    solvates = collections.Counter((s, m) for r in solvated for s, m in r["solvates"])

    deltas = []
    for r in kept:
        delta = {e: round(r["found"][e] - r["calc"][e], 2) for e in r["calc"] if e in r["found"]}
        # Larger deviations are typos (values of another compound), not measurement errors
        if all(abs(d) <= max_abs_delta for d in delta.values()):
            deltas.append(delta)
    return {
        "source": source or "PubMed Central open-access articles, 'Anal. Calcd ... Found ...' statements",
        "num_statements": len(rows),
        "num_unique": len(unique),
        "num_consistent": len(kept),
        "max_abs_delta": max_abs_delta,
        "report_prob": report_prob,
        "report_absent_prob": report_absent_prob,
        "solvate_prob": len(solvated) / len(neutral),
        "solvates": [
            [s, m, n] for (s, m), n in solvates.most_common() if n >= 3 and re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", s)
        ],
        "deltas": deltas,
    }


def prepare(args: argparse.Namespace) -> None:
    if args.from_dataset:
        noise_models_from_dataset(args.from_dataset, args.output_dir, args.max_abs_delta)
        return
    if not args.skip_download:
        download(args.output_dir, args.num_articles)
    rows = parse_articles(args.output_dir)
    with gzip.open(os.path.join(args.output_dir, "ea_pairs.jsonl.gz"), "wt") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    model = build_noise_model(rows, args.max_abs_delta)
    with open(os.path.join(args.output_dir, "ea_noise.json"), "w") as f:
        json.dump(model, f)
    summary = {k: v for k, v in model.items() if k != "deltas"}
    summary["num_deltas"] = len(model["deltas"])
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an elemental analysis noise model from PubMed Central articles.")
    parser.add_argument("--output-dir", type=str, default="data/ea", help="Articles, parsed pairs and the noise model.")
    parser.add_argument("--num-articles", type=int, default=5000, help="Number of articles to download.")
    parser.add_argument("--skip-download", action="store_true", help="Only parse articles already in <output-dir>/raw.")
    parser.add_argument("--from-dataset", help="Build the noise models from ea/build_dataset.py output (ea_dataset.jsonl.gz) instead.")
    parser.add_argument("--max-abs-delta", type=float, default=2.0, help="Drop Found - Calcd vectors with a larger deviation.")
    return parser.parse_args()
