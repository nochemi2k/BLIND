#!/usr/bin/env python3
"""
Supplementary materials (supporting information) of PubMed Central articles, as plain text.

Experimental sections, with the elemental analyses, are often only in the supporting information.
The PMCIDs come from a Europe PMC search; for each article the supplementary files are fetched from
the PMC Open Access bucket on AWS (static files, fast; https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/) or,
for articles not there, from the Europe PMC REST service (a zip, built per request: slow), and the text of its PDF (pdftotext, poppler) and DOCX files is
written to <output-dir>/si/<PMCID>.txt.gz, paragraphs separated by blank lines; article metadata goes to
<output-dir>/si/meta.jsonl. modalities/ea/build_dataset.py parses these files together with the full texts.

Run from the repository root:
    python modalities/ea/download_si.py --output-dir data/ea/si_corpus --max-articles 15000
"""
import argparse
import concurrent.futures
import glob
import gzip
import io
import json
import os
import re
import statistics
import subprocess
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import zipfile

SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
SUPPLEMENTARY = "https://www.ebi.ac.uk/europepmc/webservices/rest/{}/supplementaryFiles?includeInlineImage=false"
DEFAULT_QUERY = (
    '("elemental analysis" OR "elemental analyses" OR "microanalysis" OR "microanalyses" OR "Anal. Calcd" OR "Found: C") '
    'AND ("Supporting Information" OR "Supplementary Information" OR "Supplementary Material" OR "ESI") '
    'AND ("synthesis" OR "synthesized" OR "synthesised") AND HAS_SUPPL:y AND OPEN_ACCESS:y AND IN_PMC:y'
)
S3 = "https://pmc-oa-opendata.s3.amazonaws.com/"
USER_AGENT = {"User-Agent": "blind-ea-research/0.1"}
# Larger files are mostly spectra images; skipped before download (the size is in the bucket listing)
MAX_FILE_SIZE = 15 * 2**20


def get(url: str, tries: int = 4, timeout: int = 300) -> bytes:
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=USER_AGENT), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            time.sleep(10 * (i + 1))
        except Exception:
            time.sleep(10 * (i + 1))
    raise RuntimeError(url)


def search(query: str, limit: int) -> list:
    results, cursor = [], "*"
    while len(results) < limit:
        params = {"query": query, "format": "json", "pageSize": 1000, "resultType": "lite", "cursorMark": cursor}
        data = json.loads(get(SEARCH + urllib.parse.urlencode(params)))
        if cursor == "*":
            print(f"hitCount {data['hitCount']}", flush=True)
        page = data["resultList"]["result"]
        print(f"  search: {len(results) + len(page)}/{data['hitCount']}", flush=True)
        results += [
            {"pmcid": r["pmcid"], "doi": r.get("doi"), "year": int(r["pubYear"]) if r.get("pubYear") else None, "journal": r.get("journalTitle")}
            for r in page if r.get("pmcid")
        ]
        if not page or data.get("nextCursorMark") in (None, cursor):
            break
        cursor = data["nextCursorMark"]
    return results[:limit]


def pdf_paragraphs(data: bytes) -> str:
    """pdftotext lines joined into paragraphs: a paragraph ends at a blank line or a short line."""
    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        f.write(data)
        f.flush()
        text = subprocess.run(["pdftotext", "-enc", "UTF-8", f.name, "-"], capture_output=True, timeout=300).stdout.decode("utf-8", "ignore")
    lines = [line.strip() for line in text.replace("\f", "\n").split("\n")]
    # Typical full line: median of the longer lines (table-of-contents dot leaders would skew a high quantile)
    lengths = [len(line) for line in lines if len(line) > 40 and "....." not in line]
    full = statistics.median(lengths) if len(lengths) >= 10 else 80
    paragraphs, current = [], []
    for line in lines:
        if not line:
            if current:
                paragraphs.append(current)
                current = []
            continue
        # Hyphenated line ends are joined without a space
        if current and current[-1].endswith("-") and not current[-1].endswith(" -"):
            current[-1] = current[-1] + line
        else:
            current.append(line)
        # A short line ends a paragraph unless it ends mid-sentence (",", ";", ":", "-"): headings with
        # the compound name become paragraphs of their own
        if len(line) < 0.7 * full and not re.search(r"[,;:\-–(]$", line):
            paragraphs.append(current)
            current = []
    if current:
        paragraphs.append(current)
    return "\n\n".join(" ".join(p) for p in paragraphs)


def docx_paragraphs(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("word/document.xml").decode("utf-8", "ignore")
    paragraphs = []
    for p in re.findall(r"<w:p[ >].*?</w:p>", xml, re.S):
        text = "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", p))
        if text.strip():
            paragraphs.append(text.strip())
    return "\n\n".join(paragraphs)


def s3_files(pmcid: str) -> list:
    """Supplementary PDF/DOCX keys of the latest version of the article in the PMC bucket ([] if absent)."""
    listing = get(S3 + "?" + urllib.parse.urlencode({"list-type": 2, "prefix": pmcid + ".", "max-keys": 1000})).decode()
    objects = re.findall(r"<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>", listing)
    if not objects:
        return []
    version = max({k.split("/")[0] for k, _ in objects}, key=lambda v: int(v.split(".")[-1]))
    return [
        k for k, size in objects
        if k.startswith(version + "/") and re.search(r"\.(pdf|docx)$", k, re.I) and not k.endswith(f"/{version}.pdf") and int(size) <= MAX_FILE_SIZE
    ]


def _in_bucket(pmcid: str) -> bool:
    listing = get(S3 + "?" + urllib.parse.urlencode({"list-type": 2, "prefix": pmcid + ".", "max-keys": 1})).decode()
    return "<Key>" in listing


def fetch(article: dict, si_dir: str) -> str:
    path = os.path.join(si_dir, f"{article['pmcid']}.txt.gz")
    if os.path.exists(path):
        return "exists"
    texts = []
    keys = s3_files(article["pmcid"])
    in_bucket = bool(keys) or _in_bucket(article["pmcid"])
    for key in keys:
        try:
            data = get(S3 + urllib.parse.quote(key))
            if len(data) <= MAX_FILE_SIZE:
                texts.append(pdf_paragraphs(data) if key.lower().endswith(".pdf") else docx_paragraphs(data))
        except Exception:
            continue
    data = b""
    if not in_bucket:
        try:
            data = get(SUPPLEMENTARY.format(article["pmcid"]))
        except urllib.error.HTTPError:
            pass
    if data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for info in z.infolist():
                name = info.filename.lower()
                if info.file_size > MAX_FILE_SIZE:
                    continue
                try:
                    if name.endswith(".pdf"):
                        texts.append(pdf_paragraphs(z.read(info)))
                    elif name.endswith(".docx"):
                        texts.append(docx_paragraphs(z.read(info)))
                except Exception:
                    continue
    with gzip.open(path, "wt") as f:  # an empty file marks an article without usable files
        f.write("\n\n".join(texts))
    return "ok" if texts else "empty"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--output-dir", default="data/ea/si_corpus")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--max-articles", type=int, default=15000)
    parser.add_argument("--num-workers", type=int, default=32)
    args = parser.parse_args()
    si_dir = os.path.join(args.output_dir, "si")
    os.makedirs(si_dir, exist_ok=True)

    meta_path = os.path.join(si_dir, "meta.jsonl")
    if os.path.exists(meta_path):
        articles = [json.loads(line) for line in open(meta_path)]
    else:
        articles = search(args.query, args.max_articles)
        with open(meta_path, "w") as f:
            for a in articles:
                f.write(json.dumps(a) + "\n")
    print(f"articles {len(articles)}, done {len(glob.glob(os.path.join(si_dir, 'PMC*.txt.gz')))}", flush=True)

    counts, lock = {}, threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(args.num_workers) as pool:
        futures = [pool.submit(fetch, a, si_dir) for a in articles]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                status = future.result()
            except Exception:
                status = "failed"
            with lock:
                counts[status] = counts.get(status, 0) + 1
            if i % 200 == 0:
                print(f"{i}/{len(articles)} {counts}", flush=True)
    print(f"done {counts}", flush=True)


if __name__ == "__main__":
    main()
