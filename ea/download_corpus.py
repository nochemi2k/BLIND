#!/usr/bin/env python3
"""
Articles with elemental analyses: full texts and supporting information, for a fast machine.

1. PMCIDs from Europe PMC searches (several queries, saved after each one: a rerun continues);
   --skip-ids: articles whose full text was downloaded before (a JSON list, e.g.
   data/ea/downloaded_pmcids.json); for them only the supporting information is fetched.
2. For every article, from the PMC Open Access bucket on AWS (static files, no login,
   https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/):
     <output-dir>/raw/PMCxxx.xml.gz   full-text JATS XML (NCBI efetch when the article is not in the bucket)
     <output-dir>/si/PMCxxx.txt.gz    text of the supplementary PDF/DOCX files (pdftotext), paragraphs
                                      separated by blank lines; empty when there are none
   and article metadata in <output-dir>/si/meta.jsonl. Files already there are skipped.
The output folder is read by ea/build_dataset.py like the other corpora:
    python ea/build_dataset.py --input-dirs ... <output-dir>

Needs Python 3.8+ (standard library only) and pdftotext (poppler-utils). One file is enough on the
server together with ea/download_si.py next to it. Run:
    python ea/download_corpus.py --output-dir data/ea/corpus2 --skip-ids data/ea/downloaded_pmcids.json \
        --max-articles 60000 --num-workers 128
"""
import argparse
import concurrent.futures
import gzip
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from download_si import DEFAULT_QUERY as SI_QUERY
from download_si import MAX_FILE_SIZE, S3, docx_paragraphs, get, pdf_paragraphs, search

import re

EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={}"
# Statements in the text (most specific first: the order sets download priority), then articles with
# the experimental section in the supporting information
QUERIES = [
    '"Found: C"',
    '"Anal. Calcd"',
    '"Anal. Calc"',
    '"Anal. Found"',
    '"elemental analysis calcd"',
    '"Calcd. for" AND "found" AND "elemental analysis"',
    # Not '"Found (%)"' and the like: Europe PMC drops the punctuation and searches the bare word
    '"Calcd for" AND "Found" AND ("Anal" OR "elemental analysis")',
    '("elemental analysis" OR "elemental analyses" OR "microanalysis") AND ("Calcd" OR "calculated") AND "Found"',
    SI_QUERY,
]


def download(articles: list, output_dir: str, num_workers: int) -> None:
    counts, lock, start, last = {}, threading.Lock(), time.time(), [0.0]
    with concurrent.futures.ThreadPoolExecutor(num_workers) as pool:
        futures = [pool.submit(fetch, a, output_dir) for a in articles]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                status = future.result()
            except Exception:
                status = "failed"
            with lock:
                counts[status] = counts.get(status, 0) + 1
            if time.time() - last[0] > 30 or i == len(articles):
                last[0] = time.time()
                rate = i / (time.time() - start) * 60
                eta = (len(articles) - i) / max(rate, 1e-9)
                print(f"  download: {i}/{len(articles)} {counts} {rate:.0f}/min, ~{eta:.0f} min left", flush=True)


def run(output_dir: str, queries: list, skip: set, limit: int, num_workers: int) -> None:
    """Query by query: search, then download the new articles of that query right away."""
    path = os.path.join(output_dir, "articles.jsonl")
    done_path = os.path.join(output_dir, "queries_done.json")
    articles = {json.loads(line)["pmcid"]: json.loads(line) for line in open(path)} if os.path.exists(path) else {}
    done = json.load(open(done_path)) if os.path.exists(done_path) else []
    # Articles listed before (also by --seed) first: a rerun finishes them
    pending = list(articles.values())
    for query in [None] + queries:
        if query is not None:
            if query in done:
                continue
            print(f"query: {query}", flush=True)
            full = f"({query}) AND IN_PMC:y" if "IN_PMC" not in query else query
            pending = []
            for a in search(full, 10**6):
                if a["pmcid"] not in articles:
                    articles[a["pmcid"]] = a
                    pending.append(a)
            with open(path, "w") as f:
                for a in articles.values():
                    f.write(json.dumps(a) + "\n")
            with open(os.path.join(output_dir, "si", "meta.jsonl"), "w") as f:
                for a in articles.values():
                    f.write(json.dumps(a) + "\n")
            print(f"{len(pending)} new articles ({len(articles)} in total)", flush=True)
        room = max(0, limit - (len(articles) - len(pending)))
        pending = pending[:room]
        for a in pending:
            a["have_xml"] = a["pmcid"] in skip
        if pending:
            download(pending, output_dir, num_workers)
        if query is not None:
            done.append(query)
            json.dump(done, open(done_path, "w"))
        if len(articles) >= limit:
            break


def bucket_files(pmcid: str) -> dict:
    """{name: (key, size)} of the latest version of the article in the bucket ({} if absent)."""
    listing = get(S3 + "?" + urllib.parse.urlencode({"list-type": 2, "prefix": pmcid + ".", "max-keys": 1000})).decode()
    objects = re.findall(r"<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>", listing)
    if not objects:
        return {}
    version = max({k.split("/")[0] for k, _ in objects}, key=lambda v: int(v.split(".")[-1]))
    return {k.split("/", 1)[1]: (k, int(s)) for k, s in objects if k.startswith(version + "/")}


def fetch(article: dict, output_dir: str) -> str:
    pmcid = article["pmcid"]
    xml_path = os.path.join(output_dir, "raw", f"{pmcid}.xml.gz")
    si_path = os.path.join(output_dir, "si", f"{pmcid}.txt.gz")
    need_xml = not article.get("have_xml") and not os.path.exists(xml_path)
    if not need_xml and os.path.exists(si_path):
        return "exists"
    files = bucket_files(pmcid)
    version = next(iter(files.values()))[0].split("/")[0] if files else None
    if need_xml:
        if version and f"{version}.xml" in files:
            xml = get(S3 + files[f"{version}.xml"][0])
        else:
            xml = get(EFETCH.format(pmcid.replace("PMC", "")))
        with gzip.open(xml_path + ".tmp", "wb") as f:
            f.write(xml)
        os.replace(xml_path + ".tmp", xml_path)
    if not os.path.exists(si_path):
        texts = []
        for name, (key, size) in sorted(files.items()):
            if name == f"{version}.pdf" or size > MAX_FILE_SIZE or not re.search(r"\.(pdf|docx)$", name, re.I):
                continue
            try:
                data = get(S3 + urllib.parse.quote(key))
                texts.append(pdf_paragraphs(data) if name.lower().endswith(".pdf") else docx_paragraphs(data))
            except Exception:
                continue
        with gzip.open(si_path + ".tmp", "wt") as f:
            f.write("\n\n".join(texts))
        os.replace(si_path + ".tmp", si_path)
    return "in_bucket" if files else "efetch"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--output-dir", default="data/ea/corpus2")
    parser.add_argument("--skip-ids", help="JSON list of PMCIDs whose full text is already downloaded")
    parser.add_argument("--max-articles", type=int, default=60000)
    parser.add_argument("--seed", nargs="*", help="PMCID lists to download before the searches")
    parser.add_argument("--num-workers", type=int, default=64)
    args = parser.parse_args()
    for sub in ("raw", "si"):
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)
    skip = set(json.load(open(args.skip_ids))) if args.skip_ids else set()

    if args.seed:
        # Lists of PMCIDs found before (JSON lists or jsonl with "pmcid"), downloaded first
        path = os.path.join(args.output_dir, "articles.jsonl")
        known = {json.loads(line)["pmcid"] for line in open(path)} if os.path.exists(path) else set()
        with open(path, "a") as f:
            for seed in args.seed:
                ids = [json.loads(l)["pmcid"] for l in open(seed)] if seed.endswith(".jsonl") else json.load(open(seed))
                for pmcid in ids:
                    if pmcid not in known:
                        known.add(pmcid)
                        f.write(json.dumps({"pmcid": pmcid}) + "\n")
    run(args.output_dir, QUERIES, skip, args.max_articles, args.num_workers)
    print("done; failed articles are retried by running the same command again", flush=True)


if __name__ == "__main__":
    main()
