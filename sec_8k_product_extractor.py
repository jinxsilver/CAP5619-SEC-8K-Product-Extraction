"""
SEC 8-K New Product Extraction with a Local LLM (Ollama + Llama 3.1 8B)
CAP5619 - Single-student assignment
Frank Soto

Pipeline
    1. Map S&P 500 tickers -> CIK        (https://www.sec.gov/files/company_tickers.json)
    2. List each company's recent 8-Ks    (EDGAR browse-edgar Atom feed)
    3. Download the press-release exhibits (EX-99.x) and parse them with BeautifulSoup
    4. Preprocess: keep only passages that mention a launch / new product
    5. NER: spaCy proposes ORG / PRODUCT candidates; the LLM performs the final
       entity labeling (ORGANIZATION, PRODUCT) and writes the description
    6. Validate (product must appear in the source text, description < 180 chars),
       de-duplicate, and save CSV + pipe-delimited table + run statistics

Run:
    ollama pull llama3.1:8b
    python sec_8k_product_extractor.py               # stops after >= 100 product rows
    python sec_8k_product_extractor.py --target 150  # collect more rows
"""

import argparse
import csv
import json
import os
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

try:
    import spacy
    NLP = spacy.load("en_core_web_sm")
except (ImportError, OSError):  # spaCy is optional: the LLM still does the NER
    NLP = None

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
# The SEC requires a descriptive User-Agent with a contact e-mail.
# Set SEC_USER_AGENT="Your Name your_email@ucf.edu" before running.
USER_AGENT = os.environ.get("SEC_USER_AGENT", "Frank Soto fsoto@ucf.edu")
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
OUTPUT_CSV = BASE_DIR / "product_releases.csv"
OUTPUT_TABLE = BASE_DIR / "product_releases_table.txt"
STATS_JSON = BASE_DIR / "run_stats.json"

MAX_DESC_CHARS = 179          # "less than 180 characters"
MAX_LLM_CONTEXT_CHARS = 7000  # prompt performance tweaking
SEC_DELAY = 0.15              # SEC fair-access limit is 10 requests / second

# 8-K items that normally carry a press release (Exhibit 99)
PRESS_RELEASE_ITEMS = ("2.02", "7.01", "8.01")

# S&P 500 constituents that regularly announce products (tech, consumer,
# healthcare, industrial). Order = processing order.
SP500_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AMD", "INTC", "QCOM", "AVGO",
    "TXN", "MU", "ADBE", "CRM", "ORCL", "CSCO", "IBM", "NOW", "INTU", "PANW",
    "CRWD", "FTNT", "ANET", "HPQ", "HPE", "DELL", "TSLA", "F", "GM", "NKE",
    "PG", "KO", "PEP", "MDLZ", "HSY", "GIS", "LLY", "PFE", "MRK", "ABBV",
    "JNJ", "BMY", "AMGN", "GILD", "REGN", "VRTX", "MRNA", "ABT", "MDT", "BSX",
    "SYK", "ISRG", "EW", "DXCM", "BA", "LMT", "GE", "HON", "MMM", "DE",
    "CAT", "SBUX", "MCD", "CMG", "DIS", "NFLX", "GRMN", "ZBRA", "KEYS", "ADSK",
]

PRODUCT_KEYWORDS = re.compile(
    r"\b(launch(?:ed|es|ing)?|introduc(?:e|ed|es|ing)|unveil(?:ed|s|ing)?|"
    r"debut(?:ed|s)?|new (?:product|platform|model|service|device|drug|therapy|"
    r"version|line|feature|chip|processor|system|offering)s?|now available|"
    r"generally available|general availability|began shipping|start(?:ed)? shipping|"
    r"FDA approv(?:al|ed)|approved|announc(?:e|ed|es|ing) (?:the )?(?:availability|release))\b",
    re.IGNORECASE,
)

# Some EX-99 exhibits are XHTML; parsing them as HTML is intended, so hide the warning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

session = requests.Session()
session.headers.update(HEADERS)



# 1-3. Data acquisition (SEC EDGAR)
def sec_get(url, cache_name=None):
    """GET a SEC URL politely, on-disk cache so reruns are instant."""
    if cache_name:
        cache_file = CACHE_DIR / "http" / cache_name
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8", errors="ignore")
    for attempt in range(5):
        time.sleep(SEC_DELAY)
        resp = session.get(url, timeout=30)
        if resp.status_code == 200:
            text = resp.text
            if cache_name:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(text, encoding="utf-8")
            return text
        if resp.status_code in (403, 429, 503):
            time.sleep(5 * (attempt + 1))  # back off when throttled
            continue
        resp.raise_for_status()
    raise RuntimeError(f"SEC request failed: {url}")


def load_ticker_map():
    """ticker -> {'cik': '0000320193', 'title': 'Apple Inc.'}"""
    url = "https://www.sec.gov/files/company_tickers.json"
    data = json.loads(sec_get(url, "company_tickers.json"))
    return {
        row["ticker"].upper(): {"cik": str(row["cik_str"]).zfill(10), "title": row["title"]}
        for row in data.values()
    }


def get_8k_filings(cik, count=20):
    """Parse the EDGAR Atom feed of a company's most recent 8-K filings."""
    search_url = (
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
        f"&CIK={cik}&type=8-K&count={count}&output=atom"
    )
    feed = BeautifulSoup(sec_get(search_url, f"feed_{cik}_{count}.xml"), "xml")
    filings = []
    for entry in feed.find_all("entry"):
        content = entry.find("content")
        if content is None or content.find("filing-type").text.strip() != "8-K":
            continue  # skip 8-K/A amendments
        filings.append({
            "accession": content.find("accession-number").text.strip(),
            "filing_date": content.find("filing-date").text.strip(),
            "updated": entry.find("updated").text.strip(),  # acceptance timestamp
            "index_url": content.find("filing-href").text.strip(),
            "items": (content.find("items-desc").text if content.find("items-desc") else ""),
        })
    return filings


def html_to_text(html):
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "table"]):  # tables = financial statements
        tag.decompose()
    text = soup.get_text(" ")
    return re.sub(r"\s+", " ", text).strip()


def get_press_release_text(filing):
    """Download the EX-99.x exhibits (press releases) of an 8-K filing."""
    acc = filing["accession"]
    index_html = sec_get(filing["index_url"], f"index_{acc}.htm")
    soup = BeautifulSoup(index_html, "lxml")
    table = soup.find("table", class_="tableFile")
    if table is None:
        return ""
    texts = []
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        doc_type = cells[3].get_text(strip=True).upper()
        link = cells[2].find("a")
        if not link or not doc_type.startswith("EX-99"):
            continue
        href = link["href"].replace("/ix?doc=", "")
        if not href.lower().endswith((".htm", ".html", ".txt")):
            continue  # images, PDFs
        name = f"doc_{acc}_{Path(href).name}"
        texts.append(html_to_text(sec_get("https://www.sec.gov" + href, name)))
    return "\n\n".join(texts)



# 4. Preprocessing: keep only product-related passages
def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"“(])", text) if len(s.strip()) > 20]


def extract_product_passages(text, max_chars=MAX_LLM_CONTEXT_CHARS):
    """Return the sentences around product-launch keywords and the headline."""
    sentences = split_sentences(text)
    if not sentences:
        return ""
    hits = [i for i, s in enumerate(sentences) if PRODUCT_KEYWORDS.search(s)]
    if not hits:
        return ""
    keep = {0, 1}  # headline / dateline usually names the product
    for i in hits:
        keep.update({i - 1, i, i + 1})
    passage, total = [], 0
    for i in sorted(k for k in keep if 0 <= k < len(sentences)):
        if total + len(sentences[i]) > max_chars:
            break
        passage.append(sentences[i])
        total += len(sentences[i])
    return " ".join(passage)


# 5. Named Entity Recognition + LLM extraction
def spacy_entities(text):
    """Candidate entities from spaCy's statistical NER (hints for the LLM)."""
    if NLP is None:
        return {"ORG": [], "PRODUCT": []}
    doc = NLP(text[:20000])
    ents = {"ORG": set(), "PRODUCT": set()}
    for ent in doc.ents:
        # spaCy's small model tags product names as ORG/PERSON/WORK_OF_ART
        if ent.label_ == "ORG":
            ents["ORG"].add(ent.text)
        elif ent.label_ in ("PRODUCT", "WORK_OF_ART", "PERSON"):
            ents["PRODUCT"].add(ent.text)
    return {k: sorted(v)[:25] for k, v in ents.items()}


SYSTEM_PROMPT = """You are a financial-document Named Entity Recognition (NER) system.
You read excerpts of SEC Form 8-K press releases and extract NEW PRODUCT releases.

Entity types:
- ORGANIZATION: the company that filed the 8-K.
- PRODUCT: a specific, named product, service, drug, device, model, platform or
  software that the company says it launched, introduced, unveiled, released,
  began shipping, made available, or received approval to sell, in this document.

Rules:
- Only include products that are NEW (launched/announced/approved recently).
- Do NOT include: financial results, revenue lines, business segments, generic
  categories ("AI solutions"), acquisitions, executives, dividends, buybacks,
  partnerships without a named product, or products merely mentioned in passing.
- Facilities, campuses, data centers, contracts and customer deals are NOT products.
- Use the product name exactly as written in the text.
- company_name: the company's full name as written in the press release
  (e.g. "Apple Inc.", "NVIDIA Corporation").
- product_description: ONE complete, self-contained sentence saying what the
  product is and what it does, under 180 characters. Never a fragment such as
  "new model" or "general availability".
- launch_evidence: copy the sentence from the text that shows the product is new
  (it should contain words like launched, introduced, unveiled, approved, available).
  Products that are only discussed in sales or revenue figures are NOT new: leave them out.
- Return at most 5 products: the most significant launches in the document.
- If there is no new product, return an empty "products" list.

Example
Text: "Apple Inc. today announced the launch of its latest MacBook Pro, featuring
a new M3 processor that delivers up to 60% faster performance."
Output: {"company_name": "Apple Inc.", "products": [{"product_name": "MacBook Pro",
"product_description": "Apple's latest pro laptop, powered by the new M3 chip with up to 60% faster performance.",
"launch_evidence": "Apple Inc. today announced the launch of its latest MacBook Pro"}]}

Respond with JSON only."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "company_name": {"type": "string"},
        "products": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string"},
                    "product_description": {"type": "string"},
                    "launch_evidence": {"type": "string"},
                },
                "required": ["product_name", "product_description", "launch_evidence"],
            },
        },
    },
    "required": ["company_name", "products"],
}


def query_llm(company_hint, passage, entities):
    """Ask the local Ollama model to label entities and return JSON."""
    user_prompt = (
        f"Filer ticker symbol: {company_hint}\n"
        f"spaCy NER candidates - ORG: {entities['ORG']}\n"
        f"spaCy NER candidates - PRODUCT: {entities['PRODUCT']}\n"
        "(Candidates may be wrong; verify against the text.)\n\n"
        f"8-K text excerpt:\n\"\"\"\n{passage}\n\"\"\"\n\n"
        'Return: {"company_name": "...", "products": [{"product_name": "...", '
        '"product_description": "...", "launch_evidence": "..."}]}'
    )
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "format": RESPONSE_SCHEMA,  # Ollama output (JSON schema)
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 900},
    }
    for attempt in range(2):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=240)
            resp.raise_for_status()
            break
        except requests.Timeout:
            if attempt == 1:
                raise
    content = resp.json()["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        return json.loads(match.group(0)) if match else {"company_name": "", "products": []}


# 6. Validation / formatting
def truncate(text, limit=MAX_DESC_CHARS):
    text = re.sub(r"\s+", " ", text or "").strip().strip('"')
    if len(text) <= limit:
        return text
    cut = text[: limit - 3].rsplit(" ", 1)[0].rstrip(",;:-")
    return cut + "..."


def normalize(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def product_in_text(product, text):
    """Reject hallucinations: the product name must appear in the source text."""
    p = normalize(product)
    return len(p) >= 2 and p in normalize(text)


GENERIC_NAMES = {
    "newproduct", "products", "product", "services", "platform", "software",
    "aisolutions", "solutions", "revenue", "netsales", "dividend",
    "aitools", "aiskills", "tools", "features", "aiagents", "aimodels",
}
GENERIC_SUFFIX = re.compile(r"\b(AI )?(solutions|tools|skills|offerings|capabilities)$", re.IGNORECASE)


def is_generic(name):
    """Categories such as "AI tools" or "ServiceNow AI solutions" are not product names."""
    return normalize(name) in GENERIC_NAMES or bool(GENERIC_SUFFIX.search(name))


def format_filing_time(updated):
    """'2026-11-02T16:30:42-05:00' -> '2026-11-02 16:30:42'"""
    try:
        return datetime.fromisoformat(updated).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return updated[:10]


def pick_company_name(llm_name, sec_title):
    """Pick the LLM's NER result when it agrees with the SEC registrant name."""
    first = normalize(sec_title.split()[0]) if sec_title else ""
    if llm_name and first and first[:4] in normalize(llm_name) and len(llm_name) < 80:
        return llm_name.strip()
    return sec_title


def clean_sec_name(title):
    """'ADVANCED MICRO DEVICES INC' -> 'Advanced Micro Devices, Inc.'"""
    if title != title.upper():
        return title
    words = [w if w in ("NVIDIA", "IBM", "AT&T") else w.capitalize() for w in title.split()]
    name = " ".join(words)
    name = re.sub(r" (Inc|Corp|Co)\.?$", lambda m: {"Inc": ", Inc.", "Corp": " Corporation", "Co": " Company"}[m.group(1)], name)
    return name


def harmonize_company_names(rows, ticker_map):
    """Use one company name per ticker: the most frequent name the LLM extracted."""
    by_ticker = {}
    for r in rows:
        by_ticker.setdefault(r["stock_name"], []).append(r["company_name"])
    for ticker, names in by_ticker.items():
        sec_title = ticker_map[ticker]["title"]
        llm_names = [n for n in names if n != sec_title]
        best = max(set(llm_names), key=llm_names.count) if llm_names else clean_sec_name(sec_title)
        for r in rows:
            if r["stock_name"] == ticker:
                r["company_name"] = best


def write_outputs(rows):
    fields = ["company_name", "stock_name", "filing_time", "new_product", "product_description"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    widths = {"company_name": 28, "stock_name": 10, "filing_time": 19, "new_product": 32}
    header = ["Company Name", "Stock Name", "Filing Time", "New Product", "Product Description"]
    lines = [" | ".join(h.ljust(w) for h, w in zip(header, list(widths.values()) + [0])).rstrip()]
    lines.append("-" * 140)
    for r in rows:
        cells = [r[k][:w].ljust(w) for k, w in widths.items()] + [r["product_description"]]
        lines.append(" | ".join(cells))
    OUTPUT_TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Main loop code for assignment
def run(target_rows=100, filings_per_company=20, tickers=SP500_TICKERS):
    llm_cache_file = CACHE_DIR / "llm_cache.json"
    CACHE_DIR.mkdir(exist_ok=True)
    llm_cache = json.loads(llm_cache_file.read_text()) if llm_cache_file.exists() else {}

    ticker_map = load_ticker_map()
    rows, seen = [], set()
    stats = {
        "model": MODEL, "companies_processed": 0, "companies_with_products": set(),
        "filings_listed": 0, "filings_downloaded": 0, "filings_sent_to_llm": 0,
        "filings_with_products": 0, "products_rejected_validation": 0,
        "llm_seconds": 0.0, "llm_calls_uncached": 0, "started": datetime.now().isoformat(timespec="seconds"),
    }

    for ticker in tickers:
        if len(rows) >= target_rows:
            break
        info = ticker_map.get(ticker)
        if not info:
            print(f"[skip] {ticker}: not in SEC ticker map")
            continue
        stats["companies_processed"] += 1
        try:
            filings = get_8k_filings(info["cik"], filings_per_company)
        except (requests.RequestException, RuntimeError) as exc:
            print(f"[skip] {ticker}: {exc}")
            continue
        stats["filings_listed"] += len(filings)
        print(f"\n== {ticker} ({info['title']}) - {len(filings)} 8-K filings")

        for filing in filings:
            if len(rows) >= target_rows:
                break
            if not any(item in filing["items"] for item in PRESS_RELEASE_ITEMS):
                continue
            try:
                text = get_press_release_text(filing)
            except (requests.RequestException, RuntimeError) as exc:
                print(f"   [skip] {filing['accession']}: {exc}")
                continue
            stats["filings_downloaded"] += 1
            passage = extract_product_passages(text)
            if not passage:
                continue

            acc = filing["accession"]
            if acc in llm_cache:
                result = llm_cache[acc]
            else:
                t0 = time.time()
                try:
                    result = query_llm(ticker, passage, spacy_entities(passage))
                except (requests.RequestException, ValueError) as exc:
                    print(f"   [llm error] {acc}: {exc}")
                    continue
                stats["llm_seconds"] += time.time() - t0
                stats["llm_calls_uncached"] += 1
                llm_cache[acc] = result
                llm_cache_file.write_text(json.dumps(llm_cache, indent=1))
            stats["filings_sent_to_llm"] += 1

            found = 0
            for prod in result.get("products", []) or []:
                name = re.sub(r"\s+", " ", (prod.get("product_name") or "")).strip(" .\"'")
                key = (ticker, normalize(name))
                if (not name or is_generic(name) or len(name) > 80
                        or not product_in_text(name, text)):
                    stats["products_rejected_validation"] += 1
                    continue
                desc = truncate(prod.get("product_description", ""))
                evidence = prod.get("launch_evidence") or ""
                if not (PRODUCT_KEYWORDS.search(evidence) or re.search(r"\bnew\b", evidence, re.I)):
                    stats["products_rejected_validation"] += 1  # no proof it is a launch
                    continue
                if len(desc) < 25:  # fragment, not a description
                    stats["products_rejected_validation"] += 1
                    continue
                if key in seen or found >= 5:
                    continue
                seen.add(key)
                rows.append({
                    "company_name": pick_company_name(result.get("company_name", ""), info["title"]),
                    "stock_name": ticker,
                    "filing_time": format_filing_time(filing["updated"]),
                    "new_product": name,
                    "product_description": desc,
                    "accession": acc,
                })
                found += 1
            if found:
                stats["filings_with_products"] += 1
                stats["companies_with_products"].add(ticker)
            print(f"   {filing['filing_date']} {acc}: {found} product(s)  [total {len(rows)}]")
        write_outputs(rows)  # save a checkpoint after every company

    harmonize_company_names(rows, ticker_map)
    stats["companies_with_products"] = sorted(stats["companies_with_products"])
    stats["rows_written"] = len(rows)
    stats["llm_seconds"] = round(stats["llm_seconds"], 1)
    stats["finished"] = datetime.now().isoformat(timespec="seconds")
    write_outputs(rows)
    STATS_JSON.write_text(json.dumps(stats, indent=2))
    print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV.name}")
    return rows, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", type=int, default=100, help="stop after this many product rows")
    parser.add_argument("--per-company", type=int, default=20, help="8-K filings to scan per company")
    args = parser.parse_args()
    run(args.target, args.per_company)
