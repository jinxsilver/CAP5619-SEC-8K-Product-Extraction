# SEC 8-K New Product Extraction with a Local LLM

**CAP5619 | Author: Frank Soto**

This pipeline reads SEC Form 8-K filings from S&P 500 companies. It uses a local large language model (Llama 3.1 8B through Ollama) to run Named Entity Recognition for **ORGANIZATION** and **PRODUCT**, and saves the newly announced products as a CSV.

## Output

[`product_releases.csv`](product_releases.csv) has 110 product releases from 15 companies, in this format:

```
company_name, stock_name, filing_time, new_product, product_description
"Apple Inc.", "AAPL", "2026-04-30 16:30:41", "M4-powered iPad Air", "The latest iPad Air model, powered by the M4 chip."
```

The same data in the assignment's `|`-separated display format is in [`product_releases_table.txt`](product_releases_table.txt).

## How it works

1. **Ticker → CIK:** `https://www.sec.gov/files/company_tickers.json`
2. **CIK → 8-K filings:** `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=8-K&count={count}&output=atom`, parsed with BeautifulSoup
3. **Press releases:** downloads the EX-99 exhibits of 8-Ks with Items 2.02, 7.01 or 8.01 and converts them to text
4. **Preprocessing:** keeps only the headline and the sentences around launch keywords (launched, introduced, unveiled, now available, FDA approved…)
5. **NER:** spaCy `en_core_web_sm` proposes ORG/PRODUCT candidates, and Llama 3.1 8B performs the final entity labeling with JSON-schema constrained output
6. **Validation:** the product name must appear in the filing, there must be a launch-evidence sentence, generic names are rejected, descriptions are under 180 characters, and results are de-duplicated

## Files

| File | Purpose |
|---|---|
| `sec_8k_product_extraction.ipynb` | Notebook that walks through each step and runs the full pipeline |
| `sec_8k_product_extractor.py` | All pipeline code, also runnable from the command line |
| `product_releases.csv` | Structured output |
| `product_releases_table.txt` | Pipe-separated display version of the output |
| `run_stats.json` | Run statistics (companies, filings, documents analyzed) |

## Setup

```bash
brew install ollama            # or see https://ollama.com/download
ollama serve &
ollama pull llama3.1:8b
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The SEC requires a contact User-Agent. Set it to your own name and e-mail:

```bash
export SEC_USER_AGENT="Your Name your_email@example.edu"
```

## Run

```bash
python sec_8k_product_extractor.py --target 100
```

Or open `sec_8k_product_extraction.ipynb` and run all cells. The pipeline caches SEC responses and LLM answers in `cache/` (not committed), so a fresh clone downloads and queries everything once, at about 20 seconds per filing on an Apple M4 Max.
