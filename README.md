# Streamlit Catalog Mapper — Human‑in‑the‑Loop

Map an IKEA taxonomy to a retailer catalog using leaf‑first overrides, whole‑path TF‑IDF similarity, room gating, synonyms, and optional URL verification. Export an augmented XLSX suitable for review and downstream use.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Inputs
- IKEA taxonomy (XLSX): columns HFBC/L1, PRAC/L2, PAC/L3; optional Examples, human_label.
- Retailer catalog (XLSX): either Name+Url OR Cat1..Cat5+Url (embedded headers supported).

## Output
Original IKEA columns plus:
- `<retailer>_catalog_path`
- `<retailer>_url`
- `<retailer>_score`
- `<retailer>_method`
and audit fields (`url_live`, `match_reason`, `matched_tokens`, `url_source`).
