
from __future__ import annotations
import re
import pandas as pd

KNOWN_IKEA_L1 = {"hfbc", "home furnishing business caption", "level1", "l1"}
KNOWN_IKEA_L2 = {"prac", "product range area caption", "level2", "l2"}
KNOWN_IKEA_L3 = {"pac", "product area caption", "level3", "l3"}

def read_ikea_xlsx(path: str) -> pd.DataFrame:
    try:
        df0 = pd.read_excel(path, sheet_name=0)
        if _looks_like_ikea(df0):
            return df0
    except Exception:
        pass
    raw = pd.read_excel(path, sheet_name=0, header=None)
    header_idx = None
    for i in range(min(20, len(raw))):
        joined = " ".join(raw.iloc[i].astype(str).str.lower().tolist())
        if any(k in joined for k in ("hfbc", "product range area", "product area caption", "prac", "pac")):
            header_idx = i
            break
    if header_idx is None:
        df = pd.read_excel(path, sheet_name=0)
        return df
    df = pd.read_excel(path, sheet_name=0, header=header_idx)
    return df

def _looks_like_ikea(df: pd.DataFrame) -> bool:
    cols = {str(c).strip().lower() for c in df.columns}
    return (cols & KNOWN_IKEA_L1) and (cols & KNOWN_IKEA_L2) and (cols & KNOWN_IKEA_L3)

def read_retailer_xlsx(path: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(path, sheet_name=0)
        cols_norm = {re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()): c for c in df.columns}
        name_c = cols_norm.get("name")
        url_c = cols_norm.get("url") or cols_norm.get("link") or cols_norm.get("href")
        if name_c is not None and url_c is not None:
            out = df[[name_c, url_c]].rename(columns={name_c: "Name", url_c: "Url"})
            return _clean_retailer(out)
        cat_cols = _detect_cat_cols(df)
        if cat_cols:
            name_series = df.apply(lambda r: " > ".join([str(r[c]).strip() for c in cat_cols if str(r[c]).strip() and str(r[c]).strip().lower() != "nan"]), axis=1)
            url_c2 = _pick_url_like_col(df)
            out = pd.DataFrame({"Name": name_series, "Url": df[url_c2]})
            return _clean_retailer(out)
    except Exception:
        pass
    raw = pd.read_excel(path, sheet_name=0, header=None)
    header_idx = None
    for i in range(min(200, len(raw))):
        row = raw.iloc[i].astype(str).str.strip()
        if row.str.lower().eq("url").any() and row.str.lower().str.contains(r"^cat[1-5]$", regex=True).any():
            header_idx = i
            break
    if header_idx is not None:
        data = raw.iloc[header_idx + 1:].reset_index(drop=True)
        hdr = raw.iloc[header_idx].astype(str).str.strip()
        col_idx = {val: idx for idx, val in enumerate(hdr.values)}
        url_col_idx = None
        for key in ["Url", "URL", "url", "Link", "HREF", "href"]:
            if key in col_idx:
                url_col_idx = col_idx[key]
                break
        cat_cols = [col_idx.get(f"Cat{k}") for k in range(1, 6) if col_idx.get(f"Cat{k}") is not None]
        if not cat_cols:
            cat_cols = [col_idx.get(f"L{k}") for k in range(1, 6) if col_idx.get(f"L{k}") is not None]
        url_series = data.iloc[:, url_col_idx].astype(str).str.strip() if url_col_idx is not None else ""
        name_series = []
        for r in range(len(data)):
            parts = []
            for idx in cat_cols:
                v = str(data.iloc[r, idx]).strip()
                if v and v.lower() != "nan":
                    parts.append(v)
            name_series.append(" > ".join(parts))
        out = pd.DataFrame({"Name": name_series, "Url": url_series})
        return _clean_retailer(out)
    url_col = _pick_url_like_col(raw)
    path_col = max([(c, _pathiness(raw[c])) for c in raw.columns if c != url_col], key=lambda x: x[1])[0]
    out = raw.loc[:, [path_col, url_col]].copy()
    out.columns = ["Name", "Url"]
    return _clean_retailer(out)

def _detect_cat_cols(df: pd.DataFrame) -> list:
    import re as _re
    cat_cols = [c for c in df.columns if _re.fullmatch(r"(?i)cat[1-5]|l[1-5]|level[1-5]|name_l[1-5]", str(c))]
    cat_cols = sorted(cat_cols, key=lambda c: int(_re.search(r"([1-5])", str(c)).group(1)) if _re.search(r"([1-5])", str(c)) else 99)
    return cat_cols

def _pick_url_like_col(df: pd.DataFrame):
    scores = []
    for c in df.columns:
        s = df[c].astype(str)
        frac = s.str.contains(r"^(?:https?:)?//|^www\.", regex=True, na=False).mean()
        scores.append((frac, c))
    scores.sort(reverse=True)
    return scores[0][1]

def _pathiness(series: pd.Series) -> float:
    s = series.astype(str)
    return (s.str.contains(r">\s*|›|»|→|/", regex=True, na=False)).mean()

def _clean_retailer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Name"] = df["Name"].astype(str).str.strip()
    df["Url"] = df["Url"].astype(str).str.strip()
    df = df[(df["Url"].str.contains(r"https?://|^www\.")) & (df["Name"].str.len() > 0)]
    return df.reset_index(drop=True)
