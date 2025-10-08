
"""
Input/Output utilities for reading IKEA and retailer catalog Excel files.

This module handles various Excel file formats and automatically detects
the correct header rows and column structures.

Author: Evgeny Nazarenko
"""

from __future__ import annotations
import re
import pandas as pd

# Known column identifiers for IKEA taxonomy levels
KNOWN_IKEA_L1 = {"hfbc", "home furnishing business caption", "level1", "l1"}
KNOWN_IKEA_L2 = {"prac", "product range area caption", "level2", "l2"} 
KNOWN_IKEA_L3 = {"pac", "product area caption", "level3", "l3"}

def read_ikea_xlsx(path: str) -> pd.DataFrame:
    """
    Read and parse an IKEA taxonomy Excel file with automatic header detection.
    
    This function handles various IKEA Excel file formats by:
    1. First attempting to read with default header (row 0)
    2. If that doesn't look like IKEA format, scanning for header row containing IKEA keywords
    3. Parsing with detected header row to ensure proper column alignment
    
    The function recognizes IKEA format by looking for taxonomy level identifiers:
    - L1/HFBC: Home Furnishing Business Caption (top level categories)
    - L2/PRAC: Product Range Area Caption (product ranges)  
    - L3/PAC: Product Area Caption (specific product areas)
    
    Args:
        path (str): File path to the IKEA Excel file (.xlsx format)
        
    Returns:
        pd.DataFrame: Parsed IKEA taxonomy with columns for different hierarchy levels.
                     Typical columns include HFBC, PRAC, PAC representing the 3-level
                     IKEA product categorization hierarchy.
                     
    Example:
        >>> df = read_ikea_xlsx('ikea_taxonomy.xlsx')
        >>> print(df.columns)
        ['HFBC', 'PRAC', 'PAC', ...]
        
    Note:
        The function is robust to different Excel formats and will scan up to 20 rows
        to find the correct header containing IKEA taxonomy identifiers.
    """
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
    """
    Check if a DataFrame contains IKEA taxonomy column structure.
    
    Validates that the DataFrame has columns corresponding to all three levels
    of the IKEA product hierarchy by checking for known column identifiers.
    
    Args:
        df (pd.DataFrame): DataFrame to validate for IKEA structure
        
    Returns:
        bool: True if DataFrame contains all three IKEA taxonomy levels (L1, L2, L3),
              False otherwise
              
    Example:
        >>> df = pd.DataFrame(columns=['HFBC', 'PRAC', 'PAC', 'ProductName'])
        >>> _looks_like_ikea(df)
        True
        >>> df2 = pd.DataFrame(columns=['Category', 'Product'])  
        >>> _looks_like_ikea(df2)
        False
    """
    cols = {str(c).strip().lower() for c in df.columns}
    return (cols & KNOWN_IKEA_L1) and (cols & KNOWN_IKEA_L2) and (cols & KNOWN_IKEA_L3)

def read_retailer_xlsx(path: str) -> pd.DataFrame:
    """
    Read and parse a retailer catalog Excel file with intelligent format detection.
    
    This function handles various retailer catalog formats by attempting multiple parsing strategies:
    1. Standard format: Direct Name/URL columns
    2. Category hierarchy: Cat1, Cat2, Cat3... columns combined into hierarchical names
    3. Custom header detection: Scanning for specific patterns like "Cat1", "Cat2", "Url"
    4. Fallback: Path-like column detection for hierarchical product names
    
    The function automatically:
    - Detects and normalizes column names (name, url, link, href variations)
    - Combines category columns into hierarchical paths (Cat1 > Cat2 > Cat3)
    - Identifies URL-like columns for product links
    - Cleans and validates the resulting data
    
    Args:
        path (str): File path to the retailer Excel file (.xlsx format)
        
    Returns:
        pd.DataFrame: Standardized DataFrame with columns:
                     - 'Name': Product name or hierarchical category path
                     - 'Url': Product URL or link
                     All rows are cleaned to ensure valid URLs and non-empty names.
                     
    Example:
        >>> df = read_retailer_xlsx('retailer_catalog.xlsx')
        >>> print(df.columns)
        ['Name', 'Url']
        >>> print(df.head())
           Name                              Url
        0  Furniture > Chairs > Office      https://retailer.com/office-chairs
        1  Home > Kitchen > Appliances      https://retailer.com/kitchen-appliances
        
    Note:
        The function is designed to be robust across different retailer catalog formats
        and will attempt multiple parsing strategies before falling back to basic column detection.
    """
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
    """
    Detect and sort category hierarchy columns in a retailer catalog DataFrame.
    
    Identifies columns that represent hierarchical category levels using pattern matching
    for common naming conventions like Cat1, Cat2, L1, L2, Level1, etc.
    
    Args:
        df (pd.DataFrame): DataFrame to scan for category columns
        
    Returns:
        list: Sorted list of column names representing category hierarchy levels,
              ordered from highest level (Cat1/L1) to lowest level (Cat5/L5)
              
    Example:
        >>> df = pd.DataFrame(columns=['Cat3', 'Cat1', 'Cat2', 'ProductUrl'])
        >>> _detect_cat_cols(df)
        ['Cat1', 'Cat2', 'Cat3']
        
        >>> df2 = pd.DataFrame(columns=['Level2', 'Level1', 'name_l3'])
        >>> _detect_cat_cols(df2)  
        ['Level1', 'Level2', 'name_l3']
        
    Note:
        Supports various naming patterns: Cat1-5, L1-5, Level1-5, name_l1-5
        Case-insensitive matching for maximum compatibility
    """
    import re as _re
    cat_cols = [c for c in df.columns if _re.fullmatch(r"(?i)cat[1-5]|l[1-5]|level[1-5]|name_l[1-5]", str(c))]
    cat_cols = sorted(cat_cols, key=lambda c: int(_re.search(r"([1-5])", str(c)).group(1)) if _re.search(r"([1-5])", str(c)) else 99)
    return cat_cols

def _pick_url_like_col(df: pd.DataFrame):
    """
    Identify the column most likely to contain URLs by analyzing content patterns.
    
    Scores each column based on the fraction of entries that match URL patterns
    (http://, https://, www., or protocol-relative //) and returns the highest scoring column.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze for URL-containing columns
        
    Returns:
        str: Column name with the highest concentration of URL-like content
        
    Example:
        >>> df = pd.DataFrame({
        ...     'Name': ['Product A', 'Product B'],
        ...     'Link': ['https://example.com/a', 'https://example.com/b'],  
        ...     'Price': ['$10', '$20']
        ... })
        >>> _pick_url_like_col(df)
        'Link'
        
    Note:
        Uses regex pattern matching to identify URLs in various formats:
        - Full URLs: http://example.com, https://example.com
        - Protocol-relative: //example.com
        - www URLs: www.example.com
    """
    scores = []
    for c in df.columns:
        s = df[c].astype(str)
        frac = s.str.contains(r"^(?:https?:)?//|^www\.", regex=True, na=False).mean()
        scores.append((frac, c))
    scores.sort(reverse=True)
    return scores[0][1]

def _pathiness(series: pd.Series) -> float:
    """
    Calculate the "pathiness" score of a pandas Series based on hierarchical separators.
    
    Measures how often entries in the series contain path-like separators that indicate
    hierarchical category structures (>, ›, », →, /).
    
    Args:
        series (pd.Series): Series to analyze for path-like content
        
    Returns:
        float: Fraction of entries (0.0 to 1.0) containing hierarchical separators
        
    Example:
        >>> import pandas as pd
        >>> s1 = pd.Series(['Home > Kitchen', 'Furniture > Chairs', 'Electronics'])
        >>> _pathiness(s1)
        0.6666666666666666  # 2 out of 3 entries have separators
        
        >>> s2 = pd.Series(['Product A', 'Product B', 'Product C'])
        >>> _pathiness(s2)
        0.0  # No hierarchical separators found
        
    Note:
        Recognizes multiple separator styles commonly used in product catalogs:
        - Standard arrow: >
        - Unicode arrows: ›, », →  
        - Forward slash: /
    """
    s = series.astype(str)
    return (s.str.contains(r">\s*|›|»|→|/", regex=True, na=False)).mean()

def _clean_retailer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate retailer catalog data by filtering valid entries.
    
    Performs data cleaning operations to ensure the retailer catalog contains
    only valid entries with proper URLs and non-empty product names.
    
    Args:
        df (pd.DataFrame): Retailer DataFrame with 'Name' and 'Url' columns
        
    Returns:
        pd.DataFrame: Cleaned DataFrame containing only rows with:
                     - Valid URLs (http://, https://, or www. format)
                     - Non-empty product names
                     - Trimmed whitespace from both columns
                     - Reset index for clean numbering
                     
    Example:
        >>> df = pd.DataFrame({
        ...     'Name': ['  Product A  ', '', 'Product C'],
        ...     'Url': ['https://example.com/a', 'invalid-url', 'www.example.com/c']
        ... })
        >>> cleaned = _clean_retailer(df)
        >>> print(cleaned)
             Name                    Url
        0  Product A  https://example.com/a
        1  Product C    www.example.com/c
        
    Note:
        Filters out entries with invalid URLs or empty names to ensure
        data quality for downstream mapping operations.
    """
    df = df.copy()
    df["Name"] = df["Name"].astype(str).str.strip()
    df["Url"] = df["Url"].astype(str).str.strip()
    df = df[(df["Url"].str.contains(r"https?://|^www\.")) & (df["Name"].str.len() > 0)]
    return df.reset_index(drop=True)
