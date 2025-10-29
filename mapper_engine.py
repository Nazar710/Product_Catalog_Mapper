"""
Product Catalog Mapper Engine

This module provides the core functionality for mapping IKEA product taxonomy
to retailer catalogs using TF-IDF similarity matching and human-in-the-loop feedback.

Author: Evgeny Nazarenko
"""

from __future__ import annotations
import re, unicodedata, asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import httpx

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

# Generic terms that should be filtered out as they don't provide meaningful mapping signals
GENERIC_TERMS = {
    "shop all", "collections", "new", "sale", "offers", "furniture",
    "living room", "bedroom", "dining room", "seating", "all sofas", "all furniture"
}

@dataclass
class MapperConfig:
    """
    Configuration class for the catalog mapper.
    
    Attributes:
        retailer_name (str): Name of the retailer (used for file naming and display)
        retailer_domain (str, optional): Domain for relative URL conversion
        one_to_many_ratio (float): Threshold ratio for keeping multiple matches (0.90 = keep matches >= 90% of best score)
        min_score (float): Minimum similarity score to consider a match (0.30 = 30% similarity)
        min_keep (float): Minimum score threshold for keeping the best match (0.25 = 25% similarity)
        verify_urls (bool): Whether to verify URLs are accessible via HTTP requests
        request_timeout (float): Timeout in seconds for HTTP requests
        scoring_method (str): Scoring method - 'tfidf', 'bm25', 'tfidf_advanced', 'embeddings'
        l1_weight (int): Weight multiplier for L1 (broad) categories
        l2_weight (int): Weight multiplier for L2 (mid-level) categories  
        l3_weight (int): Weight multiplier for L3 (specific) categories
        examples_weight (int): Weight multiplier for Examples column
    """
    retailer_name: str
    retailer_domain: Optional[str] = None
    one_to_many_ratio: float = 0.90
    min_score: float = 0.30
    min_keep: float = 0.25
    verify_urls: bool = False
    request_timeout: float = 6.0
    scoring_method: str = 'tfidf'
    l1_weight: int = 1
    l2_weight: int = 3
    l3_weight: int = 6
    examples_weight: int = 2

def norm_text(s: str) -> str:
    """
    Normalize text for consistent matching by:
    - Converting to lowercase
    - Normalizing Unicode characters
    - Replacing special characters with spaces
    - Removing extra whitespace
    
    Args:
        s (str): Input text to normalize
        
    Returns:
        str: Normalized text ready for comparison
        
    Example:
        >>> norm_text("Sofás & Chairs/Seating")
        "sofas and chairs seating"
    """
    if s is None: return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)  # Normalize Unicode (é -> e)
    s = s.lower().replace("&", " and ")   # Convert & to "and"
    s = re.sub(r"[\\/-]", " ", s)         # Replace slashes and backslashes with spaces
    s = re.sub(r"[^\w\s]", " ", s)        # Remove punctuation, keep words and spaces
    s = re.sub(r"\s+", " ", s)            # Collapse multiple spaces
    return s.strip()

def strip_numeric_prefix(s: str) -> str:
    """
    Remove numeric prefixes from category names.
    IKEA categories often have format "01 - Category Name"
    
    Args:
        s (str): Input string that may have numeric prefix
        
    Returns:
        str: String with numeric prefix removed
        
    Example:
        >>> strip_numeric_prefix("01 - Sofas")
        "Sofas"
    """
    if s is None: return ""
    parts = re.split(r"\s*[–-]\s*", str(s).strip(), maxsplit=1)
    return parts[1].strip() if len(parts) == 2 else str(s).strip()

def split_path(name: str) -> List[str]:
    """
    Split hierarchical category path into components.
    Retailer categories are often formatted as "Level1 > Level2 > Level3"
    
    Args:
        name (str): Hierarchical path string
        
    Returns:
        List[str]: List of path components
        
    Example:
        >>> split_path("Furniture > Living Room > Sofas")
        ["Furniture", "Living Room", "Sofas"]
    """
    return [p.strip() for p in str(name).split(">") if str(p).strip()]

def looks_generic(path: str) -> bool:
    """
    Check if a category path contains only generic terms that don't provide
    meaningful mapping signals.
    
    Args:
        path (str): Category path to check
        
    Returns:
        bool: True if path is too generic for meaningful mapping
        
    Example:
        >>> looks_generic("Shop All > New Products")
        True
    """
    parts = split_path(path)
    leaf = parts[-1].strip().lower() if parts else ""
    return (not parts) or (leaf in GENERIC_TERMS)

def detect_room(l1: str) -> str:
    """
    Detect room category from IKEA Level 1 category name.
    Used for room-based filtering of retailer matches.
    
    Args:
        l1 (str): IKEA Level 1 category name
        
    Returns:
        str: Detected room type or empty string if no match
        
    Example:
        >>> detect_room("Living room furniture")
        "living"
    """
    t = norm_text(l1)
    if "bath" in t: return "bathroom"
    if "bed" in t: return "bedroom"
    if "dining" in t or "kitchen" in t: return "dining"
    if "living" in t or "seating" in t: return "living"
    if "office" in t: return "office"
    if "rug" in t: return "rugs"
    if "light" in t: return "lighting"
    # Add media/entertainment detection
    if "media" in t or "tv" in t or "entertainment" in t or "storage" in t: return "living"
    return ""

def canonicalize(s: str, synonyms: Dict[str, List[str]]) -> str:
    """
    Replace synonyms in text with canonical forms using synonym dictionary.
    Processes longer synonyms first to avoid partial replacements.
    
    Args:
        s (str): Input text to canonicalize
        synonyms (Dict[str, List[str]]): Dictionary mapping canonical terms to synonym lists
        
    Returns:
        str: Text with synonyms replaced by canonical forms
        
    Example:
        >>> canonicalize("couch", {"sofa": ["couch", "settee"]})
        "sofa"
    """
    st = norm_text(s)
    rev = []
    for canon, syns in (synonyms or {}).items():
        for syn in [canon] + list(syns or []):
            rev.append((syn, canon))
    rev.sort(key=lambda x: -len(x[0]))  # Process longer synonyms first
    for syn, canon in rev:
        st = re.sub(rf"\b{re.escape(syn)}\b", canon, st)
    return st

def build_docs_ikea(row: pd.Series, synonyms: Dict[str, List[str]], 
                   l1_weight: int = 1, l2_weight: int = 3, l3_weight: int = 6, examples_weight: int = 2) -> str:
    """
    Build a weighted document string from IKEA taxonomy row for similarity matching.
    Uses configurable weights for different taxonomy levels and examples.
    
    Args:
        row (pd.Series): IKEA taxonomy row with L1, L2, L3 columns
        synonyms (Dict[str, List[str]]): Synonym dictionary for canonicalization
        l1_weight (int): Weight multiplier for L1 (broad) categories
        l2_weight (int): Weight multiplier for L2 (mid-level) categories
        l3_weight (int): Weight multiplier for L3 (specific) categories
        examples_weight (int): Weight multiplier for Examples column
        
    Returns:
        str: Weighted document string for similarity matching
        
    Example:
        For row with L1="Furniture", L2="Seating", L3="Sofas", weights (1,3,6,2):
        Returns: "sofas sofas sofas sofas sofas sofas seating seating seating furniture examples examples"
    """
    L1 = strip_numeric_prefix(row["L1"]) if "L1" in row else strip_numeric_prefix(row[0])
    L2 = strip_numeric_prefix(row["L2"]) if "L2" in row else strip_numeric_prefix(row[1])
    L3 = strip_numeric_prefix(row["L3"]) if "L3" in row else strip_numeric_prefix(row[2])
    ex = str(row.get("Examples", ""))
    
    l1 = canonicalize(L1, synonyms)
    l2 = canonicalize(L2, synonyms)
    l3 = canonicalize(L3, synonyms)
    
    # Build weighted document with configurable weights
    doc_parts = []
    doc_parts.extend([l3] * l3_weight)
    doc_parts.extend([l2] * l2_weight) 
    doc_parts.extend([l1] * l1_weight)
    
    if ex:
        ex_canonical = canonicalize(ex, synonyms)
        doc_parts.extend([ex_canonical] * examples_weight)
    
    doc = " ".join(doc_parts)
    return norm_text(doc)

async def verify_url_async(client: httpx.AsyncClient, url: str) -> bool:
    """
    Asynchronously verify if a URL is accessible.
    Tries HEAD request first (faster), then GET request if needed.
    
    Args:
        client (httpx.AsyncClient): HTTP client for making requests
        url (str): URL to verify
        
    Returns:
        bool: True if URL is accessible (status 200-399), False otherwise
    """
    try:
        r = await client.head(url, follow_redirects=True)
        if 200 <= r.status_code < 400: return True
        r = await client.get(url, follow_redirects=True)
        return 200 <= r.status_code < 400
    except Exception:
        return False

async def batch_verify(urls: List[str], timeout: float = 6.0) -> List[bool]:
    """
    Verify multiple URLs concurrently using async HTTP requests.
    
    Args:
        urls (List[str]): List of URLs to verify
        timeout (float): Request timeout in seconds
        
    Returns:
        List[bool]: List of verification results corresponding to input URLs
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [verify_url_async(client, u) for u in urls]
        return await asyncio.gather(*tasks)

class CatalogMapper:
    """
    Main class for mapping IKEA product taxonomy to retailer catalogs.
    
    Uses TF-IDF similarity matching with room-based filtering, synonym handling,
    and human feedback integration via overrides.
    
    Attributes:
        cfg (MapperConfig): Configuration parameters
        synonyms (Dict[str, List[str]]): Normalized synonym dictionary
        overrides (Dict): Manual mapping overrides and configuration
        room_gate (Dict): Room-based filtering rules
        domain (str): Retailer domain for URL processing
        generic_terms (Set[str]): Combined set of terms to filter out
    """
    
    def __init__(self, config: MapperConfig, synonyms: Dict[str, List[str]], overrides: Dict, generic_blacklist: List[str] = None):
        """
        Initialize the catalog mapper with configuration and data.
        
        Args:
            config (MapperConfig): Mapping configuration parameters
            synonyms (Dict[str, List[str]]): Synonym dictionary for text canonicalization
            overrides (Dict): Manual overrides and domain configuration
            generic_blacklist (List[str], optional): Additional generic terms to filter
        """
        self.cfg = config
        self.synonyms = {k.lower(): [s.lower() for s in v] for k, v in (synonyms or {}).items()}
        self.overrides = overrides or {}
        self.room_gate = (self.overrides.get("room_gate") or {})
        self.domain = self.overrides.get("domain") or self.cfg.retailer_domain
        self.generic_terms = set([norm_text(x) for x in (generic_blacklist or [])]) | GENERIC_TERMS
        
        # Initialize embedding model if needed
        self.embedding_model = None
        if self.cfg.scoring_method == 'embeddings':
            if not EMBEDDINGS_AVAILABLE:
                raise ImportError("sentence-transformers is required for embeddings scoring. Install with: pip install sentence-transformers")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def _prepare_retailer(self, df_retail: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare retailer catalog data for similarity matching.
        Extracts path components, normalizes leaf categories, builds document strings.
        """
        df = df_retail.copy()
        
        # Check if retailer data has multiple columns (path segments) or single Name column
        if 'Name' in df.columns:
            # Single column with "Level1 > Level2 > Level3" format
            df["parts"] = df["Name"].apply(split_path)
            df["full_path"] = df["Name"].copy()
        else:
            # Multiple columns - handle ALL columns, not just first 3
            # Find URL column and exclude it from path columns
            url_cols = [col for col in df.columns if col.lower() in ['url', 'urls', 'link', 'links']]
            path_cols = [col for col in df.columns if col not in url_cols]
            
            # Create full hierarchical path by combining non-empty segments
            def build_full_path(row):
                segments = []
                for col in path_cols:
                    val = str(row[col]).strip() if pd.notna(row[col]) else ""
                    if val and val.lower() != 'nan':
                        segments.append(val)
                return " > ".join(segments)
            
            df["full_path"] = df.apply(build_full_path, axis=1)
            df["parts"] = df["full_path"].apply(split_path)
        
        df["leaf"] = df["parts"].apply(lambda xs: xs[-1] if xs else "")
        df["leaf_norm"] = df["leaf"].apply(lambda s: canonicalize(s, self.synonyms))
        
        # Apply weighted document building similar to IKEA side
        def build_weighted_retail_doc(row):
            parts = row["parts"]
            if not parts:
                return ""
            
            # Apply weights: more specific levels get higher weights
            weighted_terms = []
            for i, part in enumerate(reversed(parts)):  # Start from most specific
                weight = max(1, len(parts) - i)  # Most specific gets highest weight
                canonical_part = canonicalize(part, self.synonyms)
                weighted_terms.extend([canonical_part] * weight)
            
            return norm_text(" ".join(weighted_terms))
        
        df["doc_norm"] = df.apply(build_weighted_retail_doc, axis=1)
        
        return df

    def _vectorize(self, ret_docs: pd.Series):
        """
        Create vectorizer and transform retailer documents using chosen scoring method.
        
        Args:
            ret_docs (pd.Series): Normalized retailer document strings
            
        Returns:
            Tuple[object, ndarray]: Fitted vectorizer/model and document vectors
        """
        if self.cfg.scoring_method == 'embeddings':
            # Use semantic embeddings
            embeddings = self.embedding_model.encode(ret_docs.tolist())
            return self.embedding_model, embeddings
        
        elif self.cfg.scoring_method == 'bm25':
            # Use BM25 scoring
            if not BM25_AVAILABLE:
                raise ImportError("rank-bm25 is required for BM25 scoring. Install with: pip install rank-bm25")
            tokenized_docs = [doc.split() for doc in ret_docs]
            bm25 = BM25Okapi(tokenized_docs)
            return bm25, tokenized_docs
        
        elif self.cfg.scoring_method == 'tfidf_advanced':
            # Use advanced TF-IDF with better parameters
            vec = TfidfVectorizer(
                ngram_range=(1, 3),      # Include 3-grams for better phrase matching
                max_features=10000,      # Limit vocabulary size  
                sublinear_tf=True,       # Logarithmic term frequency scaling
                norm='l2',               # L2 normalization
                smooth_idf=True,         # Add smoothing to IDF
                min_df=1,               # Keep rare terms
                max_df=0.95             # Remove very common terms
            )
            X = vec.fit_transform(ret_docs)
            return vec, X
        
        else:
            # Use standard TF-IDF (default)
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            X = vec.fit_transform(ret_docs)
            return vec, X

    def _gate_mask(self, ret_docs: pd.Series, room: str) -> np.ndarray:
        """
        Create boolean mask for room-based filtering of retailer candidates.
        Only considers retailers that contain allowed terms for the detected room.
        
        Args:
            ret_docs (pd.Series): Retailer document strings
            room (str): Detected room category (e.g., "living", "bedroom")
            
        Returns:
            np.ndarray: Boolean mask indicating allowed retailer candidates
        """
        if not room: return np.ones(len(ret_docs), dtype=bool)
        allowed = set((self.room_gate or {}).get(room, []))
        if not allowed: return np.ones(len(ret_docs), dtype=bool)
        patt = r"|".join([re.escape(k) for k in sorted(allowed, key=len, reverse=True)])
        m = ret_docs.str.contains(patt, regex=True, na=False)
        if not m.any(): return np.ones(len(ret_docs), dtype=bool)
        return m.values

    def map(self, ikea_df: pd.DataFrame, retail_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main mapping method that matches IKEA taxonomy to retailer catalog.
        
        Process:
        1. Normalizes IKEA taxonomy data and detects rooms
        2. Prepares retailer data and creates TF-IDF vectors  
        3. For each IKEA category:
           - Checks for manual overrides
           - Applies room-based filtering
           - Computes similarity scores
           - Selects best matches based on thresholds
        4. Optionally verifies URLs
        5. Returns merged results with similarity scores and methods
        
        Args:
            ikea_df (pd.DataFrame): IKEA taxonomy with L1, L2, L3 columns
            retail_df (pd.DataFrame): Retailer catalog with Name, Url columns
            
        Returns:
            pd.DataFrame: Mapping results with columns:
                - Original IKEA taxonomy columns
                - {retailer_name}_catalog_path: Matched retailer category
                - {retailer_name}_url: Matched retailer URL  
                - {retailer_name}_score: Similarity score (0-1)
                - {retailer_name}_method: Mapping method used
                - url_live: URL verification result (if enabled)
        """
        col_map = {str(c).lower().strip(): c for c in ikea_df.columns}
        L1col = next((col_map[k] for k in col_map if k in {"hfbc", "home furnishing business caption"}), list(ikea_df.columns)[0])
        L2col = next((col_map[k] for k in col_map if k in {"prac", "product range area caption"}), list(ikea_df.columns)[1])
        L3col = next((col_map[k] for k in col_map if k in {"pac", "product area caption"}), list(ikea_df.columns)[2])
        EXcol = next((col_map[k] for k in col_map if "example" in k), None)
        HLcol = next((col_map[k] for k in col_map if k.replace(" ", "_") == "human_label"), None)

        work = ikea_df.copy()
        work["L1"] = work[L1col].apply(strip_numeric_prefix)
        work["L2"] = work[L2col].apply(strip_numeric_prefix)
        work["L3"] = work[L3col].apply(strip_numeric_prefix)
        work["Examples"] = work[EXcol] if EXcol in work.columns else ""
        work["human_label"] = pd.to_numeric(work[HLcol], errors="coerce").fillna(0).astype(int) if HLcol in work.columns else 0
        work["room"] = work["L1"].apply(detect_room)

        retail = self._prepare_retailer(retail_df)
        vec, Xret = self._vectorize(retail["doc_norm"])

        # Only apply leaf overrides if they match the current retailer or are generic (no retailer specified)
        override_retailer = self.overrides.get("retailer", "").lower()
        current_retailer = self.cfg.retailer_name.lower()
        
        if not override_retailer or override_retailer == current_retailer:
            leaf_over = {k.lower(): v for k, v in (self.overrides.get("leaf_overrides") or {}).items()}
        else:
            leaf_over = {}  # Don't apply retailer-specific overrides for different retailers

        prev_path_col = f"{self.cfg.retailer_name}_catalog_path"
        prev_url_col  = f"{self.cfg.retailer_name}_url"
        prev_score_col= f"{self.cfg.retailer_name}_score"
        prev_method_col=f"{self.cfg.retailer_name}_method"
        has_prev_cols = all(c in ikea_df.columns for c in [prev_path_col, prev_url_col])

        out_rows = []
        for i, row in work.iterrows():
            if row.get("human_label", 0) == 1 and has_prev_cols:
                prev_path = ikea_df.loc[i, prev_path_col]
                prev_url  = ikea_df.loc[i, prev_url_col]
                prev_score= float(ikea_df.loc[i, prev_score_col]) if prev_score_col in ikea_df.columns else 1.0
                prev_method = ikea_df.loc[i, prev_method_col] if prev_method_col in ikea_df.columns else "human"
                out_rows.append(self._emit(i, prev_path, prev_url, prev_score, prev_method, url_source=("file" if isinstance(prev_url, str) and prev_url else "blank")))
                continue

            mask = self._gate_mask(retail["doc_norm"], row["room"])
            cand_idx = np.where(mask)[0]
            if cand_idx.size == 0:
                cand_idx = np.arange(len(retail))

            l3n = canonicalize(row["L3"], self.synonyms)
            if l3n in leaf_over:
                ov = leaf_over[l3n]
                cat_path = ov.get("catalog_path", "")
                url = ov.get("url", "")
                out_rows.append(self._emit(i, cat_path, url, 1.0, "override", url_source=("file" if url else "blank")))
                continue

            # Build IKEA document with configurable weights
            ikea_doc = build_docs_ikea(
                row, self.synonyms, 
                self.cfg.l1_weight, self.cfg.l2_weight, 
                self.cfg.l3_weight, self.cfg.examples_weight
            )
            
            if self.cfg.scoring_method == 'embeddings':
                # Use semantic embeddings
                ikea_embedding = vec.encode([ikea_doc])
                sims = cosine_similarity(ikea_embedding, Xret)[0]
            
            elif self.cfg.scoring_method == 'bm25':
                # Use BM25 scoring
                ikea_tokens = ikea_doc.split()
                sims = np.array(vec.get_scores(ikea_tokens))
                
            else:
                # Use TF-IDF (standard or advanced)
                qv = vec.transform([ikea_doc])
                sims = cosine_similarity(qv, Xret)[0]
                
            sims_masked = np.full_like(sims, -1.0)
            sims_masked[cand_idx] = sims[cand_idx]

            if sims_masked.size == 0 or (sims_masked.max() < self.cfg.min_keep):
                out_rows.append(self._emit(i, "", "", 0.0, "low-confidence", url_source="blank"))
                continue

            best = float(sims_masked.max())

            keep = []
            for j, sc in enumerate(sims_masked):
                if sc < 0: continue
                if sc >= max(self.cfg.min_score, best * self.cfg.one_to_many_ratio):
                    # Get the full path for both display and blacklist checking
                    full_path = retail.loc[j, "full_path"] if "full_path" in retail.columns else retail.loc[j, "Name"]
                    ikea_context = f"{row['L1']} {row['L2']} {row['L3']}"
                    
                    if not looks_generic(full_path) and not self._is_blacklisted(full_path, ikea_context):
                        keep.append((j, float(sc)))

            if not keep:
                jbest = int(np.nanargmax(sims_masked))
                # Add additional validation for fallback matches to prevent garbage mappings
                if best < 0.15:  # Very low similarity threshold for garbage detection
                    out_rows.append(self._emit(i, "", "", 0.0, "low-confidence", url_source="blank"))
                    continue
                keep = [(jbest, best)]

            # Cap one-to-many results to maximum 3 matches
            keep = sorted(keep, key=lambda x: x[1], reverse=True)[:3]

            for j, sc in keep:
                # Use full_path which now contains the complete hierarchical path
                full_path_val = retail.loc[j, "full_path"]
                out_rows.append(self._emit(i, full_path_val, retail.loc[j, "Url"], sc, "cosine", url_source="file"))

        out = pd.DataFrame(out_rows)

        if self.cfg.verify_urls and not out["_url"].isna().all():
            urls = out["_url"].fillna("").tolist()
            try:
                live = asyncio.run(batch_verify(urls, timeout=self.cfg.request_timeout))
            except RuntimeError:
                live = [None] * len(urls)
            out["url_live"] = live
        else:
            out["url_live"] = np.nan

        base_cols = ["L1", "L2", "L3"]
        if "Examples" in work.columns: base_cols.append("Examples")
        if "human_label" in work.columns: base_cols.append("human_label")

        base = work[base_cols].copy()
        base["_row"] = np.arange(len(base))
        
        # Ensure ALL IKEA rows are included - use LEFT join instead of RIGHT join
        out["_row"] = out["_row"].astype(int)
        merged = base.merge(out, left_on="_row", right_on="_row", how="left").drop(columns=["_row"])
        
        # Fill missing values for rows that didn't get any matches
        merged[f"{self.cfg.retailer_name}_catalog_path"] = merged.pop("_path").fillna("")
        merged[f"{self.cfg.retailer_name}_url"] = merged.pop("_url").fillna("")
        merged[f"{self.cfg.retailer_name}_score"] = merged.pop("_score").fillna(0.0)
        merged[f"{self.cfg.retailer_name}_method"] = merged.pop("_method").fillna("no-match")

        return merged

    def _is_blacklisted(self, path: str, ikea_context: str = "") -> bool:
        """
        Check if a retailer category path should be filtered out due to generic terms.
        
        Args:
            path (str): Retailer category path to check
            ikea_context (str): IKEA category context for semantic filtering
            
        Returns:
            bool: True if path contains generic terms and should be filtered
        """
        p = norm_text(path)
        parts = split_path(path)
        leaf = parts[-1].lower() if parts else ""
        
        # Add semantic filtering to prevent obviously wrong mappings
        # Only apply semantic filtering if IKEA context suggests media/TV furniture
        if ikea_context and any(term in norm_text(ikea_context) for term in ["media", "tv", "entertainment"]):
            if any(term in p for term in ["bedroom", "bed", "mattress", "bedding"]):
                return True
            
        return (leaf in self.generic_terms) or (p in self.generic_terms)

    def _emit(self, idx: int, path: str, url: str, score: float, method: str, url_source: str = "file") -> Dict:
        """
        Create a result row dictionary for a mapping match.
        
        Args:
            idx (int): IKEA taxonomy row index
            path (str): Matched retailer category path  
            url (str): Matched retailer URL
            score (float): Similarity score (0-1)
            method (str): Mapping method used ("cosine", "override", "low-confidence")
            url_source (str): Source of URL ("file", "blank")
            
        Returns:
            Dict: Result row with mapping details and metadata
        """
        tokens_d = set(norm_text(path).split())
        inter = ",".join(sorted(tokens_d))[:120]  # Truncate for display
        return {
            "_row": idx,
            "_path": path,
            "_url": url,
            "_score": float(score),
            "_method": method,
            "matched_tokens": inter,
        }

