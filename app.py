
"""
Streamlit Web Application for Human-in-the-Loop Catalog Mapping.

This application provides an interactive web interface for mapping IKEA taxonomy
to retailer product catalogs using TF-IDF similarity matching with human oversight.

Key Features:
- Upload IKEA taxonomy and retailer catalog Excel files
- Configure mapping parameters (similarity thresholds, URL verification)
- Review and edit mapping results with real-time feedback
- Export edited results and create override rules
- Support for synonyms and custom mapping overrides

The app uses session state to preserve user edits and supports multiple
editing modes (checkbox vs direct input) for human labeling.

Author: Evgeny Nazarenko
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd

from io_utils import read_ikea_xlsx, read_retailer_xlsx
from mapper_engine import CatalogMapper, MapperConfig

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DEFAULT_SYNS = DATA_DIR / "synonyms.json"
DEFAULT_OVR = DATA_DIR / "overrides_example.json"
DEFAULT_BL  = DATA_DIR / "generic_blacklist.txt"

def to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    """
    Convert a pandas DataFrame to Excel file bytes for download.
    
    Creates an in-memory Excel file from the DataFrame using xlsxwriter engine
    and returns the binary content for Streamlit download functionality.
    
    Args:
        df (pd.DataFrame): DataFrame to convert to Excel format
        
    Returns:
        bytes: Binary content of the Excel file ready for download
        
    Example:
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> excel_bytes = to_xlsx_bytes(df)
        >>> st.download_button("Download", excel_bytes, "file.xlsx")
    """
    import io
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="mapping")
        return output.getvalue()
    except ImportError:
        # Fallback to openpyxl if xlsxwriter is not available
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="mapping")
        return output.getvalue()

st.set_page_config(page_title="Catalog Mapper", layout="wide")
st.title("🧭 Catalog Mapper — Human‑in‑the‑Loop")

with st.sidebar:
    st.header("Settings")
    retailer_name = st.text_input("Retailer name", value="Retailer")
    retailer_domain = st.text_input("Retailer domain (for relative URLs)", value="")

    # Scoring method selection
    try:
        from sentence_transformers import SentenceTransformer
        embeddings_available = True
    except ImportError:
        embeddings_available = False
    
    try:
        from rank_bm25 import BM25Okapi
        bm25_available = True
    except ImportError:
        bm25_available = False
    
    # Build scoring options based on availability
    scoring_options = ["tfidf", "tfidf_advanced"]
    scoring_labels = {
        "tfidf": "🔤 TF-IDF Standard (Fast, reliable)",
        "tfidf_advanced": "⚡ TF-IDF Advanced (Better phrase matching)",
    }
    
    if bm25_available:
        scoring_options.append("bm25")
        scoring_labels["bm25"] = "🎯 BM25 (Best balance: fast + accurate)"
    
    if embeddings_available:
        scoring_options.append("embeddings")
        scoring_labels["embeddings"] = "🧠 Semantic Embeddings (Slowest, understands meaning)"
    
    scoring_method = st.selectbox(
        "🎛️ Scoring Method",
        options=scoring_options,
        index=2 if bm25_available else 1,  # Default to BM25 if available, else TF-IDF Advanced
        format_func=lambda x: scoring_labels[x],
        help="Choose scoring method: TF-IDF (fast keyword), BM25 (balanced), Embeddings (semantic understanding)"
    )
    
    # Add recommendations based on selected method
    if scoring_method == "tfidf":
        st.info("💡 **TF-IDF Standard**: Perfect for simple catalogs with straightforward category names. Use default weights (1,3,6,2).")
    elif scoring_method == "tfidf_advanced":
        st.info("💡 **TF-IDF Advanced**: Great for catalogs with multi-word phrases like 'corner sofa beds'. Good with default weights.")
    elif scoring_method == "bm25":
        st.success("🎯 **BM25 Recommended**: Best balance of speed and accuracy. For complex catalogs, try increasing Examples weight to 3-4.")
    elif scoring_method == "embeddings":
        st.info("🧠 **Semantic Embeddings**: Best for synonym-heavy catalogs where 'couch' should match 'sofa'. Slower but most intelligent.")
    
    # Show installation tips for missing methods
    missing_methods = []
    if not bm25_available:
        missing_methods.append("`pip install rank-bm25` for BM25 scoring")
    if not embeddings_available:
        missing_methods.append("`pip install sentence-transformers` for semantic embeddings")
    
    if missing_methods:
        st.info("💡 Install: " + " | ".join(missing_methods))

    # Column weighting controls - always visible
    st.subheader("⚖️ IKEA Column Weights")
    st.caption("Control how much weight each IKEA column gets in similarity matching")
    
    col1, col2 = st.columns(2)
    with col1:
        l1_weight = st.number_input("🏢 L1 Weight (Broad categories)", min_value=0, max_value=10, value=1, 
                                  help="Weight for L1 categories like 'Furniture'")
        l2_weight = st.number_input("🏷️ L2 Weight (Mid-level)", min_value=0, max_value=10, value=3,
                                  help="Weight for L2 categories like 'Seating'")
    
    with col2:
        l3_weight = st.number_input("🎯 L3 Weight (Specific)", min_value=0, max_value=10, value=6,
                                  help="Weight for L3 categories like 'Sofas'")
        examples_weight = st.number_input("📝 Examples Weight", min_value=0, max_value=10, value=2,
                                        help="Weight for Examples column terms")
    
    st.caption(f"🎯 Current weighting: L1×{l1_weight}, L2×{l2_weight}, L3×{l3_weight}, Examples×{examples_weight}")
    
    # Weight recommendations
    with st.expander("💡 Weight Tuning Tips", expanded=False):
        st.markdown("""
        **🎯 When to adjust weights:**
        - **Increase L3 weight (7-10)**: When specific categories are most important
        - **Increase Examples weight (3-5)**: When examples contain key matching terms
        - **Decrease L1 weight (0-1)**: When broad categories cause noise
        - **Balance L2/L3 (4-6 each)**: When mid-level categories are very descriptive
        
        **📊 Quick presets:**
        - **Precise matching**: L1=0, L2=2, L3=8, Examples=3
        - **Balanced approach**: L1=1, L2=3, L3=6, Examples=2 (default)
        - **Example-heavy**: L1=1, L2=2, L3=5, Examples=5
        """)
    
    st.divider()

    ratio = st.slider("One‑to‑many ratio (≥ best ×)", 0.5, 1.0, 0.90, 0.05)
    min_score = st.slider("Min candidate score", 0.0, 1.0, 0.30, 0.05)
    min_keep = st.slider("Min keep score (best)", 0.0, 1.0, 0.25, 0.05)
    verify = st.checkbox("Verify URLs (HTTP)", value=False)

    st.divider()
    st.caption("Synonyms & Overrides (optional uploads)")
    syn_file = st.file_uploader("Synonyms JSON", type=["json"], key="syns")
    ovr_file = st.file_uploader("Overrides JSON", type=["json"], key="ovr")

    if syn_file is None and DEFAULT_SYNS.exists():
        synonyms = json.loads(DEFAULT_SYNS.read_text())
    elif syn_file is not None:
        synonyms = json.load(syn_file)
    else:
        synonyms = {}

    if ovr_file is None and DEFAULT_OVR.exists():
        overrides = json.loads(DEFAULT_OVR.read_text())
    elif ovr_file is not None:
        overrides = json.load(ovr_file)
    else:
        overrides = {}

    if DEFAULT_BL.exists():
        generic_blacklist = [x.strip() for x in DEFAULT_BL.read_text().splitlines() if x.strip()]
    else:
        generic_blacklist = []
        
    # Experiment mode indicator: Show when data is ready for experimentation
    # This helps users understand they can now iterate on settings without re-uploading
    if st.session_state.get('mapping_results') is not None:
        st.success("🧪 **Experiment Mode**: Change settings above and click 'Re-score' to test!")
        st.caption(f"📊 Ready to experiment with {len(st.session_state.mapping_results)} existing mappings")
    
    # Best practices section
    with st.expander("🎯 Best Practices & Troubleshooting", expanded=False):
        st.markdown("""
        **🚀 For best results:**
        - Start with BM25 method and default weights
        - If getting too generic matches, increase L3 weight
        - If missing obvious matches, increase Examples weight
        - Use Semantic Embeddings only for catalogs with many synonyms
        
        **🔧 Common issues:**
        - **Too many "Sofas" results**: Decrease L3 weight, increase L2 weight
        - **Missing specific items**: Increase Examples weight to 4-5
        - **Generic category matches**: Try Advanced TF-IDF with higher L3 weight
        - **Weird semantic matches**: Switch to BM25 or Standard TF-IDF
        
        **📈 Performance tips:**
        - Clean your catalog data (remove extra spaces, fix typos)
        - BM25 works best with varied vocabulary
        - Semantic embeddings need good internet connection
        - Test with small batches first
        """)
    
    st.markdown("""
    **📋 Quick start:**
    1. Select your retailer catalog type
    2. Choose scoring method (BM25 recommended for most cases)
    3. Adjust column weights based on your catalog structure
    4. Upload your catalog and let the mapper process it
    5. Edit results in the table and download when satisfied
    """)

c1, c2 = st.columns(2)
with c1:
    ikea_up = st.file_uploader("IKEA taxonomy XLSX", type=["xlsx"], key="ikea")
with c2:
    ret_up = st.file_uploader("Retailer catalog XLSX", type=["xlsx"], key="ret")

# Initialize session state
if 'mapping_results' not in st.session_state:
    st.session_state.mapping_results = None
if 'mapping_config' not in st.session_state:
    st.session_state.mapping_config = None
if 'edited_results' not in st.session_state:
    st.session_state.edited_results = None
if 'original_ikea_df' not in st.session_state:
    st.session_state.original_ikea_df = None
if 'original_retail_df' not in st.session_state:
    st.session_state.original_retail_df = None

col1, col2 = st.columns([2, 1])
with col1:
    run = st.button("Run Mapping", type="primary")
with col2:
    # Real-time experimentation: Only show re-score button if we have existing data
    # This allows users to experiment with different scoring methods and weights
    # without re-uploading files, enabling rapid iteration and comparison
    if st.session_state.get('mapping_results') is not None:
        rescore = st.button("🔄 Re-score with Current Settings", help="Apply new weights/method to existing data")
    else:
        rescore = False

if run:
    if not ikea_up or not ret_up:
        st.warning("Please upload both IKEA and Retailer files.")
        st.stop()

    with st.spinner("Parsing files…"):
        ikea_df = read_ikea_xlsx(ikea_up)
        retail_df = read_retailer_xlsx(ret_up)
        
        # Store original dataframes for re-scoring experiments
        # This enables real-time experimentation without re-uploading files
        st.session_state.original_ikea_df = ikea_df.copy()
        st.session_state.original_retail_df = retail_df.copy()

    st.success(f"Parsed IKEA rows: {len(ikea_df)} | Retailer rows: {len(retail_df)}")

    cfg = MapperConfig(
        retailer_name=retailer_name,
        retailer_domain=retailer_domain or overrides.get("domain"),
        one_to_many_ratio=ratio,
        min_score=min_score,
        min_keep=min_keep,
        verify_urls=verify,
        scoring_method=scoring_method,
        l1_weight=l1_weight,
        l2_weight=l2_weight,
        l3_weight=l3_weight,
        examples_weight=examples_weight,
    )

    mapper = CatalogMapper(cfg, synonyms=synonyms, overrides=overrides, generic_blacklist=generic_blacklist)

    with st.spinner("Scoring and selecting candidates…"):
        result = mapper.map(ikea_df, retail_df)

    # Store results and configuration in session state for experimentation
    # This enables users to change settings and re-score without losing data
    st.session_state.mapping_results = result.copy()
    st.session_state.mapping_config = {
        'retailer_name': retailer_name,
        'synonyms': synonyms,
        'overrides': overrides,
        'generic_blacklist': generic_blacklist,
        # Store current scoring settings for display and re-scoring
        'scoring_method': scoring_method,
        'l1_weight': l1_weight,
        'l2_weight': l2_weight,
        'l3_weight': l3_weight,
        'examples_weight': examples_weight
    }
    # Reset edited results since we have new mapping with potentially different settings
    st.session_state.edited_results = None

    st.success(f"✅ Generated {len(result)} mapped rows (one‑to‑many duplicates included).")
    st.info(f"🎛️ **Applied Settings**: {scoring_method.upper()} scoring • Weights: L1={l1_weight}, L2={l2_weight}, L3={l3_weight}, Examples={examples_weight}")

# Real-time experimentation: Handle re-scoring with new parameters
# This allows users to experiment with different scoring methods and weights
# on the same dataset without re-uploading files
if rescore and st.session_state.mapping_config:
    if 'original_ikea_df' in st.session_state and 'original_retail_df' in st.session_state:
        # Get current sidebar configuration
        current_config = MapperConfig(
            retailer_name=retailer_name,
            retailer_domain=retailer_domain,
            ratio=ratio,
            min_score=min_score,
            min_keep=min_keep,
            verify_urls=verify,
            scoring_method=scoring_method,
            l1_weight=l1_weight,
            l2_weight=l2_weight,
            l3_weight=l3_weight,
            examples_weight=examples_weight,
        )
        
        # Use stored original data but current sidebar settings
        restored_synonyms = st.session_state.mapping_config['synonyms']
        restored_overrides = st.session_state.mapping_config['overrides']
        restored_blacklist = st.session_state.mapping_config['generic_blacklist']
        
        with st.spinner("🔄 Re-scoring with new settings..."):
            mapper = CatalogMapper(current_config, synonyms=restored_synonyms, overrides=restored_overrides, generic_blacklist=restored_blacklist)
            result = mapper.map(st.session_state.original_ikea_df, st.session_state.original_retail_df)
            
            # Update results in session state
            st.session_state.mapping_results = result.copy()
            # Update configuration with new settings
            st.session_state.mapping_config.update({
                'scoring_method': scoring_method,
                'l1_weight': l1_weight,
                'l2_weight': l2_weight,
                'l3_weight': l3_weight,
                'examples_weight': examples_weight
            })
            # Reset edited results since we have new mapping
            st.session_state.edited_results = None
            
            st.success(f"🔄 Re-scored! Generated {len(result)} mapped rows with new settings.")
            st.info(f"📊 **New Settings Applied**: {scoring_method.upper()} scoring, weights: L1={l1_weight}, L2={l2_weight}, L3={l3_weight}, Examples={examples_weight}")
    else:
        st.warning("⚠️ Original data not found. Please re-upload files and run mapping first.")
        st.info("💡 **Tip**: Upload files and run mapping once, then you can experiment with different settings!")

# Show results if they exist in session state
if st.session_state.get('mapping_results') is not None:
    result = st.session_state.mapping_results.copy()
    # Restore config from session state
    if st.session_state.mapping_config:
        retailer_name = st.session_state.mapping_config['retailer_name']
        synonyms = st.session_state.mapping_config['synonyms']
        overrides = st.session_state.mapping_config['overrides']
        generic_blacklist = st.session_state.mapping_config['generic_blacklist']

    # Settings transparency: Show current applied settings prominently
    # This helps users understand which parameters generated the current results
    # and confirms when new settings have been applied after experimentation
    if st.session_state.mapping_config:
        config = st.session_state.mapping_config
        st.info(f"🎛️ **Current Settings**: {config.get('scoring_method', 'default').upper()} scoring • Weights: L1={config.get('l1_weight', 1)}, L2={config.get('l2_weight', 3)}, L3={config.get('l3_weight', 6)}, Examples={config.get('examples_weight', 2)}")
    
    # Add a clear results button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Review & Edit")
    with col2:
        if st.button("🗑️ Clear Results", help="Clear current mapping results and start over"):
            st.session_state.mapping_results = None
            st.session_state.mapping_config = None
            st.session_state.edited_results = None
            st.rerun()
    
    # Add editing mode selector
    edit_mode = st.radio(
        "Choose editing mode:",
        ["💻 Direct Input (type 0 or 1)", "☑️ Checkbox (click to approve)"],
        horizontal=True,
        help="Choose how you want to edit the human labels",
        key="edit_mode_selector"
    )
    
    # Add instruction based on mode
    if "Checkbox" in edit_mode:
        st.info("📝 **Excel-like Editing**: ✓ **Checkboxes** to approve/reject mappings • **Double-click** catalog paths and URLs to edit • Changes save automatically")
    else:
        st.info("📝 **Excel-like Editing**: **Type** `1`/`0` to approve/reject • **Double-click** catalog paths and URLs to edit • Press **Enter** to save changes")
    
    # Ensure human_label column exists and is the right type
    if 'human_label' not in result.columns:
        result['human_label'] = 0
    
    # Convert human_label to appropriate type based on mode
    if "Checkbox" in edit_mode:
        result['human_label'] = result['human_label'].astype(bool)
        column_config = {
            "human_label": st.column_config.CheckboxColumn(
                "Approve",
                help="Check to approve mapping, uncheck to reject",
                width="small",
            )
        }
        st.caption("☑️ **Approve/Reject**: Check boxes • **Edit Paths/URLs**: Double-click cells • **Auto-save**: Changes persist automatically")
    else:
        result['human_label'] = result['human_label'].astype(int)
        column_config = {
            "human_label": st.column_config.NumberColumn(
                "Human Label",
                help="1 = Approve mapping, 0 = Reject mapping",
                width="small",
                min_value=0,
                max_value=1,
                step=1,
                format="%d"
            )
        }
        st.caption("💻 **Approve/Reject**: Type 1/0 • **Edit Paths/URLs**: Double-click cells • **Save**: Press Enter")
    
    # Make key columns editable like Excel - allow editing catalog path, URL, and human label
    editable_columns = ['human_label']
    
    # Find the retailer-specific columns dynamically
    retailer_path_col = f"{retailer_name}_catalog_path"
    retailer_url_col = f"{retailer_name}_url"
    
    if retailer_path_col in result.columns:
        editable_columns.append(retailer_path_col)
    if retailer_url_col in result.columns:
        editable_columns.append(retailer_url_col)
    
    # Create list of columns to disable (all except editable ones)
    disabled_columns = [col for col in result.columns if col not in editable_columns]
    
    # Hide less important columns for cleaner interface
    columns_to_hide = ['matched_tokens', '_method'] if len(result.columns) > 8 else []
    
    # Use edited results from session state if available, otherwise use original
    if st.session_state.edited_results is not None:
        data_to_edit = st.session_state.edited_results.copy()
        # Ensure the data type matches the current edit mode
        if "Checkbox" in edit_mode:
            if data_to_edit['human_label'].dtype != bool:
                data_to_edit['human_label'] = data_to_edit['human_label'].astype(bool)
        else:
            if data_to_edit['human_label'].dtype != int:
                data_to_edit['human_label'] = data_to_edit['human_label'].astype(int)
    else:
        data_to_edit = result.copy()
    
    # Enhanced column configuration for better editing experience
    if retailer_path_col in data_to_edit.columns:
        column_config[retailer_path_col] = st.column_config.TextColumn(
            "📁 Catalog Path",
            help="Edit the retailer catalog path - double-click to modify",
            width="medium"
        )
    
    if retailer_url_col in data_to_edit.columns:
        column_config[retailer_url_col] = st.column_config.LinkColumn(
            "🔗 URL", 
            help="Edit the retailer URL - double-click to modify",
            width="medium"
        )
    
    # Dynamic editor refresh: Create a key that changes when results change
    # This ensures the data editor shows fresh data when re-mapping with different settings
    # Without this, the editor would show cached data from previous scoring attempts
    if st.session_state.mapping_config:
        config = st.session_state.mapping_config
        result_signature = f"{len(result)}_{config.get('scoring_method', 'default')}_{config.get('l1_weight', 1)}_{config.get('l2_weight', 3)}_{config.get('l3_weight', 6)}_{config.get('examples_weight', 2)}"
    else:
        result_signature = f"{len(result)}_default"
    editor_key = f"mapping_editor_{abs(hash(result_signature)) % 10000}"
    
    # Update column config to hide less important columns
    for col in columns_to_hide:
        if col in data_to_edit.columns:
            column_config[col] = None  # Hide column
    
    edited_result = st.data_editor(
        data_to_edit, 
        width='stretch', 
        height=520,
        column_config=column_config,
        disabled=disabled_columns,
        key=editor_key,
        hide_index=True
    )
    
    # Always update session state with current edits
    st.session_state.edited_results = edited_result.copy()
    
    # Robust Excel export: Handle missing dependencies gracefully
    # Provides fallback to openpyxl if xlsxwriter is not available
    # This ensures users can always download results even with partial dependencies
    try:
        xls_bytes = to_xlsx_bytes(edited_result)
        st.download_button(
            "⬇️ Download Augmented XLSX (with your edits)",
            data=xls_bytes,
            file_name=f"range_taxonomy_{retailer_name}_mapping_edited.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except ImportError:
        st.warning("⚠️ Excel download requires xlsxwriter: `pip install xlsxwriter`")
    except Exception as e:
        st.error(f"❌ Download error: {e}")
    
    # Add a manual save button for better control
    if st.button("💾 Apply Changes", help="Manually save your current edits"):
        st.success("✅ Changes applied and saved!")

    # Show summary of human labels
    if 'human_label' in edited_result.columns:
        # Handle both boolean and integer types
        if "Checkbox" in edit_mode:
            approved = edited_result['human_label'].sum()
            rejected = (~edited_result['human_label']).sum()
        else:
            approved = (edited_result['human_label'] == 1).sum()
            rejected = (edited_result['human_label'] == 0).sum()
        
        total = len(edited_result)
        st.info(f"📊 Summary: {approved} approved, {rejected} rejected out of {total} total mappings")
        
        # Show progress bar
        if total > 0:
            progress = approved / total
            st.progress(progress, text=f"Progress: {approved}/{total} mappings approved ({progress:.1%})")
    
    st.divider()
    
    # Instructions
    with st.expander("📚 Excel-like Editing Guide", expanded=False):
        st.markdown("""
        **🎯 Main Editing (Excel-like Experience):**
        - **Double-click** any cell in 📁 **Catalog Path** to edit retailer category names
        - **Double-click** any cell in 🔗 **URL** to edit retailer links  
        - **Click checkboxes** or **type 1/0** in **Human Label** to approve/reject
        - **Changes save automatically** - no need to click save buttons!
        
        **⚡ Quick Actions:**
        - **Edit paths**: Fix category names, add missing levels, correct typos
        - **Edit URLs**: Update broken links, add missing URLs, fix domains
        - **Approve/Reject**: Mark good matches (✓) vs poor matches (✗)
        
        **💡 Pro Tips:**
        - Edit paths to match your retailer's exact category structure
        - Add or fix URLs to ensure working links in final output
        - Use the progress bar to track your review completion
        """)




    # Check if there are any changes from original
    if 'human_label' in result.columns and 'human_label' in edited_result.columns:
        try:
            # Get original values from the initial mapping results
            original_result = st.session_state.mapping_results
            if 'human_label' not in original_result.columns:
                original_result = original_result.copy()
                original_result['human_label'] = 0
            
            # Convert both to same type for comparison
            if "Checkbox" in edit_mode:
                original_vals = original_result['human_label'].astype(bool)
                edited_vals = edited_result['human_label'].astype(bool)
            else:
                original_vals = original_result['human_label'].astype(int)
                edited_vals = edited_result['human_label'].astype(int)
            
            changes_made = not original_vals.equals(edited_vals)
            if changes_made:
                changed_count = (original_vals != edited_vals).sum()
                st.success(f"✏️ {changed_count} changes detected! Your edits are automatically saved.")
        except Exception:
            pass  # Skip change detection if there are type issues
    
    # Save edited results feature
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Edited Results as CSV", help="Save current edits as CSV for future reference"):
            csv_path = DATA_DIR / f"edited_mapping_{retailer_name.lower()}.csv"
            edited_result.to_csv(csv_path, index=False)
            st.success(f"✅ Saved edited results to: {csv_path}")
    
    with col2:
        if st.button("🔄 Reset All Labels to 0", help="Reset all human labels to 0 (rejected)"):
            # Reset edited results
            reset_result = st.session_state.mapping_results.copy()
            reset_result['human_label'] = 0
            st.session_state.edited_results = reset_result
            st.rerun()

else:
    st.info("👆 Upload IKEA and Retailer files above, then click 'Run Mapping' to begin.")
