
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
    
    if embeddings_available:
        scoring_method = st.selectbox(
            "Scoring Method",
            options=["tfidf", "embeddings"],
            format_func=lambda x: {
                "tfidf": "🔤 TF-IDF (Fast, good for simple catalogs)",
                "embeddings": "🧠 Semantic Embeddings (Slower, better for complex catalogs)"
            }[x],
            help="TF-IDF: Fast keyword matching. Embeddings: Understands synonyms and context."
        )
    else:
        scoring_method = "tfidf"
        st.info("💡 Install `sentence-transformers` to enable semantic embeddings: `pip install sentence-transformers`")

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

run = st.button("Run Mapping", type="primary")

if run:
    if not ikea_up or not ret_up:
        st.warning("Please upload both IKEA and Retailer files.")
        st.stop()

    with st.spinner("Parsing files…"):
        ikea_df = read_ikea_xlsx(ikea_up)
        retail_df = read_retailer_xlsx(ret_up)

    st.success(f"Parsed IKEA rows: {len(ikea_df)} | Retailer rows: {len(retail_df)}")

    cfg = MapperConfig(
        retailer_name=retailer_name,
        retailer_domain=retailer_domain or overrides.get("domain"),
        one_to_many_ratio=ratio,
        min_score=min_score,
        min_keep=min_keep,
        verify_urls=verify,
        scoring_method=scoring_method,
    )

    mapper = CatalogMapper(cfg, synonyms=synonyms, overrides=overrides, generic_blacklist=generic_blacklist)

    with st.spinner("Scoring and selecting candidates…"):
        result = mapper.map(ikea_df, retail_df)

    # Store results in session state
    st.session_state.mapping_results = result.copy()
    st.session_state.mapping_config = {
        'retailer_name': retailer_name,
        'synonyms': synonyms,
        'overrides': overrides,
        'generic_blacklist': generic_blacklist
    }

    st.success(f"Generated {len(result)} mapped rows (one‑to‑many duplicates included).")

# Show results if they exist in session state
if st.session_state.mapping_results is not None:
    result = st.session_state.mapping_results.copy()
    # Restore config from session state
    if st.session_state.mapping_config:
        retailer_name = st.session_state.mapping_config['retailer_name']
        synonyms = st.session_state.mapping_config['synonyms']
        overrides = st.session_state.mapping_config['overrides']
        generic_blacklist = st.session_state.mapping_config['generic_blacklist']

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
        st.info("📝 **Checkbox Mode**: Click checkboxes to approve (✓) or reject ( ) mappings. Changes save automatically.")
    else:
        st.info("📝 **Direct Input Mode**: Type `1` to approve or `0` to reject, then press Enter. Changes save automatically.")
    
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
        st.caption("☑️ Check boxes to approve mappings, leave unchecked to reject.")
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
        st.caption("💻 Type 1 to approve, 0 to reject. Press Enter to save each change.")
    
    # Create list of columns to disable (all except human_label)
    disabled_columns = [col for col in result.columns if col != 'human_label']
    
    # Hide less important columns for cleaner interface
    columns_to_hide = ['matched_tokens', '_method'] if len(result.columns) > 8 else []
    
    # Use data_editor for interactive editing
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
    
    # Create a stable key that doesn't change with mode switches
    editor_key = "mapping_editor_stable"
    
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
    with st.expander("ℹ️ How to edit human labels", expanded=False):
        st.markdown("""
        **Editing Instructions:**
        1. **Click** on any cell in the 'Human Label' column
        2. **Type** `1` to approve the mapping or `0` to reject it
        3. **Press Enter** or click outside the cell to save the change
        4. The summary above updates automatically
        5. Use the download button below to save your edited results
        
        **Legend:**
        - `0` = Reject mapping (poor quality/incorrect)
        - `1` = Approve mapping (good quality/correct)
        """)

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
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="mapping")
        return output.getvalue()

    # Use edited results for download
    xls_bytes = to_xlsx_bytes(edited_result)
    st.download_button(
        "⬇️ Download Augmented XLSX (with your edits)",
        data=xls_bytes,
        file_name=f"range_taxonomy_{retailer_name}_mapping_edited.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

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

    st.divider()
    st.subheader("Apply feedback → overrides")
    st.caption("Add deterministic overrides for IKEA categories. This helps improve future mappings.")

    # Create tabs for different override types
    tab1, tab2 = st.tabs(["🔗 IKEA Category Override", "➕ Add New Mapping"])
    
    with tab1:
        st.markdown("**Override existing IKEA category mapping:**")
        cc1, cc2 = st.columns(2)
        with cc1:
            leaf_in = st.text_input(
                "IKEA leaf (canonical, e.g., 'sofas')", 
                help="Enter the IKEA category name (usually L3 level)"
            )
            path_in = st.text_input(
                "Retailer catalog path", 
                help="Enter the exact retailer category path as it appears"
            )
        with cc2:
            url_in = st.text_input(
                "Retailer URL (optional)", 
                help="Full URL to the retailer category page"
            )
            save_btn = st.button("💾 Save Category Override")

        if save_btn and leaf_in.strip() and path_in.strip():
            overrides.setdefault("leaf_overrides", {})[leaf_in.strip().lower()] = {
                "catalog_path": path_in.strip(),
                **({"url": url_in.strip()} if url_in.strip() else {})
            }
            out_path = DATA_DIR / f"overrides_{retailer_name.lower()}.json"
            out_path.write_text(json.dumps(overrides, indent=2))
            st.success(f"✅ Saved category override → {out_path}")

    with tab2:
        st.markdown("**Add completely new retailer category mapping:**")
        cc3, cc4 = st.columns(2)
        with cc3:
            new_path_in = st.text_input(
                "New retailer catalog path", 
                help="Enter any retailer category path that was missed"
            )
            new_url_in = st.text_input(
                "Retailer URL for new path", 
                help="Full URL to the retailer category page"
            )
        with cc4:
            ikea_match = st.text_input(
                "Best matching IKEA category", 
                help="Which IKEA category should this map to?"
            )
            add_new_btn = st.button("➕ Add New Mapping")

        if add_new_btn and new_path_in.strip():
            # Add to a new section for manual additions
            overrides.setdefault("manual_additions", []).append({
                "catalog_path": new_path_in.strip(),
                "url": new_url_in.strip() if new_url_in.strip() else "",
                "ikea_match": ikea_match.strip() if ikea_match.strip() else "",
                "added_manually": True
            })
            out_path = DATA_DIR / f"overrides_{retailer_name.lower()}.json"
            out_path.write_text(json.dumps(overrides, indent=2))
            st.success(f"✅ Added new mapping → {out_path}")
            st.info("💡 This new path will be included in future mapping runs.")

else:
    st.info("👆 Upload IKEA and Retailer files above, then click 'Run Mapping' to begin.")
