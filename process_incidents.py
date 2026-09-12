"""
Textile QA – Shortfall Incident Extractor
Reads two Excel files, extracts shortfall records, anonymises them,
and writes a clean CSV ready for downstream analysis.
"""

import re
import pandas as pd

# ── 1. Paths ──────────────────────────────────────────────────────────────────
COMPOSITE_PATH = "data/raw/Composite_Mapping.xlsx"
SF_PATH        = "data/raw/SF-Daily_Update.xlsx"
OUTPUT_PATH    = "data/clean/incidents.csv"

# ── 2. Read source files ──────────────────────────────────────────────────────
# Load the QA mapping sheet (kept for reference / future join)
composite_df = pd.read_excel(COMPOSITE_PATH, sheet_name="Testing")

# Load the salvage-shortfall sheet; sheet name may vary, so we grab the first
# sheet whose name contains "SF" or "Batch" (case-insensitive).
xl = pd.ExcelFile(SF_PATH)
sf_sheet = next(
    (s for s in xl.sheet_names if re.search(r"sf|batch", s, re.I)),
    xl.sheet_names[0],          # fallback: first sheet
)
sf_df = pd.read_excel(SF_PATH, sheet_name=sf_sheet)

# ── 3. Normalise column names (strip whitespace, lower-case) ──────────────────
sf_df.columns = sf_df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)

# ── 4. Locate the three required columns with flexible name matching ───────────
# Maps our canonical name → regex pattern to find the real column name.
COL_PATTERNS = {
    "reason":           r"reason",
    "responsible_dept": r"responsible.*(dept|department)",
    "action_plan":      r"action.*plan",
}

col_map = {}
for canonical, pattern in COL_PATTERNS.items():
    match = next((c for c in sf_df.columns if re.search(pattern, c, re.I)), None)
    if match is None:
        raise KeyError(f"Cannot find a column matching '{pattern}' in {sf_sheet}. "
                       f"Available columns: {list(sf_df.columns)}")
    col_map[canonical] = match

# ── 5. Extract and rename to canonical schema ─────────────────────────────────
incidents = sf_df[[col_map["reason"],
                   col_map["responsible_dept"],
                   col_map["action_plan"]]].copy()

incidents.columns = ["reason", "responsible_dept", "action_plan"]

# Drop rows where all three fields are empty (blank spacer rows in Excel)
incidents.dropna(how="all", inplace=True)
incidents.reset_index(drop=True, inplace=True)

# ── 6. Add a stable incident ID ───────────────────────────────────────────────
incidents.insert(0, "incident_id", ["INC-{:04d}".format(i + 1) for i in incidents.index])

# ── 7. Anonymisation ──────────────────────────────────────────────────────────
# Patterns that identify sensitive tokens in free-text fields.
ANON_RULES = [
    # Order / PO codes  e.g. PO-12345, ORD/2024/001
    (r"\b(PO|ORD|SO|WO)[-/]?\d[\w/-]*", "<ORDER_CODE>"),
    # Project / style codes  e.g. PRJ-ABC, STY2024
    (r"\b(PRJ|STY|PROJ)[-]?[A-Z0-9]+", "<PROJECT_CODE>"),
    # Operator / employee IDs  e.g. OP-007, EMP1234
    (r"\b(OP|EMP|OPR)[-]?\d+", "<OPERATOR_ID>"),
    # Standalone numeric IDs of 4+ digits (catch-all for codes not matched above)
    (r"\b\d{4,}\b", "<ID>"),
]

def anonymise(text: str) -> str:
    if not isinstance(text, str):
        return text
    for pattern, replacement in ANON_RULES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text.strip()

for col in ("reason", "action_plan"):          # free-text columns only
    incidents[col] = incidents[col].apply(anonymise)

# ── 8. Save ───────────────────────────────────────────────────────────────────
incidents.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")  # utf-8-sig for Excel compat
print(f"Saved {len(incidents)} incidents → {OUTPUT_PATH}")
