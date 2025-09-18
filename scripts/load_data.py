import pandas as pd
from typing import Optional, Iterable

PPD_COLS = [
    "transaction_id","price_paid","date_of_transfer","postcode","property_type",
    "new_build_flag","tenure","primary_address","secondary_address","street",
    "locality","town_city","district","county","ppd_category_type","record_status"
]

PROPERTY_TYPE_MAP = {"D":"Detached","S":"Semi-Detached","T":"Terraced","F":"Flat/Maisonette","O":"Other"}
YN_MAP = {"Y":1,"N":0}
TENURE_MAP = {"F":"Freehold","L":"Leasehold"}

# Fallback: outward codes that broadly cover Greater London
LONDON_PREFIXES = (
    "E","EC","N","NW","SE","SW","W","WC","HA","UB","TW","EN","IG","RM","DA","BR","CR","SM","KT"
)

def load_clean_ppd(
    csv_path: str,
    *,
    ons_lookup: Optional[pd.DataFrame] = None,   # optional ONS Postcode Directory with ['postcode','lat','lon','lad','lsoa','msoa']
    london_only: bool = True,
    keep_category_b: bool = False,
    min_price: int = 10_000,
    drop_changed_deleted: bool = True,
    extra_postcode_prefixes: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """
    Load & clean HM Land Registry Price Paid Data (PPD).

    Params
    ------
    csv_path : path to raw PPD CSV (no header).
    ons_lookup : optional ONS postcode directory merged by exact postcode (no space, upper).
                 If provided, London filter will use ONS 'lad' (local authority district) containing 'London'.
    london_only : if True, keep London rows only (via ONS or postcode prefix fallback).
    keep_category_b : if True, keep PPD category 'B' rows (defaults to only 'A').
    min_price : drop rows with price below this.
    drop_changed_deleted : if True, keep only record_status == 'A'.
    extra_postcode_prefixes : optional iterable of additional outward prefixes treated as London.
    """
    df = pd.read_csv(csv_path, header=None, names=PPD_COLS, dtype={"price_paid":"Int64"}, low_memory=False)

    # Basic trims
    df["postcode"] = df["postcode"].astype(str).str.upper().str.replace(" ", "", regex=False)
    df["date_of_transfer"] = pd.to_datetime(df["date_of_transfer"], errors="coerce")

    # Filter status/category
    if drop_changed_deleted:
        df = df[df["record_status"] == "A"]
    if not keep_category_b:
        df = df[df["ppd_category_type"] == "A"]

    # Valid ranges / null handling
    df = df[df["price_paid"].fillna(0) >= min_price]
    df = df.dropna(subset=["date_of_transfer", "postcode"])

    # Map codes
    df["property_type"] = df["property_type"].map(PROPERTY_TYPE_MAP).fillna("Other")
    df["new_build_flag"] = df["new_build_flag"].map(YN_MAP).astype("Int64")
    df["tenure"] = df["tenure"].map(TENURE_MAP).fillna(df["tenure"])

    # Time features
    df["year"] = df["date_of_transfer"].dt.year
    df["quarter"] = df["date_of_transfer"].dt.quarter

    # Merge ONS (optional)
    if ons_lookup is not None:
        ons = ons_lookup.copy()
        ons["postcode"] = ons["postcode"].astype(str).str.upper().str.replace(" ", "", regex=False)
        use_cols = [c for c in ["postcode","lat","lon","lad","lsoa","msoa","borough"] if c in ons.columns]
        df = df.merge(ons[use_cols].drop_duplicates("postcode"), on="postcode", how="left")

    # London filter
    if london_only:
        if "lad" in df.columns:
            # Use ONS LAD name/code; keep Greater London boroughs (lad often contains 'London')
            london_mask = df["lad"].astype(str).str.contains("London", case=False, na=False)
        else:
            # Fallback by postcode outward prefixes
            prefixes = set(LONDON_PREFIXES) | set(extra_postcode_prefixes or [])
            # outward = letters+digits until first digit sequence ends (simple slice works for most)
            london_mask = df["postcode"].str.startswith(tuple(sorted(prefixes, key=len, reverse=True)))
        df = df[london_mask]

    # Final tidy columns
    ordered = [
        "transaction_id","price_paid","date_of_transfer","year","quarter","postcode",
        "property_type","new_build_flag","tenure","primary_address","secondary_address",
        "street","locality","town_city","district","county","ppd_category_type","record_status"
    ] + [c for c in ["borough","lad","lsoa","msoa","lat","lon"] if c in df.columns]

    return df[ordered].reset_index(drop=True)

if __name__ == "__main__":
    df = load_clean_ppd("data/ppd/raw2025.csv", london_only=True)
    print(df.head(10))
    #df.to_parquet("data/london_ppd_prepared.parquet", index=False)
