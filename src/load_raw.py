"""Convert the Handbook's daily raw pickles into one time-sorted Parquet file,
run sanity checks, and write a manifest.

Run from the project root:
    python -m src.load_raw
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow
import pyarrow.parquet as pq

REQUIRED_COLS = ["TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID",
                 "TERMINAL_ID", "TX_AMOUNT", "TX_FRAUD"]
EXPECTED_FILES = 183
EXPECTED_CUSTOMERS = 5000
EXPECTED_TERMINALS = 10000
EXPECTED_ROWS = (1_500_000, 2_000_000)  # proposal says about 1.75M


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(folder: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(folder), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def find_files(raw_dir: Path) -> list[Path]:
    files = sorted(raw_dir.rglob("*.pkl"))
    if not files:
        seen = sorted({p.suffix for p in raw_dir.rglob("*") if p.is_file()})
        sys.exit(f"No .pkl files found under {raw_dir}. "
                 f"File extensions present: {seen or 'none'}")
    return files


def run_checks(df: pd.DataFrame, n_files: int):
    errors, warnings = [], []

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return errors, warnings

    if df[REQUIRED_COLS].isna().any().any():
        errors.append("Nulls found in required columns")
    if df["TRANSACTION_ID"].duplicated().any():
        errors.append("Duplicate TRANSACTION_ID values")
    if (df["TX_AMOUNT"] < 0).any():
        errors.append("Negative TX_AMOUNT values")
    if not set(df["TX_FRAUD"].unique()) <= {0, 1}:
        errors.append("TX_FRAUD contains values other than 0/1")

    if n_files != EXPECTED_FILES:
        warnings.append(f"{n_files} files found, proposal expects {EXPECTED_FILES}")
    if not (EXPECTED_ROWS[0] <= len(df) <= EXPECTED_ROWS[1]):
        warnings.append(f"{len(df):,} rows, proposal expects about 1.75M")
    if df["CUSTOMER_ID"].nunique() != EXPECTED_CUSTOMERS:
        warnings.append(f"{df['CUSTOMER_ID'].nunique()} customers, expected {EXPECTED_CUSTOMERS}")
    if df["TERMINAL_ID"].nunique() != EXPECTED_TERMINALS:
        warnings.append(f"{df['TERMINAL_ID'].nunique()} terminals, expected {EXPECTED_TERMINALS}")
    span = (df["TX_DATETIME"].max().normalize() - df["TX_DATETIME"].min().normalize()).days + 1
    if span != EXPECTED_FILES:
        warnings.append(f"Data spans {span} days, proposal expects {EXPECTED_FILES}")
    return errors, warnings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/parquet/handbook_transactions.parquet"))
    ap.add_argument("--manifest", type=Path, default=Path("manifests/handbook_manifest.json"))
    args = ap.parse_args()

    files = find_files(args.raw_dir)
    print(f"Found {len(files)} pickle files "
          f"({files[0].name} ... {files[-1].name})")

    # Only load pickles from the official repo at a pinned commit: pickle can run code.
    df = pd.concat((pd.read_pickle(f) for f in files), ignore_index=True)
    print("Columns:", df.columns.tolist())
    print(df.dtypes.to_string())

    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])
    was_sorted = bool(df["TX_DATETIME"].is_monotonic_increasing)

    errors, warnings = run_checks(df, len(files))
    for w in warnings:
        print("WARNING:", w)
    if errors:
        for e in errors:
            print("ERROR:", e)
        sys.exit("Hard checks failed; nothing was written.")

    df = df.sort_values(["TX_DATETIME", "TRANSACTION_ID"], kind="stable").reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, engine="pyarrow", compression="zstd", index=False)
    assert pq.ParquetFile(args.out).metadata.num_rows == len(df)

    row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
    content_hash = format(int(row_hashes.sum(dtype="uint64")), "016x")
    listing = "\n".join(f"{p.name}:{p.stat().st_size}" for p in files)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo": "https://github.com/Fraud-Detection-Handbook/simulated-data-raw",
        "source_commit": git_commit(files[0].parent),
        "n_source_files": len(files),
        "source_listing_sha256": hashlib.sha256(listing.encode()).hexdigest(),
        "input_was_time_sorted": was_sorted,
        "rows": int(len(df)),
        "columns": {c: str(t) for c, t in df.dtypes.items()},
        "date_min": str(df["TX_DATETIME"].min()),
        "date_max": str(df["TX_DATETIME"].max()),
        "n_customers": int(df["CUSTOMER_ID"].nunique()),
        "n_terminals": int(df["TERMINAL_ID"].nunique()),
        "fraud_rate": float(df["TX_FRAUD"].mean()),
        "parquet_file": str(args.out),
        "parquet_sha256": sha256_file(args.out),
        "content_hash": content_hash,
        "warnings": warnings,
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "pyarrow": pyarrow.__version__, "os": platform.platform()},
    }
    args.manifest.write_text(json.dumps(manifest, indent=2))

    print(f"\nWrote {args.out} ({len(df):,} rows)")
    print(f"Wrote {args.manifest}")
    print(f"Fraud rate: {manifest['fraud_rate']:.4%}   "
          f"Dates: {manifest['date_min']} to {manifest['date_max']}")


if __name__ == "__main__":
    main()