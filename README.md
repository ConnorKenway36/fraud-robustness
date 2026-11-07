# Constraint-Aware Robustness of Fraud Detectors (Month 1 foundation)

Foundation work for a study of whether adversarial defences for financial
fraud detectors stay robust across retraining cycles with delayed labels.

**Scope of this repository (current):** prior-art review, data generation
and cleaning, and frozen field-level constraint files. Attacks, defences
and the retraining study are future work.

**Detectors planned:** LightGBM and random forest.

## Environment

- Python: <fill in, e.g. 3.11.x>
- OS: <fill in>
- Environment frozen on: <date>
- Setup:

  ```bash
  python -m venv .venv
  source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```

## Data

- Primary dataset: Fraud Detection Handbook simulator (synthetic).
- Source: https://github.com/Fraud-Detection-Handbook/simulated-data-raw
- Source commit hash: <fill in after cloning>
- Data is not stored in this repository. To recreate the folders:

  ```bash
  mkdir -p data/raw data/parquet
  ```

## Repository layout

| Folder | Contents |
|---|---|
| `src/` | Loading, feature engineering, validation code |
| `configs/` | Cycle boundaries, label delays, seeds |
| `constraints/` | Frozen constraint files and schema |
| `manifests/` | Source commit hash, row counts, checksums |
| `notebooks/` | Exploration only |
| `docs/` | Literature review and threat-model write-up |

## Notes

- The data is synthetic. Results describe the simulator, not a real bank.
- "Valid" means valid under the declared constraints, not certified
  feasible in a live bank.
- The Handbook code is GPL-3.0. Licence for this repository: <decide>.