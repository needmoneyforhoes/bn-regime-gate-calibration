# bn-regime-gate-calibration

BN-regime gate calibration for Polymarket: pro calibrator, audits, data checks, and Part-15 DN-suppression sweeps.

Offline research tooling for the **Polymarket 5-minute crypto up/down** trading suite (BTC/XRP). It calibrates and validates the bot's *regime gates* — filters that suppress fires whose side disagrees with the prevailing **BN** (Binance-vs-Polymarket basis) signal at a given countdown checkpoint.

## Why it exists
The live bot demotes/blocks fires when the order-book regime contradicts the BN delta. Picking *which* checkpoint (countdown `cd`) and *what* BN threshold to gate on is a statistics problem, not a guess. This repo finds those parameters with proper rigor (Bonferroni correction, out-of-sample splits, permutation tests, Wilson CIs) before anything is wired into the production engine.

## What's inside
All scripts are **offline analysis** — they read recorded market/recap history and print recommendations. Nothing trades.

| Script | Role |
| --- | --- |
| `regime_gate_pro_calibrator.py` | Flagship 14-phase calibrator: vectorized BN reconstruction, grid search, ROC/AUC, L2-logistic + RF benchmarks, permutation test, walk-forward CV, bootstrap CIs, Kelly sizing. Emits JSON/TXT/CSV (+ optional plots). |
| `part15_cd_sweep.py` | Sweeps every `cd` checkpoint 300→150 × BN threshold to find the earliest reliable DN-suppression signal, full Bonferroni across all combos. |
| `part15_gateaware.py` | Part-15 analysis counting **only** DN fires that pass all existing live gates — the incremental value of the new gate on top of what's already deployed. |
| `part15_sweep_gateaware.py` | Gate-aware `cd` sweep maximising out-of-sample EV while holding precision. |
| `strong_up_gate_analysis.py` | Hypothesis test for the Part-15 DN-suppression gate (strong-UP BN + crowd at T-200), the symmetric mirror of Part 14. |
| `shadow_gates.py` | **Library**, not a script: log-only Gate 3 (strong-BN-against) + Gate 4 (late BN-flip) evaluators meant to be imported into the live bot for pre-registered shadow validation. Run with `--self-test`. |
| `regime_gate_quick_audit.py` | Fast sanity check on the gate's false-positive rate from recap history alone (no tick data needed). |
| `regime_gate_data_check.py` | Inspects a history file: tick counts, columns, BN availability. Run this first. |
| `regime_gate_install_check.py` | Verifies numpy / scipy / sklearn / matplotlib are present. |

## Requirements
- Python 3.9+
- `numpy`, `scipy`, `scikit-learn` (required); `matplotlib` optional (plots).
- Run `python regime_gate_install_check.py` to verify.
- No wallet, private key, or network access — this repo never touches funds.

## Usage
```bash
# 0. verify deps and that your data has BN/tick columns
python regime_gate_install_check.py
python regime_gate_data_check.py ~/polymarket-bot/market_history.jsonl

# 1. flagship calibration (writes regime_gate_analysis.json / _recommendation.txt / _dataset.csv)
python regime_gate_pro_calibrator.py market_history.jsonl market_recap_history.jsonl

# 2. Part-15 DN-suppression checkpoint sweeps
python part15_cd_sweep.py
python part15_sweep_gateaware.py

# quick FP-rate audit (recap only)
python regime_gate_quick_audit.py market_recap_history.jsonl

# shadow-gate library unit tests
python shadow_gates.py --self-test
```

## Data
These scripts load `market_history.jsonl` (per-market tick data: `cd`, `bn_delta_pct`, `crowd_*`, …) and `market_recap_history.jsonl` (fires + winners). The Part-15 scripts default to `~/polymarket-bot/*.jsonl`; the calibrator and audits take the paths as arguments. Both files come from the **private `polymarket-data` repo** and are git-ignored here (`*.jsonl`/`*.json`/`*.csv`/`*.pkl`) — point the scripts at your local copy.

> Private research software. No warranty; trades/handles real funds at your own risk.
