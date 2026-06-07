# bn-regime-gate-calibration

Offline tooling to calibrate and validate the Polymarket 5-minute crypto up/down bot's regime gates: filters that suppress fires whose side disagrees with the BN (Binance-vs-Polymarket basis) signal at a given countdown checkpoint.

All scripts read recorded history and print recommendations. Nothing trades. Statistics use Bonferroni correction, 60/40 in/out-of-sample splits, 10k-iteration permutation tests, and Wilson CIs.

## Scripts

| Script | What it does |
| --- | --- |
| `regime_gate_pro_calibrator.py` | 14-phase calibrator: vectorized BN reconstruction, grid search, ROC/AUC, L2-logistic + random forest, permutation test, walk-forward CV, bootstrap CIs, Kelly sizing. Writes JSON/TXT/CSV and optional plots. |
| `part15_cd_sweep.py` | Sweeps cd 300 to 150 (step 10) x 6 BN thresholds for the earliest reliable DN-suppression signal, Bonferroni across all combos. |
| `part15_gateaware.py` | Counts only DN fires that pass all live gates, to measure the new gate's incremental value. |
| `part15_sweep_gateaware.py` | Gate-aware cd sweep (290 to 150) maximizing out-of-sample EV while holding precision. |
| `strong_up_gate_analysis.py` | Hypothesis test for the Part-15 DN-suppression gate (strong-UP BN + crowd at T-200), the mirror of Part 14. |
| `shadow_gates.py` | Library, not a CLI: log-only Gate 3 (strong-BN-against) and Gate 4 (late BN-flip) evaluators for import into the live bot. Run `--self-test`. |
| `regime_gate_quick_audit.py` | False-positive-rate check from recap history alone, no tick data. |
| `regime_gate_data_check.py` | Inspects a history file: tick counts, columns, BN availability. Run first. |
| `regime_gate_install_check.py` | Checks numpy / scipy / sklearn / matplotlib. |

## Requirements

Python 3.9+, numpy, scipy, scikit-learn. matplotlib optional for plots. Read-only; no credentials required.

## Usage

```bash
python regime_gate_install_check.py
python regime_gate_data_check.py $DATA_DIR/market_history.jsonl

# flagship calibration (writes regime_gate_analysis.json / _recommendation.txt / _dataset.csv)
python regime_gate_pro_calibrator.py market_history.jsonl market_recap_history.jsonl

# Part-15 DN-suppression checkpoint sweeps
python part15_cd_sweep.py
python part15_sweep_gateaware.py

# FP-rate audit (recap only)
python regime_gate_quick_audit.py market_recap_history.jsonl

# shadow-gate self-test
python shadow_gates.py --self-test
```

## Data

Scripts load `market_history.jsonl` (per-market ticks: `cd`, `bn_delta_pct`, `crowd_*`) and `market_recap_history.jsonl` (fires + winners). Part-15 scripts default to `$DATA_DIR/*.jsonl`; the calibrator and audits take paths as arguments. Both files come from the private polymarket-data repo and are git-ignored.
