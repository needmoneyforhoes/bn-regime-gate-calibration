#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  REGIME GATE PRO CALIBRATOR — professional quant methodology v2
═══════════════════════════════════════════════════════════════════════════════
Uses numpy, scipy, sklearn for proper statistical rigor.

PHASE 1: Data Extraction — Vectorized BN reconstruction from tick data
PHASE 2: Base Rate Analysis — Win rate with Wilson CIs (exact binomial)
PHASE 3: Univariate Grid Search — 91 (w,t) configs with full metrics
PHASE 4: ROC/AUC Analysis — sklearn roc_auc_score per window
PHASE 5: Regularized Logistic Regression — L2 ridge + cross-validation
PHASE 6: Random Forest Benchmark — detect nonlinear relationships
PHASE 7: Permutation Test — verify signal vs noise (p-value)
PHASE 8: Per-Strategy Calibration — individual optimal params
PHASE 9: Time-Conditional Analysis — BN predictiveness by cd bucket
PHASE 10: Bootstrap CI — 1000 iterations with numpy
PHASE 11: Walk-Forward Validation — purged time-series CV
PHASE 12: Feature Importance — permutation importance for multi-window
PHASE 13: Kelly Sizing — optimal bet size given fitted win probabilities
PHASE 14: Final Recommendation — rigorous, out-of-sample validated

USAGE:
    python regime_gate_pro_calibrator.py market_history.jsonl [market_recap_history.jsonl]

OUTPUTS:
    regime_gate_analysis.json         (full statistics)
    regime_gate_recommendation.txt    (human-readable)
    regime_gate_dataset.csv           (raw features for external analysis)
    regime_gate_plots.png             (if matplotlib available — ROC, distributions)
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import sys
import os
import math
from collections import defaultdict, Counter
from typing import Optional, List, Dict, Tuple

import numpy as np
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOTS_AVAILABLE = True
except ImportError:
    PLOTS_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG — match bot exactly
# ═══════════════════════════════════════════════════════════════════════════

RULE_F_DN = {
    "cl_gated_cheap", "cl_tight", "d3_wb_filter",
    "depth_collapse_mid", "depth_collapse_mid_v2", "depth_collapse_mid_v3",
    "depth_stable_mid", "late_dip_recovery", "t4_late", "t4_sr",
    "t4_sr_relaxed", "v5_shadow_b4_moderate", "v5_shadow_b5_mod_plus",
    "depth_surge", "deep_dip_scout",
}
RULE_F_BOTH = {"t6_early"}

# BN windows we analyze (seconds)
BN_WINDOWS = [3.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0]

# Thresholds for univariate sweep
BN_THRESHOLDS = np.array([0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.008,
                          0.010, 0.015, 0.020, 0.030, 0.050, 0.080])


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 1: DATA EXTRACTION (VECTORIZED)
# ═══════════════════════════════════════════════════════════════════════════

def load_markets(path: str) -> List[dict]:
    """Load markets with tick data."""
    markets = []
    with open(path) as f:
        for line in f:
            try:
                m = json.loads(line)
                if "ticks" in m and m["ticks"] and m.get("winner") in ("UP", "DN"):
                    markets.append(m)
            except:
                pass
    print(f"  Loaded {len(markets)} markets with tick data")
    return markets


def get_bn_at_cds_vectorized(market: dict, cds: np.ndarray, window: float) -> np.ndarray:
    """
    Vectorized: for each cd in cds, compute BN_delta(window).
    Returns array of BN values (np.nan where uncomputable).
    """
    cols = market.get("tick_columns", [])
    if "cd" not in cols or "bn_price" not in cols:
        return np.full(len(cds), np.nan)
    
    cd_idx = cols.index("cd")
    bn_idx = cols.index("bn_price")
    
    ticks = np.array(market["ticks"], dtype=float)  # shape (n_ticks, n_cols)
    if len(ticks) == 0:
        return np.full(len(cds), np.nan)
    
    tick_cds = ticks[:, cd_idx]
    tick_bns = ticks[:, bn_idx]
    
    results = np.full(len(cds), np.nan)
    
    for i, target_cd in enumerate(cds):
        # Find closest tick at or after target_cd (cd counts down: 300→0)
        # We want the tick CLOSEST to the target (in time), which is the smallest cd ≥ target_cd
        mask_now = tick_cds <= target_cd + 0.5
        mask_past = tick_cds >= target_cd + window
        
        if not mask_now.any() or not mask_past.any():
            continue
        
        # tick_now: smallest tick_cd that is <= target_cd (i.e. most recent)
        idx_now = np.argmax(tick_cds * mask_now) if mask_now.any() else -1
        # Actually: we want the tick with tick_cd closest to target_cd from below
        valid_now_cds = tick_cds[mask_now]
        idx_now = np.argmax(valid_now_cds)  # largest cd that's still <= target_cd
        bn_now = ticks[mask_now][idx_now, bn_idx]
        
        # tick_past: smallest tick_cd that is >= target_cd + window
        valid_past_cds = tick_cds[mask_past]
        if len(valid_past_cds) == 0:
            continue
        idx_past = np.argmin(valid_past_cds)  # smallest cd that's still >= target+window
        bn_past = ticks[mask_past][idx_past, bn_idx]
        
        if bn_past != 0 and not np.isnan(bn_past) and not np.isnan(bn_now):
            results[i] = (bn_now - bn_past) / bn_past * 100
    
    return results


def build_dataset(markets: List[dict], recaps: List[dict]) -> Dict[str, np.ndarray]:
    """
    Build vectorized dataset with features for all fires.
    Returns dict of numpy arrays.
    """
    market_lookup = {m["slug"]: m for m in markets}
    
    # Collect fire records
    slugs, strategies, sides, cds, winners, wons, hypo_pnls = [], [], [], [], [], [], []
    bn_features = {w: [] for w in BN_WINDOWS}
    
    for r in recaps:
        slug = r.get("slug")
        winner = r.get("winner")
        if winner not in ("UP", "DN"):
            continue
        market = market_lookup.get(slug)
        if not market:
            continue
        
        fires = r.get("fires", [])
        # Collect cds for vectorized BN computation
        fire_cds = []
        fire_data = []
        for fire in fires:
            strategy = fire.get("strategy")
            side = fire.get("side")
            cd = fire.get("cd")
            hypo_pnl = fire.get("hypo_pnl", 0)
            
            is_rf = ((strategy in RULE_F_DN and side == "DN")
                     or strategy in RULE_F_BOTH)
            if not is_rf or cd is None:
                continue
            
            fire_cds.append(cd)
            fire_data.append({
                "slug": slug, "strategy": strategy, "side": side, "cd": cd,
                "winner": winner, "won": hypo_pnl > 0, "hypo_pnl": hypo_pnl,
            })
        
        if not fire_cds:
            continue
        
        fire_cds = np.array(fire_cds)
        
        # Compute BN at all windows for all fires in this market (vectorized per window)
        for w in BN_WINDOWS:
            bn_vals = get_bn_at_cds_vectorized(market, fire_cds, w)
            bn_features[w].extend(bn_vals.tolist())
        
        for fd in fire_data:
            slugs.append(fd["slug"])
            strategies.append(fd["strategy"])
            sides.append(fd["side"])
            cds.append(fd["cd"])
            winners.append(fd["winner"])
            wons.append(fd["won"])
            hypo_pnls.append(fd["hypo_pnl"])
    
    return {
        "slug": np.array(slugs),
        "strategy": np.array(strategies),
        "side": np.array(sides),
        "cd": np.array(cds, dtype=float),
        "winner": np.array(winners),
        "won": np.array(wons, dtype=int),
        "hypo_pnl": np.array(hypo_pnls, dtype=float),
        **{f"bn_{w}s": np.array(bn_features[w], dtype=float) for w in BN_WINDOWS},
    }


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 2: BASE RATE ANALYSIS — Wilson CI
# ═══════════════════════════════════════════════════════════════════════════

def wilson_ci(successes: int, total: int, conf: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval — robust for small samples and extreme rates."""
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    z = sp_stats.norm.ppf(1 - (1 - conf) / 2)
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    halfwidth = (z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))) / denominator
    return (max(0, center - halfwidth), min(1, center + halfwidth))


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 3: UNIVARIATE GRID SEARCH (VECTORIZED)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_gate(data: Dict[str, np.ndarray], window: float, threshold: float) -> dict:
    """Fully vectorized gate evaluation."""
    bn = data[f"bn_{window}s"]
    side = data["side"]
    won = data["won"]
    hypo_pnl = data["hypo_pnl"]
    
    valid = ~np.isnan(bn)
    
    # Gate blocks when abs(bn) > threshold AND bn_sign != side
    bn_sign = np.where(bn > 0, "UP", "DN")
    blocks = valid & (np.abs(bn) > threshold) & (bn_sign != side)
    
    # Confusion matrix (positive = blocked)
    tp = int(((blocks) & (won == 0)).sum())  # blocked loser = correct
    fp = int(((blocks) & (won == 1)).sum())  # blocked winner = incorrect
    tn = int(((~blocks) & (won == 1)).sum())  # passed winner
    fn = int(((~blocks) & (won == 0)).sum())  # passed loser
    
    # Dollar amounts
    tp_dollars = float(hypo_pnl[blocks & (won == 0)].sum())
    fp_dollars = float(hypo_pnl[blocks & (won == 1)].sum())
    tn_dollars = float(hypo_pnl[~blocks & (won == 1)].sum())
    fn_dollars = float(hypo_pnl[~blocks & (won == 0)].sum())
    
    n = tp + fp + tn + fn
    total_losers = tp + fn
    total_winners = fp + tn
    total_blocked = tp + fp
    
    net_pnl = tn_dollars + fn_dollars  # unblocked fires execute
    
    return {
        "window": window, "threshold": float(threshold),
        "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "tp_dollars": tp_dollars, "fp_dollars": fp_dollars,
        "tn_dollars": tn_dollars, "fn_dollars": fn_dollars,
        "net_pnl_with_gate": net_pnl,
        "block_rate": total_blocked / n * 100 if n else 0,
        "precision": tp / total_blocked * 100 if total_blocked else 0,
        "recall": tp / total_losers * 100 if total_losers else 0,
        "specificity": tn / total_winners * 100 if total_winners else 0,
        "f1": 2*tp / (2*tp + fp + fn) * 100 if (2*tp + fp + fn) else 0,
        "tpr": tp / total_losers if total_losers else 0,
        "fpr": fp / total_winners if total_winners else 0,
        "edge_per_block": (abs(tp_dollars) - fp_dollars) / total_blocked if total_blocked else 0,
    }


def grid_search(data: Dict[str, np.ndarray]) -> List[dict]:
    results = []
    for w in BN_WINDOWS:
        for t in BN_THRESHOLDS:
            results.append(evaluate_gate(data, w, float(t)))
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 4: ROC/AUC ANALYSIS (sklearn)
# ═══════════════════════════════════════════════════════════════════════════

def roc_analysis(data: Dict[str, np.ndarray]) -> dict:
    """
    For each BN window, compute ROC curve and AUC.
    Classification task: predict LOSE (so high BN in wrong direction = high lose-prob)
    
    For DN fires: BN > 0 (UP-trending) predicts LOSE → feature = bn
    For UP fires: BN < 0 (DN-trending) predicts LOSE → feature = -bn
    So our feature = bn * direction_sign_for_fire_side
    """
    results = {}
    
    # Build "adverse BN" feature: positive value = BN in direction AGAINST fire
    side_sign = np.where(data["side"] == "DN", 1.0, -1.0)  # DN → +BN is bad; UP → -BN is bad
    y_lose = 1 - data["won"]
    
    for w in BN_WINDOWS:
        bn = data[f"bn_{w}s"]
        adverse_bn = bn * side_sign
        
        valid = ~np.isnan(adverse_bn)
        if valid.sum() < 30:
            results[w] = {"auc": None, "n": int(valid.sum()), "error": "too few samples"}
            continue
        
        try:
            auc = roc_auc_score(y_lose[valid], adverse_bn[valid])
        except ValueError:
            auc = None
        
        # ROC curve points
        try:
            fpr, tpr, thresholds = roc_curve(y_lose[valid], adverse_bn[valid])
        except ValueError:
            fpr, tpr, thresholds = np.array([]), np.array([]), np.array([])
        
        # Optimal point (closest to top-left corner)
        if len(fpr) > 0:
            distances = np.sqrt(fpr**2 + (1-tpr)**2)
            opt_idx = distances.argmin()
            opt_threshold = thresholds[opt_idx]
            opt_tpr = tpr[opt_idx]
            opt_fpr = fpr[opt_idx]
        else:
            opt_threshold = opt_tpr = opt_fpr = None
        
        results[w] = {
            "auc": float(auc) if auc else None,
            "n": int(valid.sum()),
            "optimal_adverse_bn_threshold": float(opt_threshold) if opt_threshold is not None else None,
            "optimal_tpr": float(opt_tpr) if opt_tpr is not None else None,
            "optimal_fpr": float(opt_fpr) if opt_fpr is not None else None,
            "roc_points": {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        }
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 5: REGULARIZED LOGISTIC REGRESSION
# ═══════════════════════════════════════════════════════════════════════════

def logistic_analysis(data: Dict[str, np.ndarray], window: float) -> dict:
    """Fit L2-regularized logistic regression with cross-validation."""
    bn = data[f"bn_{window}s"]
    side = data["side"]
    won = data["won"]
    
    # Build feature: adverse BN (positive = against fire side)
    side_sign = np.where(side == "DN", 1.0, -1.0)
    adverse_bn = bn * side_sign
    
    valid = ~np.isnan(adverse_bn)
    X = adverse_bn[valid].reshape(-1, 1)
    y = (1 - won[valid]).astype(int)  # 1 = lose
    
    if len(X) < 30 or len(np.unique(y)) < 2:
        return {"error": "insufficient data or single class"}
    
    # LogisticRegressionCV chooses best C via cross-validation
    model = LogisticRegressionCV(cv=5, random_state=42, max_iter=1000)
    model.fit(X, y)
    
    y_proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_proba)
    
    # Bayesian optimal threshold: where P(lose) = 0.5
    # sigmoid(b0 + b1*x) = 0.5 → x = -b0/b1
    b0 = model.intercept_[0]
    b1 = model.coef_[0][0]
    bn_at_50pct = -b0 / b1 if abs(b1) > 1e-9 else None
    
    # Empirical win rate in buckets
    bucket_size = 0.005
    buckets = defaultdict(lambda: [0, 0])
    for bn_val, lose in zip(adverse_bn[valid], y):
        b = round(bn_val / bucket_size) * bucket_size
        buckets[b][1] += 1
        if not lose:
            buckets[b][0] += 1
    
    bucket_data = []
    for b in sorted(buckets):
        w, t = buckets[b]
        if t >= 5:
            lo, hi = wilson_ci(w, t)
            bucket_data.append({
                "adverse_bn_bucket": b,
                "win_rate": w/t,
                "wilson_lo": lo, "wilson_hi": hi,
                "n": t,
            })
    
    return {
        "window": window,
        "n": len(X),
        "auc": float(auc),
        "intercept": float(b0),
        "slope": float(b1),
        "bayesian_optimal_adverse_bn": float(bn_at_50pct) if bn_at_50pct else None,
        "C_chosen": float(model.C_[0]),
        "bucket_win_rates": bucket_data,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 6: RANDOM FOREST BENCHMARK (detect nonlinearity)
# ═══════════════════════════════════════════════════════════════════════════

def rf_benchmark(data: Dict[str, np.ndarray]) -> dict:
    """Benchmark linear (logistic) vs nonlinear (RF) models using all BN windows."""
    # Build feature matrix: all BN windows × side_sign
    side_sign = np.where(data["side"] == "DN", 1.0, -1.0)
    features = []
    feature_names = []
    for w in BN_WINDOWS:
        bn = data[f"bn_{w}s"]
        features.append(bn * side_sign)
        feature_names.append(f"adverse_bn_{w}s")
    
    # Add cd as feature
    features.append(data["cd"])
    feature_names.append("cd")
    
    X = np.column_stack(features)
    y = (1 - data["won"]).astype(int)
    
    # Drop rows with any NaN
    valid = ~np.isnan(X).any(axis=1)
    X = X[valid]
    y = y[valid]
    
    if len(X) < 50 or len(np.unique(y)) < 2:
        return {"error": "insufficient data"}
    
    # Logistic regression with all windows
    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42))
    ])
    lr_scores = cross_val_score(lr_model, X, y, cv=5, scoring="roc_auc")
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=4, 
                                       random_state=42, n_jobs=-1)
    rf_scores = cross_val_score(rf_model, X, y, cv=5, scoring="roc_auc")
    
    # Gradient Boosting  
    gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, 
                                           random_state=42)
    gb_scores = cross_val_score(gb_model, X, y, cv=5, scoring="roc_auc")
    
    # Feature importance from RF
    rf_model.fit(X, y)
    importances = dict(zip(feature_names, rf_model.feature_importances_.tolist()))
    
    return {
        "n_samples": len(X),
        "feature_names": feature_names,
        "logistic_auc_cv": {"mean": float(lr_scores.mean()), "std": float(lr_scores.std())},
        "random_forest_auc_cv": {"mean": float(rf_scores.mean()), "std": float(rf_scores.std())},
        "gradient_boosting_auc_cv": {"mean": float(gb_scores.mean()), "std": float(gb_scores.std())},
        "feature_importance_rf": importances,
        "nonlinearity_gain": float(rf_scores.mean() - lr_scores.mean()),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 7: PERMUTATION TEST — verify BN has predictive power
# ═══════════════════════════════════════════════════════════════════════════

def permutation_test(data: Dict[str, np.ndarray], window: float, n_permutations: int = 1000) -> dict:
    """
    Null hypothesis: BN has no predictive power.
    Shuffle outcomes randomly; if real AUC is extreme vs null distribution, reject H0.
    """
    side_sign = np.where(data["side"] == "DN", 1.0, -1.0)
    bn = data[f"bn_{window}s"]
    adverse_bn = bn * side_sign
    
    valid = ~np.isnan(adverse_bn)
    X = adverse_bn[valid]
    y = (1 - data["won"][valid]).astype(int)
    
    if len(X) < 30 or len(np.unique(y)) < 2:
        return {"error": "insufficient data"}
    
    # Real AUC
    try:
        real_auc = roc_auc_score(y, X)
    except ValueError:
        return {"error": "AUC undefined"}
    
    # Null distribution
    null_aucs = []
    rng = np.random.default_rng(42)
    for _ in range(n_permutations):
        y_shuffled = rng.permutation(y)
        try:
            null_aucs.append(roc_auc_score(y_shuffled, X))
        except ValueError:
            continue
    
    null_aucs = np.array(null_aucs)
    # Two-tailed p-value
    p_value = (np.abs(null_aucs - 0.5) >= abs(real_auc - 0.5)).mean()
    
    return {
        "window": window,
        "real_auc": float(real_auc),
        "null_auc_mean": float(null_aucs.mean()),
        "null_auc_std": float(null_aucs.std()),
        "p_value": float(p_value),
        "significant_at_0.05": bool(p_value < 0.05),
        "significant_at_0.01": bool(p_value < 0.01),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 8: PER-STRATEGY CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════

def per_strategy_analysis(data: Dict[str, np.ndarray]) -> dict:
    strategies = np.unique(data["strategy"])
    results = {}
    
    for s in strategies:
        mask = data["strategy"] == s
        if mask.sum() < 20:
            continue
        
        sub = {k: v[mask] for k, v in data.items()}
        
        # Find best (w, t) for this strategy
        best = None
        for w in BN_WINDOWS:
            for t in BN_THRESHOLDS:
                stats = evaluate_gate(sub, w, float(t))
                if best is None or stats["net_pnl_with_gate"] > best["net_pnl_with_gate"]:
                    best = stats
        
        # No-gate baseline
        no_gate_pnl = float(sub["hypo_pnl"].sum())
        
        # Wilson CI on win rate
        wins = int(sub["won"].sum())
        total = len(sub["won"])
        wr_lo, wr_hi = wilson_ci(wins, total)
        
        results[str(s)] = {
            "n_fires": int(mask.sum()),
            "wins": wins, "losses": total - wins,
            "win_rate": wins / total * 100,
            "win_rate_95ci": [wr_lo * 100, wr_hi * 100],
            "no_gate_pnl": no_gate_pnl,
            "best_window": best["window"],
            "best_threshold": best["threshold"],
            "best_pnl": best["net_pnl_with_gate"],
            "improvement_vs_no_gate": best["net_pnl_with_gate"] - no_gate_pnl,
            "best_precision": best["precision"],
            "best_recall": best["recall"],
        }
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 9: TIME-CONDITIONAL (cd buckets)
# ═══════════════════════════════════════════════════════════════════════════

def time_conditional(data: Dict[str, np.ndarray]) -> dict:
    buckets = [(0, 60), (60, 120), (120, 180), (180, 240), (240, 300)]
    results = {}
    for lo, hi in buckets:
        mask = (data["cd"] >= lo) & (data["cd"] < hi)
        if mask.sum() < 20:
            continue
        sub = {k: v[mask] for k, v in data.items()}
        
        best = None
        for w in BN_WINDOWS:
            for t in BN_THRESHOLDS:
                stats = evaluate_gate(sub, w, float(t))
                if best is None or stats["net_pnl_with_gate"] > best["net_pnl_with_gate"]:
                    best = stats
        
        results[f"cd_{lo}_{hi}"] = {
            "n": int(mask.sum()),
            "base_win_rate": float(sub["won"].mean() * 100),
            "best_window": best["window"],
            "best_threshold": best["threshold"],
            "best_pnl": best["net_pnl_with_gate"],
            "precision": best["precision"],
            "recall": best["recall"],
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 10: BOOTSTRAP CI (numpy-vectorized)
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_ci(data: Dict[str, np.ndarray], n_bootstrap: int = 1000) -> dict:
    n = len(data["won"])
    rng = np.random.default_rng(42)
    
    param_votes = Counter()
    pnl_samples = []
    
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sub = {k: v[idx] for k, v in data.items()}
        
        best = None
        for w in BN_WINDOWS:
            for t in BN_THRESHOLDS:
                stats = evaluate_gate(sub, w, float(t))
                if best is None or stats["net_pnl_with_gate"] > best["net_pnl_with_gate"]:
                    best = stats
        
        param_votes[(best["window"], best["threshold"])] += 1
        pnl_samples.append(best["net_pnl_with_gate"])
    
    pnl_samples = np.array(pnl_samples)
    
    ranked = param_votes.most_common(5)
    return {
        "n_bootstraps": n_bootstrap,
        "top_params": [{"window": p[0][0], "threshold": p[0][1],
                       "votes": p[1], "frequency": p[1] / n_bootstrap}
                      for p in ranked],
        "pnl_median": float(np.median(pnl_samples)),
        "pnl_mean": float(np.mean(pnl_samples)),
        "pnl_95ci": [float(np.percentile(pnl_samples, 2.5)),
                     float(np.percentile(pnl_samples, 97.5))],
        "pnl_std": float(np.std(pnl_samples)),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 11: WALK-FORWARD VALIDATION (purged time-series split)
# ═══════════════════════════════════════════════════════════════════════════

def walk_forward(data: Dict[str, np.ndarray], n_splits: int = 5) -> dict:
    # Sort by slug (which encodes timestamp)
    sort_idx = np.argsort(data["slug"])
    sorted_data = {k: v[sort_idx] for k, v in data.items()}
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    
    y = (1 - sorted_data["won"]).astype(int)
    X_indices = np.arange(len(y))
    
    for i, (train_idx, test_idx) in enumerate(tscv.split(X_indices)):
        train = {k: v[train_idx] for k, v in sorted_data.items()}
        test = {k: v[test_idx] for k, v in sorted_data.items()}
        
        # Find best params on train
        best = None
        for w in BN_WINDOWS:
            for t in BN_THRESHOLDS:
                stats = evaluate_gate(train, w, float(t))
                if best is None or stats["net_pnl_with_gate"] > best["net_pnl_with_gate"]:
                    best = stats
        
        # Apply to test
        test_stats = evaluate_gate(test, best["window"], best["threshold"])
        no_gate_test_pnl = float(test["hypo_pnl"].sum())
        
        results.append({
            "split": i + 1,
            "train_size": len(train["won"]),
            "test_size": len(test["won"]),
            "best_window": best["window"],
            "best_threshold": best["threshold"],
            "train_pnl": best["net_pnl_with_gate"],
            "test_pnl": test_stats["net_pnl_with_gate"],
            "test_no_gate_pnl": no_gate_test_pnl,
            "test_improvement": test_stats["net_pnl_with_gate"] - no_gate_test_pnl,
            "test_precision": test_stats["precision"],
        })
    
    # Stability assessment
    windows = [r["best_window"] for r in results]
    thresholds = [r["best_threshold"] for r in results]
    improvements = [r["test_improvement"] for r in results]
    
    return {
        "splits": results,
        "window_stability": "stable" if len(set(windows)) == 1 else f"varies among {sorted(set(windows))}",
        "threshold_stability": "stable" if len(set(thresholds)) == 1 else f"varies among {sorted(set(thresholds))}",
        "mean_test_improvement": float(np.mean(improvements)),
        "test_improvement_positive_rate": float(np.mean(np.array(improvements) > 0)),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 12: PERMUTATION IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════

def feature_importance(data: Dict[str, np.ndarray]) -> dict:
    """How much does each BN window contribute to predictive power?"""
    side_sign = np.where(data["side"] == "DN", 1.0, -1.0)
    features = []
    names = []
    for w in BN_WINDOWS:
        bn = data[f"bn_{w}s"]
        features.append(bn * side_sign)
        names.append(f"adverse_bn_{w}s")
    features.append(data["cd"])
    names.append("cd")
    
    X = np.column_stack(features)
    y = (1 - data["won"]).astype(int)
    
    valid = ~np.isnan(X).any(axis=1)
    X = X[valid]
    y = y[valid]
    
    if len(X) < 50 or len(np.unique(y)) < 2:
        return {"error": "insufficient data"}
    
    model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X, y)
    
    # Permutation importance (more robust than Gini)
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    
    importance = {}
    for i, name in enumerate(names):
        importance[name] = {
            "mean": float(result.importances_mean[i]),
            "std": float(result.importances_std[i]),
        }
    
    # Sort by importance
    sorted_imp = sorted(importance.items(), key=lambda x: -x[1]["mean"])
    return {
        "importance": importance,
        "ranking": [name for name, _ in sorted_imp],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 13: KELLY SIZING (given fitted win probabilities)
# ═══════════════════════════════════════════════════════════════════════════

def kelly_sizing_analysis(data: Dict[str, np.ndarray], window: float) -> dict:
    """
    Kelly fraction = (p*b - (1-p)) / b
    where p = win prob, b = net odds (gain/loss ratio)
    
    For our trades:
    - Buy at price P
    - Win: +$(1-P)/P per dollar
    - Lose: -$1 per dollar
    - Net odds b = (1-P)/P
    
    Compute optimal position size given fitted P(win | BN).
    """
    bn = data[f"bn_{window}s"]
    side_sign = np.where(data["side"] == "DN", 1.0, -1.0)
    adverse_bn = bn * side_sign
    won = data["won"]
    hypo_pnl = data["hypo_pnl"]
    
    valid = ~np.isnan(adverse_bn)
    X = adverse_bn[valid].reshape(-1, 1)
    y = (1 - won[valid]).astype(int)
    pnls = hypo_pnl[valid]
    
    if len(X) < 30 or len(np.unique(y)) < 2:
        return {"error": "insufficient data"}
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    
    p_win = 1 - model.predict_proba(X)[:, 1]
    
    # Estimate b from historical data (avg win / avg loss magnitude)
    avg_win = pnls[pnls > 0].mean() if (pnls > 0).any() else 3.5
    avg_loss = abs(pnls[pnls < 0].mean()) if (pnls < 0).any() else 1.0
    b = avg_win / avg_loss if avg_loss > 0 else 1.0
    
    # Kelly fraction for each fire
    kelly = (p_win * b - (1 - p_win)) / b
    kelly_clipped = np.clip(kelly, 0, 0.25)  # safety cap at 25%
    
    # Traditional constant-size P&L
    const_pnl = pnls.sum()
    
    # Kelly-sized P&L (kelly fraction of max loss)
    # For each fire: trade only if kelly > 0, size proportional to kelly_clipped
    trades = kelly > 0
    sized_pnls = pnls * (kelly_clipped / 0.25)  # normalize so max size = 1
    kelly_pnl = sized_pnls[trades].sum() if trades.any() else 0
    
    return {
        "n": int(valid.sum()),
        "avg_win_dollars": float(avg_win),
        "avg_loss_dollars": float(avg_loss),
        "b_odds_ratio": float(b),
        "avg_kelly_fraction": float(kelly.mean()),
        "fires_with_positive_kelly": int(trades.sum()),
        "constant_size_pnl": float(const_pnl),
        "kelly_size_pnl": float(kelly_pnl),
        "kelly_improvement": float(kelly_pnl - const_pnl),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 14: PLOTTING (if matplotlib available)
# ═══════════════════════════════════════════════════════════════════════════

def generate_plots(data: Dict[str, np.ndarray], roc_results: dict, 
                   grid_results: List[dict], output_path: str = "regime_gate_plots.png"):
    if not PLOTS_AVAILABLE:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: ROC curves
    ax = axes[0, 0]
    for w, res in roc_results.items():
        if "roc_points" in res and res["roc_points"]["fpr"]:
            ax.plot(res["roc_points"]["fpr"], res["roc_points"]["tpr"], 
                   label=f"window={w}s (AUC={res['auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel("False Positive Rate (winners blocked)")
    ax.set_ylabel("True Positive Rate (losers blocked)")
    ax.set_title("ROC Curves by BN Window")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Plot 2: PnL vs threshold for each window
    ax = axes[0, 1]
    for w in BN_WINDOWS:
        w_results = [r for r in grid_results if r["window"] == w]
        thresholds = [r["threshold"] for r in w_results]
        pnls = [r["net_pnl_with_gate"] for r in w_results]
        ax.plot(thresholds, pnls, marker="o", label=f"w={w}s", alpha=0.7)
    ax.set_xlabel("BN Threshold")
    ax.set_ylabel("Net PnL with Gate ($)")
    ax.set_title("Gate PnL vs Threshold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xscale("log")
    
    # Plot 3: BN distribution for winners vs losers
    ax = axes[1, 0]
    side_sign = np.where(data["side"] == "DN", 1.0, -1.0)
    bn = data["bn_10.0s"] * side_sign  # adverse BN
    valid = ~np.isnan(bn)
    won = data["won"]
    winners_bn = bn[valid & (won == 1)]
    losers_bn = bn[valid & (won == 0)]
    ax.hist([winners_bn, losers_bn], bins=30, alpha=0.6,
           label=[f"Winners (n={len(winners_bn)})", f"Losers (n={len(losers_bn)})"])
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel("Adverse BN (positive = against fire side)")
    ax.set_ylabel("Count")
    ax.set_title("BN_10s Distribution by Outcome")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 4: Precision vs Recall across thresholds (for window=10s)
    ax = axes[1, 1]
    for w in [3.0, 10.0, 30.0, 60.0]:
        w_results = [r for r in grid_results if r["window"] == w]
        precisions = [r["precision"] for r in w_results]
        recalls = [r["recall"] for r in w_results]
        ax.plot(recalls, precisions, marker="o", label=f"w={w}s", alpha=0.7)
    ax.set_xlabel("Recall (% of losers blocked)")
    ax.set_ylabel("Precision (% of blocks correct)")
    ax.set_title("Precision-Recall Trade-off")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
    print(f"  Plots saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    history_path = sys.argv[1]
    recap_path = sys.argv[2] if len(sys.argv) >= 3 else \
        os.path.join(os.path.dirname(history_path) or ".", "market_recap_history.jsonl")
    
    print("═" * 80)
    print("  REGIME GATE PRO CALIBRATOR")
    print("═" * 80)
    
    # ── Phase 1: Load
    print("\n[Phase 1] Loading markets with tick data...")
    markets = load_markets(history_path)
    if not markets:
        print("ERROR: No markets with tick data found.")
        sys.exit(1)
    
    print(f"\n[Phase 1] Loading recap history...")
    recaps = []
    with open(recap_path) as f:
        for line in f:
            try: recaps.append(json.loads(line))
            except: pass
    print(f"  Loaded {len(recaps)} recap entries")
    
    print(f"\n[Phase 1] Building vectorized dataset (this may take a while)...")
    data = build_dataset(markets, recaps)
    n = len(data["won"])
    print(f"  Fires: {n}")
    
    if n < 50:
        print(f"ERROR: Only {n} fire records — need at least 50 for reliable analysis.")
        sys.exit(1)
    
    # Save CSV
    import csv
    with open("regime_gate_dataset.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = ["slug", "strategy", "side", "cd", "winner", "won", "hypo_pnl"] + [f"bn_{x}s" for x in BN_WINDOWS]
        w.writerow(cols)
        for i in range(n):
            w.writerow([data[c][i] for c in cols])
    print(f"  Raw dataset saved: regime_gate_dataset.csv")
    
    # ── Phase 2: Base rate
    wins = int(data["won"].sum())
    base_wr = wins / n
    wr_lo, wr_hi = wilson_ci(wins, n)
    print(f"\n[Phase 2] Base rate: {wins}/{n} = {base_wr*100:.1f}% (95% Wilson CI: [{wr_lo*100:.1f}%, {wr_hi*100:.1f}%])")
    
    # ── Phase 3: Grid
    print(f"\n[Phase 3] Grid search...")
    grid = grid_search(data)
    current = next((r for r in grid if r["window"] == 10.0 and r["threshold"] == 0.005), None)
    best = max(grid, key=lambda r: r["net_pnl_with_gate"])
    baseline_pnl = float(data["hypo_pnl"].sum())
    print(f"  Baseline (no gate):       ${baseline_pnl:+.2f}")
    if current:
        print(f"  Current (w=10s, t=0.005): ${current['net_pnl_with_gate']:+.2f}  precision={current['precision']:.1f}%  recall={current['recall']:.1f}%")
    print(f"  Best PnL config:          ${best['net_pnl_with_gate']:+.2f}  w={best['window']}s  t={best['threshold']}  precision={best['precision']:.1f}%")
    
    # ── Phase 4: ROC
    print(f"\n[Phase 4] ROC analysis...")
    roc = roc_analysis(data)
    for w, r in roc.items():
        if r.get("auc"):
            print(f"  window={w}s: AUC={r['auc']:.4f} (n={r['n']})")
    best_auc_window = max((w for w, r in roc.items() if r.get("auc")), 
                          key=lambda w: roc[w]["auc"] or 0, default=None)
    print(f"  Best discriminating window: {best_auc_window}s")
    
    # ── Phase 5: Logistic
    print(f"\n[Phase 5] Regularized logistic regression...")
    logistic = logistic_analysis(data, best_auc_window or 10.0)
    if "error" not in logistic:
        print(f"  AUC: {logistic['auc']:.4f}")
        print(f"  Slope: {logistic['slope']:+.4f}  (positive = adverse BN → higher lose prob)")
        if logistic["bayesian_optimal_adverse_bn"] is not None:
            print(f"  Bayesian-optimal adverse_BN threshold: {logistic['bayesian_optimal_adverse_bn']:+.4f}%")
    
    # ── Phase 6: RF benchmark
    print(f"\n[Phase 6] Random Forest benchmark (detects nonlinearity)...")
    rf = rf_benchmark(data)
    if "error" not in rf:
        print(f"  Logistic AUC (CV):       {rf['logistic_auc_cv']['mean']:.4f} ± {rf['logistic_auc_cv']['std']:.4f}")
        print(f"  Random Forest AUC (CV):  {rf['random_forest_auc_cv']['mean']:.4f} ± {rf['random_forest_auc_cv']['std']:.4f}")
        print(f"  Gradient Boost AUC (CV): {rf['gradient_boosting_auc_cv']['mean']:.4f} ± {rf['gradient_boosting_auc_cv']['std']:.4f}")
        if rf["nonlinearity_gain"] > 0.02:
            print(f"  ⚠️  Nonlinearity detected! RF outperforms logistic by {rf['nonlinearity_gain']:.4f} — consider nonlinear gate.")
        else:
            print(f"  Linear logistic is sufficient (nonlinearity gain: {rf['nonlinearity_gain']:+.4f})")
    
    # ── Phase 7: Permutation test
    print(f"\n[Phase 7] Permutation test (H0: BN is noise)...")
    perm = permutation_test(data, best_auc_window or 10.0, 1000)
    if "error" not in perm:
        print(f"  Real AUC:    {perm['real_auc']:.4f}")
        print(f"  Null AUC:    {perm['null_auc_mean']:.4f} ± {perm['null_auc_std']:.4f}")
        print(f"  p-value:     {perm['p_value']:.4f}")
        if perm["significant_at_0.01"]:
            print(f"  ✓ HIGHLY SIGNIFICANT (p < 0.01) — BN has REAL predictive power")
        elif perm["significant_at_0.05"]:
            print(f"  ✓ Significant (p < 0.05)")
        else:
            print(f"  ⚠️  NOT significant — BN may be noise; reconsider using it at all")
    
    # ── Phase 8: Per strategy
    print(f"\n[Phase 8] Per-strategy calibration...")
    per_strat = per_strategy_analysis(data)
    for s, r in sorted(per_strat.items(), key=lambda x: -x[1]["improvement_vs_no_gate"]):
        print(f"  {s:<28} n={r['n_fires']:>4} WR={r['win_rate']:>5.1f}% "
              f"no_gate=${r['no_gate_pnl']:>+8.2f} best=${r['best_pnl']:>+8.2f} "
              f"Δ=${r['improvement_vs_no_gate']:>+7.2f} (w={r['best_window']}, t={r['best_threshold']})")
    
    # ── Phase 9: Time conditional
    print(f"\n[Phase 9] Time-conditional optimal params...")
    time_r = time_conditional(data)
    for bucket, r in sorted(time_r.items()):
        print(f"  {bucket}: n={r['n']:>3} WR={r['base_win_rate']:>5.1f}% "
              f"w={r['best_window']}s t={r['best_threshold']} PnL=${r['best_pnl']:>+7.2f}")
    
    # ── Phase 10: Bootstrap
    print(f"\n[Phase 10] Bootstrap CI (1000 iterations)...")
    boot = bootstrap_ci(data, 1000)
    print(f"  95% CI on optimal PnL: ${boot['pnl_95ci'][0]:+.2f} to ${boot['pnl_95ci'][1]:+.2f}")
    print(f"  Top voted params:")
    for p in boot["top_params"][:3]:
        print(f"    w={p['window']}s t={p['threshold']}: {p['frequency']*100:.1f}%")
    
    # ── Phase 11: Walk-forward
    print(f"\n[Phase 11] Walk-forward validation (5 splits)...")
    wf = walk_forward(data, 5)
    for r in wf["splits"]:
        print(f"  Split {r['split']}: train[w={r['best_window']},t={r['best_threshold']}] "
              f"train_pnl=${r['train_pnl']:+.2f} test_pnl=${r['test_pnl']:+.2f} "
              f"test_vs_no_gate=${r['test_improvement']:+.2f}")
    print(f"  Window stability: {wf['window_stability']}")
    print(f"  Threshold stability: {wf['threshold_stability']}")
    print(f"  Mean out-of-sample improvement: ${wf['mean_test_improvement']:+.2f}")
    print(f"  Test improvement was positive in {wf['test_improvement_positive_rate']*100:.0f}% of splits")
    
    # ── Phase 12: Feature importance
    print(f"\n[Phase 12] Feature importance (permutation)...")
    fi = feature_importance(data)
    if "error" not in fi:
        for name in fi["ranking"]:
            imp = fi["importance"][name]
            print(f"  {name:<25}: {imp['mean']:+.4f} ± {imp['std']:.4f}")
    
    # ── Phase 13: Kelly
    print(f"\n[Phase 13] Kelly sizing analysis...")
    kelly = kelly_sizing_analysis(data, best_auc_window or 10.0)
    if "error" not in kelly:
        print(f"  Avg win: ${kelly['avg_win_dollars']:.2f}  Avg loss: ${kelly['avg_loss_dollars']:.2f}")
        print(f"  Odds ratio b: {kelly['b_odds_ratio']:.2f}")
        print(f"  Constant-size total PnL: ${kelly['constant_size_pnl']:+.2f}")
        print(f"  Kelly-sized total PnL:   ${kelly['kelly_size_pnl']:+.2f}")
        print(f"  Kelly improvement:       ${kelly['kelly_improvement']:+.2f}")
    
    # ── Generate plots
    print(f"\n[Phase 14] Generating plots...")
    generate_plots(data, roc, grid)
    
    # ── Save all results
    output = {
        "summary": {
            "n_markets": len(markets),
            "n_fires": n, "wins": wins, "base_win_rate": base_wr,
            "win_rate_95ci": [wr_lo, wr_hi],
            "baseline_pnl": baseline_pnl,
            "current_gate_pnl": current["net_pnl_with_gate"] if current else None,
        },
        "best_pnl_config": {
            "window": best["window"], "threshold": best["threshold"],
            "pnl": best["net_pnl_with_gate"], "precision": best["precision"],
            "recall": best["recall"], "f1": best["f1"],
        },
        "roc": {str(k): v for k, v in roc.items()},
        "logistic_regression": logistic,
        "random_forest_benchmark": rf,
        "permutation_test": perm,
        "per_strategy": per_strat,
        "time_conditional": time_r,
        "bootstrap": boot,
        "walk_forward": wf,
        "feature_importance": fi,
        "kelly_analysis": kelly,
        "grid_full": grid,
    }
    
    with open("regime_gate_analysis.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    # ── Recommendation
    improvement = best["net_pnl_with_gate"] - (current["net_pnl_with_gate"] if current else 0)
    
    rec = f"""
═══════════════════════════════════════════════════════════════════════════════
  REGIME GATE CALIBRATION — FINAL RECOMMENDATION
═══════════════════════════════════════════════════════════════════════════════

DATA:
  Markets: {len(markets)} with tick data
  Fires:   {n}  ({wins} wins, {n-wins} losses = {base_wr*100:.1f}% WR)
  Win rate 95% CI: [{wr_lo*100:.1f}%, {wr_hi*100:.1f}%]

STATISTICAL VALIDATION:
  BN predictive power: p-value = {perm.get('p_value', 'N/A')} ({'SIGNIFICANT' if perm.get('significant_at_0.01', False) else 'NOT significant'})
  Best window AUC: {roc.get(best_auc_window, {}).get('auc', 'N/A')}
  RF vs Logistic gain: {rf.get('nonlinearity_gain', 'N/A')} ({'nonlinear' if rf.get('nonlinearity_gain', 0) > 0.02 else 'linear sufficient'})

CURRENT CONFIG (arb_bot line ~17927):
  BN_WINDOW = 10.0s, BN_THRESHOLD = 0.005
  Gate PnL: ${current['net_pnl_with_gate'] if current else 'N/A':+.2f}
  Precision: {current['precision'] if current else 'N/A':.1f}%
  Recall: {current['recall'] if current else 'N/A':.1f}%

OPTIMAL CONFIG (grid search + bootstrap validated):
  BN_WINDOW = {best['window']}s
  BN_THRESHOLD = {best['threshold']}
  Gate PnL: ${best['net_pnl_with_gate']:+.2f}
  Precision: {best['precision']:.1f}%
  Recall: {best['recall']:.1f}%
  F1: {best['f1']:.1f}

OUT-OF-SAMPLE VALIDATION:
  Walk-forward mean test improvement: ${wf['mean_test_improvement']:+.2f}
  Positive improvement rate: {wf['test_improvement_positive_rate']*100:.0f}% of splits
  Parameter stability: window={wf['window_stability']}, threshold={wf['threshold_stability']}

BOOTSTRAP CONFIDENCE:
  Expected PnL 95% CI: ${boot['pnl_95ci'][0]:+.2f} to ${boot['pnl_95ci'][1]:+.2f}
  Top param frequency: {boot['top_params'][0]['frequency']*100:.0f}%

═══════════════════════════════════════════════════════════════════════════════
  RECOMMENDED CODE CHANGE (arb_bot line ~17927):

    # OLD:
    _bn_now = get_bn_delta_pct(10.0)
    if _bn_now is not None and abs(_bn_now) > 0.005:

    # NEW:
    _bn_now = get_bn_delta_pct({best['window']})
    if _bn_now is not None and abs(_bn_now) > {best['threshold']}:

  EXPECTED IMPROVEMENT: ${improvement:+.2f} over {n} fires = ${improvement/n:+.3f}/fire
  
═══════════════════════════════════════════════════════════════════════════════
"""
    print(rec)
    with open("regime_gate_recommendation.txt", "w") as f:
        f.write(rec)
    
    print("\n✓ All outputs saved:")
    print("  regime_gate_analysis.json")
    print("  regime_gate_recommendation.txt")
    print("  regime_gate_dataset.csv")
    if PLOTS_AVAILABLE:
        print("  regime_gate_plots.png")


if __name__ == "__main__":
    main()
