"""
strong_up_gate_analysis.py — Statistical analysis for Part 15: DN suppression gate

HYPOTHESIS:
  When BN is strongly positive AND crowd strongly backs UP at T-200,
  DN fires have significantly lower win rate → gate should block them.

  This is the exact symmetric mirror of Part 14 (which blocks UP fires
  when BN strongly negative at T-200).

METHODOLOGY (rigorous, no future bias):
  1. Simulate gate trigger from tick data (BN > threshold AND crowd > threshold)
  2. For triggered markets: measure DN fire win rate vs baseline
  3. Bonferroni correction across all candidate thresholds
  4. Walk-forward OOS split (60% IS / 40% OOS)
  5. Permutation test (10k iterations)
  6. Kelly criterion on OOS data
  7. Binomial vs breakeven

Run: python3 strong_up_gate_analysis.py
"""

import json, os, sys, numpy as np
from pathlib import Path
from scipy import stats as scipy_stats
from math import ceil

MARKET_HISTORY_PATH = os.path.expanduser('~/polymarket-bot/market_history.jsonl')
RECAP_HISTORY_PATH  = os.path.expanduser('~/polymarket-bot/market_recap_history.jsonl')
N_PERMUTATIONS      = 10_000
IS_SPLIT            = 0.60
ALPHA               = 0.05
N_CANDIDATES        = 8   # number of threshold candidates tested (Bonferroni)
ALPHA_BONF          = ALPHA / N_CANDIDATES

def get_bn_at_t200(ticks, cols):
    """BN delta_pct at first tick with cd <= 200. No future bias — same as T-200 checkpoint."""
    ci = {c:i for i,c in enumerate(cols)}
    cd_i = ci.get('cd', 0)
    bn_i = ci.get('bn_delta_pct', 12)
    for t in ticks:
        cd = t[cd_i]
        if cd and cd <= 200:
            return t[bn_i] or 0
    return None

def get_crowd_at_t200(ticks, cols):
    """crowd_conviction and crowd_side at first tick with cd <= 200."""
    ci = {c:i for i,c in enumerate(cols)}
    cd_i = ci.get('cd', 0)
    cv_i = ci.get('crowd_conviction', 20)
    cs_i = ci.get('crowd_side', 19)
    for t in ticks:
        cd = t[cd_i]
        if cd and cd <= 200:
            cv = t[cv_i] if cv_i < len(t) else 0
            cs = t[cs_i] if cs_i < len(t) else None
            return cv or 0, cs
    return 0, None

def get_up_ask_at_t200(ticks, cols):
    """UP ask at cd=200."""
    ci = {c:i for i,c in enumerate(cols)}
    cd_i = ci.get('cd', 0)
    ua_i = ci.get('up_ask', 2)
    for t in ticks:
        cd = t[cd_i]
        if cd and cd <= 200:
            return t[ua_i] or 0
    return 0

def pnl(fires): return sum(f.get('hypo_pnl', 0) for f in fires)

# ── LOAD ──────────────────────────────────────────────────────────────────
print("Loading...", flush=True)
mkt_hist = {}
with open(MARKET_HISTORY_PATH) as f:
    for line in f:
        try:
            d = json.loads(line)
            if d.get('slug') and d.get('ticks') and d.get('tick_columns'):
                mkt_hist[d['slug']] = d
        except: pass

recap = {}
with open(RECAP_HISTORY_PATH) as f:
    for line in f:
        try:
            d = json.loads(line)
            if d.get('slug') and d.get('fires') and d.get('winner') in ('UP','DN'):
                recap[d['slug']] = d
        except: pass

common = sorted(set(mkt_hist) & set(recap))
print(f"  {len(common)} markets")

# ── COLLECT FEATURES + DN FIRE OUTCOMES ───────────────────────────────────
print("Computing features...", flush=True)
markets = []
for slug in common:
    mkt   = mkt_hist[slug]
    rec   = recap[slug]
    ticks = mkt['ticks']
    cols  = mkt['tick_columns']

    bn200  = get_bn_at_t200(ticks, cols)
    cv200, cs200 = get_crowd_at_t200(ticks, cols)
    ua200  = get_up_ask_at_t200(ticks, cols)
    winner = rec['winner']
    fires  = rec['fires']

    # All free DN fires (what the bot actually traded)
    # Excludes pre_gate_held and opp_gate_blocked to avoid contamination
    dn_fires_free = [
        f for f in fires
        if f.get('side') == 'DN'
        and not f.get('pre_gate_held')
        and not f.get('opp_gate_blocked')
        and not f.get('opp_gate_would_block')
    ]

    dn_pnl   = pnl(dn_fires_free)
    n_dn     = len(dn_fires_free)
    dn_wins  = sum(1 for f in dn_fires_free if f.get('hypo_pnl', 0) > 0)

    # crowd_up_conv: high crowd conviction FOR UP (signal: block DN)
    crowd_up_conv = cv200 if cs200 == 'UP' else -(cv200 or 0)  # signed

    markets.append({
        'slug': slug, 'winner': winner,
        'bn200': bn200 or 0,
        'cv200': cv200 or 0,
        'cs200': cs200,
        'ua200': ua200,
        'crowd_up_conv': crowd_up_conv,
        'n_dn': n_dn,
        'dn_pnl': dn_pnl,
        'dn_wins': dn_wins,
    })

# Only markets with DN fires
with_dn = [m for m in markets if m['n_dn'] > 0]
print(f"  {len(markets)} total markets, {len(with_dn)} with free DN fires")

# Baseline DN fire win rate
total_dn_fires = sum(m['n_dn']   for m in with_dn)
total_dn_wins  = sum(m['dn_wins'] for m in with_dn)
baseline_prec  = total_dn_wins / total_dn_fires if total_dn_fires > 0 else 0
baseline_pnl   = sum(m['dn_pnl'] for m in with_dn)
avg_dn_pnl_per_mkt = baseline_pnl / len(with_dn)

print(f"\n{'═'*65}")
print(f"BASELINE DN FIRE STATISTICS")
print(f"{'═'*65}")
print(f"  Markets with free DN fires: {len(with_dn)}")
print(f"  Total DN fires: {total_dn_fires}")
print(f"  DN fire win rate: {baseline_prec*100:.1f}%")
print(f"  Total DN PnL: {baseline_pnl:+.2f}")
print(f"  Avg DN PnL/market: {avg_dn_pnl_per_mkt:+.3f}")

# ── CANDIDATE GATE SIGNALS ────────────────────────────────────────────────
print(f"\n{'═'*65}")
print(f"CANDIDATE SIGNALS (at T-200, no future bias)")
print(f"Block DN fires when signal exceeds threshold")
print(f"Bonferroni α = {ALPHA}/{N_CANDIDATES} = {ALPHA_BONF:.4f}")
print(f"{'═'*65}")

candidates = [
    # (name, feature_fn, direction, thresholds_to_test)
    ('bn200_pos',    lambda m: m['bn200'],          +1,   [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]),
    ('crowd_up',     lambda m: m['crowd_up_conv'],  +1,   [0.50, 0.60, 0.65, 0.70, 0.75, 0.80]),
    ('bn_and_crowd', lambda m: m['bn200'] * (m['crowd_up_conv'] if m['crowd_up_conv'] > 0 else 0),
                               +1,   [0.01, 0.02, 0.04, 0.06]),
]

best_overall = None

for cname, feat_fn, direction, thresholds in candidates:
    print(f"\n  Signal: {cname}")
    vals = [(m, feat_fn(m) * direction) for m in with_dn]

    print(f"  {'Threshold':>10} {'n_trig':>7} {'dn_wins%':>9} {'vs_base':>9} "
          f"{'EV_change':>10} {'p_raw':>8}")
    best_thresh = None
    best_ev = -999

    for thresh in thresholds:
        triggered = [m for m, v in vals if v >= thresh]
        not_trig  = [m for m, v in vals if v < thresh]
        if len(triggered) < 5: continue

        trig_n_fires = sum(m['n_dn'] for m in triggered)
        trig_wins    = sum(m['dn_wins'] for m in triggered)
        trig_prec    = trig_wins / trig_n_fires if trig_n_fires > 0 else 0
        trig_pnl     = sum(m['dn_pnl'] for m in triggered)

        # EV of blocking DN fires in triggered markets
        # Positive = blocking helps (DN fires were net negative → blocking saves money)
        ev_of_blocking = -trig_pnl  # negative pnl blocked = positive savings
        ev_per_mkt = ev_of_blocking / len(triggered) if triggered else 0

        # t-test: is dn_pnl/market in triggered significantly different from non-triggered?
        trig_pnl_per = [m['dn_pnl'] for m in triggered]
        not_pnl_per  = [m['dn_pnl'] for m in not_trig] if not_trig else [0]
        _, p_raw = scipy_stats.ttest_ind(trig_pnl_per, not_pnl_per, equal_var=False) \
            if len(trig_pnl_per) > 1 and len(not_pnl_per) > 1 else (0, 1.0)

        prec_delta = trig_prec - baseline_prec
        sig = '***' if p_raw * N_CANDIDATES < ALPHA else ''

        print(f"  {thresh:>10.3f} {len(triggered):>7} {trig_prec*100:>8.1f}% "
              f"{prec_delta*100:>+8.1f}% {ev_per_mkt:>+10.3f} {p_raw:>8.4f} {sig}")

        if ev_of_blocking > best_ev and len(triggered) >= 10:
            best_ev = ev_of_blocking
            best_thresh = thresh
            best_cand = (cname, feat_fn, direction, thresh, triggered, not_trig)

    if best_thresh and best_overall is None:
        best_overall = best_cand

# ── OOS VALIDATION OF BEST CANDIDATE ─────────────────────────────────────
if best_overall:
    cname, feat_fn, direction, thresh, triggered_all, not_trig_all = best_overall
    print(f"\n{'═'*65}")
    print(f"OOS VALIDATION: {cname} ≥ {thresh}")
    print(f"{'═'*65}")

    split_i     = int(len(with_dn) * IS_SPLIT)
    IS_markets  = with_dn[:split_i]
    OOS_markets = with_dn[split_i:]

    def gate_triggered(m): return feat_fn(m) * direction >= thresh

    IS_trig  = [m for m in IS_markets  if gate_triggered(m)]
    OOS_trig = [m for m in OOS_markets if gate_triggered(m)]

    for label, subset, trig in [('IS ', IS_markets,  IS_trig),
                                  ('OOS', OOS_markets, OOS_trig)]:
        if not trig: continue
        pnl_trig     = sum(m['dn_pnl'] for m in trig)
        ev_blocking  = -pnl_trig
        n_fires_trig = sum(m['n_dn'] for m in trig)
        wins_trig    = sum(m['dn_wins'] for m in trig)
        prec_trig    = wins_trig / n_fires_trig if n_fires_trig > 0 else 0
        print(f"  {label}: n_triggered={len(trig):3}  fires={n_fires_trig:4}  "
              f"prec={prec_trig*100:.1f}%  "
              f"EV_blocking={ev_blocking:+.2f}  per_mkt={ev_blocking/len(trig):+.3f}")

    # Permutation test on OOS
    oos_deltas = [-m['dn_pnl'] if gate_triggered(m) else 0 for m in OOS_markets]
    obs_mean = np.mean(oos_deltas)
    np.random.seed(42)
    perm = [np.mean(np.random.choice([-1,1], len(oos_deltas)) * oos_deltas)
            for _ in range(N_PERMUTATIONS)]
    p_perm = np.mean(np.abs(perm) >= abs(obs_mean))
    print(f"\n  OOS Permutation p = {p_perm:.4f}  {'*** sig' if p_perm<ALPHA else 'ns'}")

    # Binomial test on OOS: is DN win rate significantly LOWER when gate triggers?
    oos_trig    = [m for m in OOS_markets if gate_triggered(m)]
    if oos_trig:
        oos_fires   = sum(m['n_dn'] for m in oos_trig)
        oos_wins    = sum(m['dn_wins'] for m in oos_trig)
        oos_prec    = oos_wins / oos_fires if oos_fires > 0 else 0
        result_binom = scipy_stats.binomtest(oos_wins, oos_fires,
                                              baseline_prec, alternative='less')
        print(f"  OOS binomial test (H1: prec < baseline {baseline_prec*100:.1f}%):")
        print(f"    OOS prec = {oos_prec*100:.1f}%  p = {result_binom.pvalue:.4f}  "
              f"{'*** sig' if result_binom.pvalue < ALPHA else 'ns'}")

        # Kelly
        p_k = 1 - oos_prec  # probability DN fire LOSES (gate is correct to block)
        q_k = oos_prec       # probability DN fire wins (gate wrong to block)
        oos_fired_pnl = [f['hypo_pnl'] for m in oos_trig
                         for f in recap[m['slug']]['fires']
                         if f.get('side')=='DN' and not f.get('pre_gate_held')
                         and not f.get('opp_gate_blocked')]
        if oos_fired_pnl:
            avg_loss = abs(np.mean([p for p in oos_fired_pnl if p < 0] or [-1]))
            avg_win  = abs(np.mean([p for p in oos_fired_pnl if p > 0] or [1]))
            b_k = avg_loss / avg_win  # payoff ratio for blocking
            f_star = p_k - q_k / b_k
            print(f"  Kelly f* = {p_k:.3f} - {q_k:.3f}/{b_k:.3f} = {f_star:.4f}  "
                  f"{'(block EV-positive)' if f_star > 0 else '(block EV-negative)'}")

print(f"\n{'═'*65}")
print("CONCLUSION")
print(f"{'═'*65}")
print(f"""
The question: should we add a gate that blocks DN fires when
BN is strongly positive and/or crowd strongly backs UP at T-200?

Two rigorous requirements to implement:
  1. Bonferroni-corrected signal (p_bonf < {ALPHA_BONF:.4f})
  2. OOS permutation p < {ALPHA} AND OOS binomial p < {ALPHA}

If neither candidate reaches significance: KEEP CURRENT BEHAVIOR.
The loss in market 1777551300 is the cost of uncertainty — strategies
fire DN because at fire time they see a valid DN entry signal.
Blocking DN fires without confirmed statistical edge = adding noise.

First deploy the race-condition fix (sha1: 283bc77...) and collect
50+ more markets with correct gate data before re-evaluating.
""")
