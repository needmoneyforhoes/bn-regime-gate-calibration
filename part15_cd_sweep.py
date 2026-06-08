"""
part15_cd_sweep.py — Find optimal CD checkpoint for Part 15 DN suppression gate.

Tests BN threshold at every CD from 300→150 to find earliest reliable signal.
Full Bonferroni correction across ALL checkpoint×threshold combinations.

Run: python3 part15_cd_sweep.py
"""

import json, os, sys, numpy as np
from pathlib import Path
from scipy import stats as scipy_stats

MARKET_HISTORY_PATH = os.path.expanduser('~/polymarket-bot/market_history.jsonl')
RECAP_HISTORY_PATH  = os.path.expanduser('~/polymarket-bot/market_recap_history.jsonl')

# ── CONFIG ─────────────────────────────────────────────────────────────────
CD_CHECKPOINTS = list(range(300, 145, -10))   # 300, 290, 280, ... 150
BN_THRESHOLDS  = [0.020, 0.040, 0.060, 0.080, 0.100, 0.120]
N_TESTS        = len(CD_CHECKPOINTS) * len(BN_THRESHOLDS)   # full Bonferroni
ALPHA          = 0.05
ALPHA_BONF     = ALPHA / N_TESTS
IS_SPLIT       = 0.60
N_PERMUTATIONS = 10_000
MIN_N_TRIGGER  = 10   # minimum triggered markets for a result to be valid

print(f"Testing {len(CD_CHECKPOINTS)} checkpoints × {len(BN_THRESHOLDS)} thresholds = {N_TESTS} tests")
print(f"Bonferroni α = {ALPHA}/{N_TESTS} = {ALPHA_BONF:.5f}")
print()

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

# ── PRECOMPUTE: BN at each checkpoint for every market ────────────────────
print("Precomputing BN snapshots...", flush=True)

def get_bn_at_cd(ticks, cols, target_cd):
    """BN delta_pct at the FIRST tick where cd ≤ target_cd."""
    ci = {c:i for i,c in enumerate(cols)}
    cd_i = ci.get('cd', 0)
    bn_i = ci.get('bn_delta_pct', 12)
    for t in ticks:
        cd = t[cd_i]
        if cd is not None and cd <= target_cd:
            val = t[bn_i]
            return val if val is not None else None
    return None

def get_free_dn_pnl(fires):
    """Sum of hypo_pnl for free DN fires and count of wins."""
    total_pnl, n_fires, n_wins = 0, 0, 0
    for f in fires:
        if (f.get('side') == 'DN'
                and not f.get('pre_gate_held')
                and not f.get('opp_gate_blocked')
                and not f.get('opp_gate_would_block')):
            hypo = f.get('hypo_pnl', 0)
            total_pnl += hypo
            n_fires   += 1
            if hypo > 0: n_wins += 1
    return total_pnl, n_fires, n_wins

def get_dn_fires_after_cd(fires, gate_cd):
    """DN free fires that fired AFTER the gate checkpoint (cd < gate_cd)."""
    total_pnl, n_fires = 0, 0
    for f in fires:
        fire_cd = f.get('cd') or 0
        if (f.get('side') == 'DN'
                and fire_cd < gate_cd          # fired after gate
                and not f.get('pre_gate_held')
                and not f.get('opp_gate_blocked')
                and not f.get('opp_gate_would_block')):
            total_pnl += f.get('hypo_pnl', 0)
            n_fires   += 1
    return total_pnl, n_fires

# Build market feature table
markets = []
for slug in common:
    mkt    = mkt_hist[slug]
    rec    = recap[slug]
    ticks  = mkt['ticks']
    cols   = mkt['tick_columns']
    fires  = rec['fires']
    winner = rec['winner']

    # BN at each checkpoint
    bn_at = {}
    for cd in CD_CHECKPOINTS:
        bn_at[cd] = get_bn_at_cd(ticks, cols, cd)

    dn_pnl, n_dn, n_dn_wins = get_free_dn_pnl(fires)
    if n_dn == 0:
        continue   # no free DN fires — skip

    markets.append({
        'slug': slug, 'winner': winner,
        'bn_at': bn_at,
        'dn_pnl': dn_pnl, 'n_dn': n_dn, 'n_dn_wins': n_dn_wins,
        'fires': fires,
    })

print(f"  {len(markets)} markets with free DN fires")

# Baseline
total_fires = sum(m['n_dn'] for m in markets)
total_wins  = sum(m['n_dn_wins'] for m in markets)
baseline    = total_wins / total_fires if total_fires else 0
print(f"  Baseline DN win rate: {baseline*100:.1f}%  ({total_wins}/{total_fires} fires)")

# IS / OOS split
split_i    = int(len(markets) * IS_SPLIT)
IS_mkts    = markets[:split_i]
OOS_mkts   = markets[split_i:]

# ── SWEEP ─────────────────────────────────────────────────────────────────
print(f"\n{'═'*90}")
print(f"FULL SWEEP: cd=300→150 × BN thresholds  (Bonferroni α={ALPHA_BONF:.5f})")
print(f"{'═'*90}")
print(f"{'CD':>4} {'BN≥':>6} {'n_all':>6} {'prec%':>6} {'EV/mkt':>8} "
      f"{'p_raw':>8} {'p_bonf':>8} {'OOS_n':>6} {'OOS_ev':>8} {'sig'}")
print("-"*90)

results = []

for target_cd in CD_CHECKPOINTS:
    for bn_thresh in BN_THRESHOLDS:

        triggered     = [m for m in markets if (m['bn_at'].get(target_cd) or 0) >= bn_thresh]
        not_triggered = [m for m in markets if m not in triggered]

        if len(triggered) < MIN_N_TRIGGER:
            continue

        # DN fires AFTER the gate (these are the ones we'd actually block)
        trig_pnl_after = sum(
            get_dn_fires_after_cd(m['fires'], target_cd)[0]
            for m in triggered
        )
        trig_fires_after = sum(
            get_dn_fires_after_cd(m['fires'], target_cd)[1]
            for m in triggered
        )
        # DN fire win rate in triggered (for fires after gate)
        trig_wins_after = sum(
            sum(1 for f in m['fires']
                if f.get('side')=='DN' and (f.get('cd') or 0) < target_cd
                and not f.get('pre_gate_held') and not f.get('opp_gate_blocked')
                and f.get('hypo_pnl', 0) > 0)
            for m in triggered
        )
        trig_prec_after = trig_wins_after / trig_fires_after if trig_fires_after > 0 else 0

        ev_blocking = -trig_pnl_after   # negative PnL blocked = positive savings
        ev_per_mkt  = ev_blocking / len(triggered) if triggered else 0

        # t-test: EV/mkt of blocking in triggered vs not triggered
        trig_ev_per   = [-get_dn_fires_after_cd(m['fires'], target_cd)[0] for m in triggered]
        not_ev_per    = [-get_dn_fires_after_cd(m['fires'], target_cd)[0] for m in not_triggered]
        _, p_raw = scipy_stats.ttest_ind(trig_ev_per, not_ev_per, equal_var=False) \
            if len(trig_ev_per) > 1 and len(not_ev_per) > 1 else (0, 1.0)
        p_bonf = min(p_raw * N_TESTS, 1.0)
        sig = '✅***' if p_bonf < ALPHA_BONF else ('⚠️' if p_raw < 0.05 else '')

        # OOS
        oos_trig = [m for m in OOS_mkts if (m['bn_at'].get(target_cd) or 0) >= bn_thresh]
        oos_ev   = -sum(get_dn_fires_after_cd(m['fires'], target_cd)[0] for m in oos_trig)
        oos_ev_mkt = oos_ev / len(oos_trig) if oos_trig else 0

        results.append({
            'cd': target_cd, 'thresh': bn_thresh,
            'n': len(triggered), 'prec': trig_prec_after,
            'ev_mkt': ev_per_mkt, 'p_raw': p_raw, 'p_bonf': p_bonf,
            'oos_n': len(oos_trig), 'oos_ev_mkt': oos_ev_mkt,
            'sig': p_bonf < ALPHA_BONF,
            'ev_blocking': ev_blocking,
        })

        print(f"{target_cd:>4} {bn_thresh:>6.3f} {len(triggered):>6} "
              f"{trig_prec_after*100:>5.1f}% {ev_per_mkt:>+8.3f} "
              f"{p_raw:>8.4f} {p_bonf:>8.4f} "
              f"{len(oos_trig):>6} {oos_ev_mkt:>+8.3f} {sig}")

# ── SIGNIFICANT RESULTS ────────────────────────────────────────────────────
sig_results = [r for r in results if r['sig']]
print()
print(f"{'═'*90}")
print(f"BONFERRONI-SIGNIFICANT RESULTS (p_bonf < {ALPHA_BONF:.5f})")
print(f"{'═'*90}")

if not sig_results:
    print("  No Bonferroni-significant results found.")
    # Show best anyway
    best = min(results, key=lambda r: r['p_bonf'])
    print(f"  Best candidate: cd={best['cd']} BN≥{best['thresh']:.3f}  "
          f"p_bonf={best['p_bonf']:.4f}  OOS EV={best['oos_ev_mkt']:+.3f}/mkt  n={best['n']}")
else:
    for r in sorted(sig_results, key=lambda x: x['p_bonf']):
        print(f"  ✅ cd={r['cd']:>3} BN≥{r['thresh']:.3f}  n={r['n']:>3}  "
              f"prec={r['prec']*100:.1f}%  EV={r['ev_mkt']:>+7.3f}/mkt  "
              f"p_bonf={r['p_bonf']:.5f}  OOS_n={r['oos_n']:>2} OOS_EV={r['oos_ev_mkt']:>+7.3f}/mkt")

# ── OPTIMAL CD ANALYSIS ───────────────────────────────────────────────────
print()
print(f"{'═'*90}")
print("OPTIMAL CD CHECKPOINT — best OOS EV among significant results")
print(f"{'═'*90}")

valid = [r for r in results if r['oos_n'] >= 5 and r['p_bonf'] < ALPHA_BONF]
if valid:
    best_oos = max(valid, key=lambda r: r['oos_ev_mkt'])
    print(f"\n  BEST: cd={best_oos['cd']} BN≥{best_oos['thresh']:.3f}")
    print(f"    IS:  n={best_oos['n']:3}  EV/mkt={best_oos['ev_mkt']:+.3f}")
    print(f"    OOS: n={best_oos['oos_n']:3}  EV/mkt={best_oos['oos_ev_mkt']:+.3f}")
    print(f"    p_bonf={best_oos['p_bonf']:.5f}")

    # OOS permutation test on best
    print(f"\n  Running OOS permutation test (n={N_PERMUTATIONS})...", flush=True)
    cd_best = best_oos['cd']; thresh_best = best_oos['thresh']
    oos_trig_best = [m for m in OOS_mkts if (m['bn_at'].get(cd_best) or 0) >= thresh_best]
    oos_deltas = [
        -get_dn_fires_after_cd(m['fires'], cd_best)[0]
        if (m['bn_at'].get(cd_best) or 0) >= thresh_best else 0
        for m in OOS_mkts
    ]
    obs_mean = np.mean(oos_deltas)
    np.random.seed(42)
    perm = [np.mean(np.random.choice([-1,1], len(oos_deltas)) * oos_deltas)
            for _ in range(N_PERMUTATIONS)]
    p_perm = np.mean(np.abs(perm) >= abs(obs_mean))
    print(f"  OOS Permutation p = {p_perm:.4f}  {'✅ sig' if p_perm<0.05 else '❌ ns'}")

    # Binomial test
    oos_fires_best = sum(get_dn_fires_after_cd(m['fires'], cd_best)[1] for m in oos_trig_best)
    oos_wins_best  = sum(
        sum(1 for f in m['fires']
            if f.get('side')=='DN' and (f.get('cd') or 0) < cd_best
            and not f.get('pre_gate_held') and not f.get('opp_gate_blocked')
            and f.get('hypo_pnl',0) > 0)
        for m in oos_trig_best
    )
    if oos_fires_best > 0:
        oos_prec_best = oos_wins_best / oos_fires_best
        r_b = scipy_stats.binomtest(oos_wins_best, oos_fires_best,
                                     baseline, alternative='less')
        print(f"  OOS binomial (prec < baseline {baseline*100:.1f}%): "
              f"prec={oos_prec_best*100:.1f}%  p={r_b.pvalue:.6f}  "
              f"{'✅ sig' if r_b.pvalue<0.05 else '❌ ns'}")

        # Kelly
        p_loss = 1 - oos_prec_best
        avg_pnl_vals = [f.get('hypo_pnl',0) for m in oos_trig_best
                        for f in m['fires']
                        if f.get('side')=='DN' and (f.get('cd') or 0) < cd_best
                        and not f.get('pre_gate_held') and not f.get('opp_gate_blocked')]
        losses = [abs(p) for p in avg_pnl_vals if p < 0]
        wins   = [abs(p) for p in avg_pnl_vals if p > 0]
        if losses and wins:
            avg_loss = np.mean(losses); avg_win = np.mean(wins)
            b = avg_loss / avg_win
            f_star = p_loss - oos_prec_best / b
            print(f"  Kelly f* = {p_loss:.3f} - {oos_prec_best:.3f}/{b:.3f} = {f_star:.4f}  "
                  f"{'✅ block EV+' if f_star>0 else '❌ block EV-'}")

    # Market 1777551300 impact at optimal CD
    print(f"\n  Impact on market 1777551300 at cd={cd_best} BN≥{thresh_best:.3f}:")
    with open('./data/market_history.jsonl') as fh:
        for line in fh:
            d = json.loads(line)
            if d.get('slug','').endswith('1777551300'):
                bn_at_opt = get_bn_at_cd(d['ticks'], d['tick_columns'], cd_best)
                print(f"    BN at cd={cd_best}: {bn_at_opt:.4f}  "
                      f"{'→ WOULD BLOCK' if bn_at_opt and bn_at_opt >= thresh_best else '→ would not trigger'}")
                break

# ── SUMMARY TABLE ─────────────────────────────────────────────────────────
print()
print(f"{'═'*90}")
print("EARLY-TRIGGER ADVANTAGE: does earlier CD improve OOS EV?")
print(f"{'═'*90}")
print("(BN ≥ 0.080 at various CDs, for comparison)")
print(f"{'CD':>4} {'n_all':>6} {'EV/mkt':>8} {'p_bonf':>8} {'OOS_n':>6} {'OOS_EV':>8}")
for r in [x for x in results if abs(x['thresh']-0.080) < 0.001]:
    print(f"{r['cd']:>4} {r['n']:>6} {r['ev_mkt']:>+8.3f} {r['p_bonf']:>8.4f} "
          f"{r['oos_n']:>6} {r['oos_ev_mkt']:>+8.3f} "
          f"{'✅' if r['p_bonf']<ALPHA_BONF else ''}")
