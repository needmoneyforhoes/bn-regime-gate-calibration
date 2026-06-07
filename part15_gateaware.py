"""
part15_gateaware.py — Part 15 analysis counting ONLY fires that pass all existing gates.

Correct fire inclusion:
  - Must be DN side
  - Must fire AFTER gate checkpoint (cd < gate_cd)
  - NOT pre_gate_held (held by ts_drop/ts_rec stage-1)
  - NOT opp_gate_blocked (blocked by live gate: dn_live etc.)
  - NOT opp_gate_would_block (shadow-tracked as would-block)
  → This is exactly what Part 15 would see in real-time

Also shows:
  - How many DN fires dn_live already catches in high-BN markets
  - Incremental value of Part 15 ON TOP of existing gates
  - Per-winner breakdown: UP win vs DN win triggered markets
"""

import json, os, numpy as np
from scipy import stats as scipy_stats
from scipy.stats import beta as beta_dist

MARKET_HISTORY_PATH = os.path.expanduser('~/polymarket-bot/market_history.jsonl')
RECAP_HISTORY_PATH  = os.path.expanduser('~/polymarket-bot/market_recap_history.jsonl')
IS_SPLIT = 0.60

def get_bn_at_cd(ticks, cols, target_cd):
    ci = {c:i for i,c in enumerate(cols)}
    cd_i = ci.get('cd',0); bn_i = ci.get('bn_delta_pct',12)
    for t in ticks:
        cd = t[cd_i]
        if cd is not None and cd <= target_cd:
            v = t[bn_i]; return v if v is not None else None
    return None

def get_gate_aware_dn_fires(fires, gate_cd):
    """
    DN fires that Part 15 would see and block:
      - DN side
      - fire_cd < gate_cd (fires AFTER the gate triggers)
      - passed ALL existing gates (not pre_gate_held, not blocked, not would_block)
    """
    passed, blocked_by_others = [], []
    for f in fires:
        if f.get('side') != 'DN': continue
        fire_cd = f.get('cd') or 0
        if fire_cd >= gate_cd: continue   # fired before gate checkpoint

        if (f.get('pre_gate_held') or
                f.get('opp_gate_blocked') or
                f.get('opp_gate_would_block')):
            blocked_by_others.append(f.get('hypo_pnl', 0))
        else:
            passed.append(f.get('hypo_pnl', 0))
    return passed, blocked_by_others

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
markets = []
for slug in common:
    mkt = mkt_hist[slug]; rec = recap[slug]
    markets.append({'slug': slug, 'winner': rec['winner'],
                    'mkt': mkt, 'fires': rec['fires']})

split_i   = int(len(markets) * IS_SPLIT)
IS_slugs  = {m['slug'] for m in markets[:split_i]}
OOS_slugs = {m['slug'] for m in markets[split_i:]}
print(f"  {len(markets)} markets  IS={split_i}  OOS={len(markets)-split_i}")

# ── BASELINE: all gate-aware DN fires across all markets ──────────────────
BASELINE_CD = 999   # all fires, no gate_cd filter
all_passed = []
for m in markets:
    passed, _ = get_gate_aware_dn_fires(m['fires'], BASELINE_CD)
    all_passed.extend(passed)

baseline_n    = len(all_passed)
baseline_wins = sum(1 for p in all_passed if p > 0)
baseline_prec = baseline_wins / baseline_n if baseline_n else 0
baseline_avg  = np.mean(all_passed) if all_passed else 0

print(f"\nBaseline (gate-filtered DN fires, no cd restriction):")
print(f"  {baseline_wins}/{baseline_n} wins = {baseline_prec*100:.1f}%  avg={baseline_avg:+.4f}")

# ── SWEEP CANDIDATES ──────────────────────────────────────────────────────
for gate_cd, bn_thresh in [(290, 0.04), (290, 0.02), (300, 0.04)]:
    print(f"\n{'═'*70}")
    print(f"CANDIDATE: cd={gate_cd}  BN≥+{bn_thresh*100:.0f}%")
    print(f"{'═'*70}")

    # Collect ALL triggered markets with full breakdown
    trig_up_wins = []   # UP win markets where gate is CORRECT
    trig_dn_wins = []   # DN win markets where gate is WRONG
    not_triggered = []

    total_blocked_by_others = 0  # already caught by dn_live etc.

    for m in markets:
        bn = get_bn_at_cd(m['mkt']['ticks'], m['mkt']['tick_columns'], gate_cd)
        passed, blocked = get_gate_aware_dn_fires(m['fires'], gate_cd)

        rec = {
            'slug': m['slug'], 'winner': m['winner'],
            'fires_passed': passed,
            'fires_blocked_by_others': blocked,
            'n_passed': len(passed),
            'wins_passed': sum(1 for p in passed if p > 0),
            'pnl_passed': sum(passed),
            'is_oos': m['slug'] in OOS_slugs,
        }

        if (bn or 0) >= bn_thresh:
            total_blocked_by_others += len(blocked)
            if m['winner'] == 'UP':
                trig_up_wins.append(rec)
            else:
                trig_dn_wins.append(rec)
        else:
            not_triggered.append(rec)

    n_trig    = len(trig_up_wins) + len(trig_dn_wins)
    prec_mkt  = len(trig_up_wins) / n_trig if n_trig else 0

    # Aggregate fires
    up_fires = [p for r in trig_up_wins for p in r['fires_passed']]
    dn_fires = [p for r in trig_dn_wins for p in r['fires_passed']]
    all_trig_fires = up_fires + dn_fires
    not_fires = [p for r in not_triggered for p in r['fires_passed']]

    up_saves = -sum(up_fires)    # blocking UP-win markets saves losses
    dn_costs =  sum(dn_fires)    # blocking DN-win markets misses profits
    net_ev   = up_saves - dn_costs

    print(f"\n  ── MARKET-LEVEL BREAKDOWN ──")
    print(f"  Triggered: {n_trig} markets  "
          f"UP_win={len(trig_up_wins)} ({prec_mkt*100:.1f}%) CORRECT  |  "
          f"DN_win={len(trig_dn_wins)} ({(1-prec_mkt)*100:.1f}%) WRONG")
    print(f"  Not triggered: {len(not_triggered)} markets")
    print(f"  Already blocked by other gates (dn_live etc): {total_blocked_by_others} fires "
          f"(Part 15 incremental only)")

    print(f"\n  ── FIRE-LEVEL DETAIL (gate-filtered, fires at cd<{gate_cd}) ──")
    print(f"  UP win triggered → {len(up_fires):3} fires  "
          f"win%={sum(1 for p in up_fires if p>0)/len(up_fires)*100:.1f}%  "
          f"PnL={sum(up_fires):+.2f}  SAVES={up_saves:+.2f}")
    if dn_fires:
        print(f"  DN win triggered → {len(dn_fires):3} fires  "
              f"win%={sum(1 for p in dn_fires if p>0)/len(dn_fires)*100:.1f}%  "
              f"PnL={sum(dn_fires):+.2f}  COSTS={dn_costs:+.2f}")
    else:
        print(f"  DN win triggered →   0 fires  (no blockable DN fires in these markets)")
    print(f"  Not triggered   → {len(not_fires):3} fires  "
          f"win%={sum(1 for p in not_fires if p>0)/len(not_fires)*100:.1f}%  "
          f"(unchanged, gate silent)")

    print(f"\n  ── NET EV ──")
    print(f"  Saves from correct blocks: {up_saves:+.2f}")
    print(f"  Costs from wrong blocks:   {dn_costs:+.2f}")
    print(f"  Net gate EV:               {net_ev:+.2f}  ({'✅ POSITIVE' if net_ev>0 else '❌ NEGATIVE'})")
    print(f"  Per triggered market:      {net_ev/n_trig if n_trig else 0:+.3f}")

    # Breakeven precision
    if dn_fires and up_fires:
        avg_save_per_fire = up_saves / len(up_fires) if up_fires else 0
        avg_cost_per_fire = dn_costs / len(dn_fires) if dn_fires else 0
        # Breakeven: prec * avg_save = (1-prec) * avg_cost_per_mkt
        saves_per_mkt = np.mean([-r['pnl_passed'] for r in trig_up_wins]) if trig_up_wins else 0
        costs_per_mkt = np.mean([r['pnl_passed']  for r in trig_dn_wins]) if trig_dn_wins else 0
        if saves_per_mkt + costs_per_mkt > 0:
            breakeven = costs_per_mkt / (saves_per_mkt + costs_per_mkt)
            print(f"  Breakeven precision:       {breakeven*100:.1f}%  "
                  f"(observed: {prec_mkt*100:.1f}%  "
                  f"margin: {(prec_mkt-breakeven)*100:+.1f}pp)")

    # OOS breakdown
    oos_up = [r for r in trig_up_wins if r['is_oos']]
    oos_dn = [r for r in trig_dn_wins if r['is_oos']]
    oos_up_fires = [p for r in oos_up for p in r['fires_passed']]
    oos_dn_fires = [p for r in oos_dn for p in r['fires_passed']]
    oos_saves = -sum(oos_up_fires)
    oos_costs =  sum(oos_dn_fires)
    oos_net   = oos_saves - oos_costs
    oos_n     = len(oos_up) + len(oos_dn)

    print(f"\n  ── OOS VALIDATION (last 40%) ──")
    print(f"  UP={len(oos_up)} ({len(oos_up)/oos_n*100:.0f}%)  DN={len(oos_dn)} ({len(oos_dn)/oos_n*100:.0f}%)")
    print(f"  OOS saves={oos_saves:+.2f}  costs={oos_costs:+.2f}  "
          f"net={oos_net:+.2f}  per_mkt={oos_net/oos_n if oos_n else 0:+.3f}")
    print(f"  IS/OOS consistent: {'✅' if (net_ev>0)==(oos_net>0) else '❌ REVERSAL'}")

    # Statistical tests on gate-filtered fires
    n_f = len(all_trig_fires)
    k_f = sum(1 for p in all_trig_fires if p > 0)
    if n_f > 0:
        r_b = scipy_stats.binomtest(k_f, n_f, baseline_prec, alternative='less')
        a_p = k_f+1; b_p = n_f-k_f+1
        p_bayes = beta_dist.cdf(baseline_prec, a_p, b_p)
        ci_lo, ci_hi = beta_dist.ppf([0.025, 0.975], a_p, b_p)
        bf = 1.0 / beta_dist.pdf(baseline_prec, a_p, b_p)

        print(f"\n  ── STATISTICS (gate-filtered fires) ──")
        print(f"  Fires: {k_f}/{n_f} wins = {k_f/n_f*100:.1f}% vs {baseline_prec*100:.1f}% baseline")
        print(f"  Binomial p:  {r_b.pvalue:.8f}  {'✅' if r_b.pvalue<0.05 else '❌'}")
        print(f"  Bayesian:    P(rate<baseline)={p_bayes:.6f}  BF={bf:.0f}  "
              f"CI=[{ci_lo*100:.1f}%,{ci_hi*100:.1f}%]")
        print(f"  Kelly f*:    ", end='')
        losses = [abs(p) for p in all_trig_fires if p < 0]
        wins_v = [abs(p) for p in all_trig_fires if p > 0]
        if losses and wins_v:
            p_l = 1 - k_f/n_f; q_w = k_f/n_f; b_k = np.mean(losses)/np.mean(wins_v)
            f_star = p_l - q_w/b_k
            print(f"f*={f_star:.4f}  {'✅ block EV+' if f_star>0 else '❌ block EV-'}")
        else:
            print("insufficient data")

print(f"\n{'═'*70}")
print("VERDICT")
print(f"{'═'*70}")
print("""
Key question: does the gate lose money in DN win triggered markets?
→ See 'DN win triggered fires' above.
→ If DN wins have few/no blockable fires after gate_cd: gate is safe.
→ If net EV is positive: gate is correct even with some wrong blocks.

The gate can never be 100% right without future bias.
The statistical question is: are correct blocks EV > wrong blocks EV?
""")
