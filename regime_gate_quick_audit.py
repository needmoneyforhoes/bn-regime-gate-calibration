#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  REGIME GATE QUICK AUDIT — fast diagnostic
═══════════════════════════════════════════════════════════════════════════════

QUICK audit of regime gate performance using recap_history only (no tick data
required). Gives a sanity check of the 39% false-positive rate claim before
running the full calibrator.

USAGE:
    python regime_gate_quick_audit.py market_recap_history.jsonl

OUTPUT:
    Prints summary statistics about current regime gate's actual performance.
"""

import json
import sys
import re
from collections import Counter, defaultdict

RULE_F_DN = {
    "cl_gated_cheap", "cl_tight", "d3_wb_filter",
    "depth_collapse_mid", "depth_collapse_mid_v2", "depth_collapse_mid_v3",
    "depth_stable_mid", "late_dip_recovery", "t4_late", "t4_sr",
    "t4_sr_relaxed", "v5_shadow_b4_moderate", "v5_shadow_b5_mod_plus",
    "depth_surge", "deep_dip_scout",
}

def main():
    if len(sys.argv) < 2:
        print("Usage: python regime_gate_quick_audit.py market_recap_history.jsonl")
        sys.exit(1)
    
    recaps = []
    with open(sys.argv[1]) as f:
        for line in f:
            try:
                recaps.append(json.loads(line))
            except:
                pass
    
    # Look for fires with regime_demote in extras
    # Pattern: extras.get("regime_demote") == "bn_side_mismatch (BN→X vs fire→Y)"
    
    total_fires = 0
    regime_blocks = []
    
    for r in recaps:
        winner = r.get("winner")
        if winner not in ("UP", "DN"):
            continue
        
        for fire in r.get("fires", []):
            if fire["strategy"] not in RULE_F_DN:
                continue
            if fire["side"] != "DN":
                continue
            
            total_fires += 1
            
            extras = fire.get("extras", {})
            regime = extras.get("regime_demote") if isinstance(extras, dict) else None
            
            if regime and "bn_side_mismatch" in regime:
                # Extract BN side from string like "bn_side_mismatch (BN→UP vs fire→DN)"
                m = re.search(r"BN→(UP|DN)", regime)
                bn_side = m.group(1) if m else None
                
                # Was the block correct? (fire_side != winner means fire would have lost → correct block)
                correct_block = (fire["side"] != winner)
                
                regime_blocks.append({
                    "slug": r["slug"],
                    "strategy": fire["strategy"],
                    "side": fire["side"],
                    "winner": winner,
                    "hypo_pnl": fire.get("hypo_pnl", 0),
                    "correct": correct_block,
                })
    
    print(f"Total Rule F DN fires: {total_fires}")
    print(f"Regime gate blocks: {len(regime_blocks)}")
    
    if not regime_blocks:
        print("\nNo regime_demote records found in extras.")
        print("The recap history may not preserve this field.")
        print("Consider using the deep calibrator which reads tick data directly.")
        return
    
    correct = sum(1 for b in regime_blocks if b["correct"])
    incorrect = len(regime_blocks) - correct
    
    saved = sum(b["hypo_pnl"] for b in regime_blocks if not b["correct"])  # what we avoided
    lost_edge = sum(b["hypo_pnl"] for b in regime_blocks if b["correct"])  # what we gave up? 
    # wait — let me reconsider
    # If block was CORRECT: fire would have lost → hypo_pnl < 0 → we SAVED money
    # If block was INCORRECT: fire would have won → hypo_pnl > 0 → we LOST edge
    
    money_saved = abs(sum(b["hypo_pnl"] for b in regime_blocks if b["correct"] and b["hypo_pnl"] < 0))
    money_lost = sum(b["hypo_pnl"] for b in regime_blocks if not b["correct"] and b["hypo_pnl"] > 0)
    
    print(f"\n  Correct blocks (blocked losers):    {correct} ({correct/len(regime_blocks)*100:.1f}%)")
    print(f"  Incorrect blocks (blocked winners): {incorrect} ({incorrect/len(regime_blocks)*100:.1f}%)")
    print(f"  Money saved from correct blocks:    ${money_saved:+.2f}")
    print(f"  Money lost from incorrect blocks:   ${money_lost:+.2f}")
    print(f"  Net gate edge: ${money_saved - money_lost:+.2f}")
    
    if money_lost > money_saved:
        print(f"\n  ⚠️  Gate is NET NEGATIVE — costing ${money_lost - money_saved:.2f} more than it saves")
    else:
        print(f"\n  ✓  Gate is NET POSITIVE — saving ${money_saved - money_lost:.2f} net")
    
    # Per-market breakdown
    by_market = defaultdict(lambda: {"correct": 0, "incorrect": 0})
    for b in regime_blocks:
        if b["correct"]:
            by_market[b["slug"]]["correct"] += 1
        else:
            by_market[b["slug"]]["incorrect"] += 1
    
    worst = sorted(by_market.items(), key=lambda x: -x[1]["incorrect"])[:10]
    print(f"\n  Top 10 markets where gate blocked most winners:")
    for slug, counts in worst:
        print(f"    {slug}: {counts['incorrect']} winners blocked, {counts['correct']} losers blocked")

if __name__ == "__main__":
    main()
