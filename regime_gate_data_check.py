"""
Quick check: what data do we have to work with?
"""
import json
import sys

if len(sys.argv) < 2:
    print("Usage: python regime_gate_data_check.py path/to/market_history.jsonl")
    sys.exit(1)

history_path = sys.argv[1]

print(f"Checking: {history_path}")
print("="*80)

n_records = 0
n_with_ticks = 0
n_with_winner = 0
tick_sample = None
n_total_ticks = 0
max_ticks = 0

with open(history_path) as f:
    for line in f:
        n_records += 1
        try:
            m = json.loads(line)
        except:
            continue
        
        if m.get("winner") in ("UP", "DN"):
            n_with_winner += 1
        
        if "ticks" in m and m["ticks"]:
            n_with_ticks += 1
            n_ticks = len(m["ticks"])
            n_total_ticks += n_ticks
            max_ticks = max(max_ticks, n_ticks)
            if tick_sample is None:
                tick_sample = m
        
        # Sample first 5 to show structure
        if n_records <= 3:
            print(f"\nRecord {n_records}:")
            print(f"  slug: {m.get('slug')}")
            print(f"  winner: {m.get('winner')}")
            print(f"  top-level keys: {list(m.keys())[:20]}")
            if "ticks" in m:
                print(f"  #ticks: {len(m['ticks'])}")
                if m.get("tick_columns"):
                    print(f"  tick_columns: {m['tick_columns'][:10]}...")
            if "fires" in m and m["fires"]:
                print(f"  #fires: {len(m['fires'])}")
                print(f"  first fire keys: {list(m['fires'][0].keys())}")

print()
print("="*80)
print(f"SUMMARY:")
print(f"  Total records:          {n_records}")
print(f"  Records with winner:    {n_with_winner}")
print(f"  Records with tick data: {n_with_ticks}")
print(f"  Total ticks available:  {n_total_ticks:,}")
print(f"  Max ticks per market:   {max_ticks:,}")
print(f"  Avg ticks per market:   {n_total_ticks // max(n_with_ticks, 1):,}")

if tick_sample and tick_sample.get("tick_columns"):
    print(f"\n  Available tick columns: {tick_sample['tick_columns']}")
    
    # Check for BN data
    if "bn_price" in tick_sample["tick_columns"]:
        print(f"\n  ✓ bn_price available — calibrator can reconstruct BN_delta at any cd")
    else:
        print(f"\n  ✗ bn_price NOT in tick_columns — calibrator cannot run")

if n_with_ticks < 20:
    print(f"\n  ⚠️  Only {n_with_ticks} markets have tick data.")
    print(f"      Need at least 20 for reliable calibration.")
elif n_with_ticks < 100:
    print(f"\n  ⚠️  {n_with_ticks} markets with tick data — acceptable but not ideal.")
else:
    print(f"\n  ✓ {n_with_ticks} markets with tick data — EXCELLENT for calibration!")
