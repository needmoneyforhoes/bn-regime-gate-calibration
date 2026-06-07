#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shadow_gates.py  —  LOG-ONLY candidate gates for validation during Round 2

Adds TWO new gate evaluators that run in parallel with the live Gate 1+2:

  Gate 3 (shadow): Strong-BN-Against filter
      Rule: If |bn at T-120| >= 0.03, fire side must match BN direction.
      Otherwise block (not fire, just log the block).

  Gate 4 (shadow): Late BN flip detector
      Rule: Track BN at every tick after T-120. If sign flips (|bn|>0.01 both
      sides of the flip), mark market as "late flip detected".
      When a market with late-flip settles, log whether positions opposite to
      the late BN sign would have benefited from early exit.

NEITHER GATE AFFECTS LIVE TRADING. They only LOG decisions to market recap so
we can validate their performance on Round 2 holdout data.

IMPORTANT: pre-registration integrity. The live bot still uses:
  - Existing Gate 1 (CL) unchanged
  - Existing Gate 2 (BN deadzone 0.005) unchanged
  - No live-fire filtering changes

When Round 2 resolves and we know which gate candidate wins, we promote the
shadow gates to live-active. This file stays here as reference.

INTEGRATION
-----------
To patch into the bot, add two call sites in arb_bot_strategiesAt122M.py:

1. Inside the per-fire code path (where a strategy decision is made):

     from shadow_gates import evaluate_shadow_gate_3
     shadow_g3_decision = evaluate_shadow_gate_3(
         fire_side=side,                     # "UP" or "DN"
         bn_at_t120=bn_at_fire_time,         # from current BN tracking
         threshold=0.03,
     )
     # Record in the fire dict:
     fire_record["shadow_g3_pass"] = shadow_g3_decision.passed
     fire_record["shadow_g3_reason"] = shadow_g3_decision.reason
     # Do NOT change any actual trading decision based on this.

2. Inside the tick-processing loop (where each tick of BN comes in):

     from shadow_gates import LateFlipTracker
     # One instance per market:
     flip_tracker = LateFlipTracker()  # at market start
     flip_tracker.ingest(cd=tick_cd, bn=tick_bn_delta_pct)  # every tick
     # At market end:
     market_record["shadow_g4_late_flip"] = flip_tracker.late_flip_detected
     market_record["shadow_g4_early_bn"] = flip_tracker.early_bn
     market_record["shadow_g4_late_bn"] = flip_tracker.late_bn
     market_record["shadow_g4_flip_time_cd"] = flip_tracker.flip_time_cd

USAGE VERIFICATION
------------------
python3 shadow_gates.py --self-test

This runs unit tests on synthetic fire data to verify:
- Gate 3 blocks fires against strong BN signal
- Gate 3 allows fires aligned with BN or when BN is weak
- LateFlipTracker correctly detects late flips
- LateFlipTracker doesn't false-alarm on noise
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — locked thresholds
# ═══════════════════════════════════════════════════════════════════════════

# Gate 3: Strong-BN-Against filter
GATE3_STRONG_BN_THRESHOLD = 0.03   # |bn| above this requires directional agreement
GATE3_WEAK_BN_IGNORE      = 0.005  # |bn| below this: signal too weak, don't filter

# Gate 4: Late flip detection
GATE4_EARLY_CD           = 120.0  # signal established by this cd
GATE4_LATE_CD            = 30.0   # flip monitored until this cd (keep buffer before settle)
GATE4_FLIP_MAGNITUDE     = 0.010  # |bn| must exceed this on BOTH sides of the flip

# ═══════════════════════════════════════════════════════════════════════════
#  GATE 3: Strong-BN-Against filter (called per fire)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Gate3Decision:
    passed:   bool
    reason:   str        # "ok" | "weak_bn_passthrough" | "no_bn" | "blocked_against_strong_bn"
    bn_used:  Optional[float]
    threshold: float

def evaluate_shadow_gate_3(
    fire_side: str,
    bn_at_t120: Optional[float],
    threshold: float = GATE3_STRONG_BN_THRESHOLD,
) -> Gate3Decision:
    """Pure function. No side effects. Returns would-have-been decision.

    Args:
        fire_side: "UP" or "DN" — what side the strategy decided to fire
        bn_at_t120: BN delta percentage at T-120s (or nearest available snapshot)
        threshold: magnitude above which fire MUST align with BN

    Returns:
        Gate3Decision with passed=True if fire would have been allowed.
    """
    if bn_at_t120 is None:
        return Gate3Decision(passed=True, reason="no_bn", bn_used=None, threshold=threshold)

    if abs(bn_at_t120) < GATE3_WEAK_BN_IGNORE:
        return Gate3Decision(passed=True, reason="weak_bn_passthrough",
                             bn_used=bn_at_t120, threshold=threshold)

    if abs(bn_at_t120) < threshold:
        # BN in "medium" range — existing Gate 2 handles; Gate 3 passes through
        return Gate3Decision(passed=True, reason="medium_bn_passthrough",
                             bn_used=bn_at_t120, threshold=threshold)

    # Strong BN: must match fire side
    bn_says = "UP" if bn_at_t120 > 0 else "DN"
    if bn_says != fire_side:
        return Gate3Decision(passed=False, reason="blocked_against_strong_bn",
                             bn_used=bn_at_t120, threshold=threshold)

    return Gate3Decision(passed=True, reason="ok_aligned_with_strong_bn",
                         bn_used=bn_at_t120, threshold=threshold)

# ═══════════════════════════════════════════════════════════════════════════
#  GATE 4: Late BN flip tracker (stateful per market)
# ═══════════════════════════════════════════════════════════════════════════

class LateFlipTracker:
    """Tracks BN sign evolution through a single market.

    Call ingest(cd, bn) on every tick. At market end, inspect:
        late_flip_detected:  bool — True if BN sign flipped between early_cd and late_cd
        early_bn:            float — BN at GATE4_EARLY_CD (cd=120 nearest)
        late_bn:             float — BN at GATE4_LATE_CD (cd=30 nearest, before settle)
        flip_time_cd:        float — cd at which the flip first happened, or None
    """

    def __init__(
        self,
        early_cd: float = GATE4_EARLY_CD,
        late_cd: float = GATE4_LATE_CD,
        flip_mag: float = GATE4_FLIP_MAGNITUDE,
    ):
        self.early_cd = early_cd
        self.late_cd  = late_cd
        self.flip_mag = flip_mag

        self.early_bn: Optional[float] = None
        self.early_cd_actual: Optional[float] = None
        self.late_bn: Optional[float] = None
        self.late_cd_actual: Optional[float] = None

        self.flip_time_cd: Optional[float] = None
        self._last_significant_sign: Optional[int] = None  # +1 or -1

    def ingest(self, cd: float, bn: Optional[float]) -> None:
        """Feed one tick's cd+BN."""
        if bn is None:
            return

        # Capture early BN — closest tick with cd >= early_cd
        if cd >= self.early_cd:
            # Always update to the most recent tick with cd >= early_cd
            self.early_bn = bn
            self.early_cd_actual = cd

        # Capture late BN — closest tick with cd <= late_cd
        if cd <= self.late_cd and self.late_bn is None:
            self.late_bn = bn
            self.late_cd_actual = cd

        # Track sign flips between early and late window only
        if self.late_cd < cd < self.early_cd and abs(bn) >= self.flip_mag:
            cur_sign = 1 if bn > 0 else -1
            if self._last_significant_sign is not None and cur_sign != self._last_significant_sign:
                if self.flip_time_cd is None:
                    self.flip_time_cd = cd
            self._last_significant_sign = cur_sign

    @property
    def late_flip_detected(self) -> bool:
        """True if BN sign meaningfully flipped between early_cd and late_cd.

        Requires BOTH early_bn and late_bn to have |bn| >= flip_mag (0.010).
        This ensures the flip is real directional commitment on both sides,
        not noise crossing zero. If either side is sub-threshold, it's noise.
        """
        if self.early_bn is None or self.late_bn is None:
            return False
        if abs(self.early_bn) < self.flip_mag or abs(self.late_bn) < self.flip_mag:
            return False
        return (self.early_bn > 0) != (self.late_bn > 0)

    def summary(self) -> dict:
        return dict(
            late_flip_detected=self.late_flip_detected,
            early_bn=self.early_bn,
            early_cd_actual=self.early_cd_actual,
            late_bn=self.late_bn,
            late_cd_actual=self.late_cd_actual,
            flip_time_cd=self.flip_time_cd,
        )

# ═══════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════════

def _self_test():
    print("=" * 70)
    print("  shadow_gates.py  —  Unit tests")
    print("=" * 70)

    tests_passed = 0
    tests_failed = 0

    def check(label: str, got, want) -> None:
        nonlocal tests_passed, tests_failed
        if got == want:
            print(f"  ✓ {label}")
            tests_passed += 1
        else:
            print(f"  ✗ {label}  got={got!r}  want={want!r}")
            tests_failed += 1

    # ── Gate 3 tests ──
    print("\n  Gate 3 (Strong-BN-Against):")

    # Strong UP BN, fire UP → pass
    d = evaluate_shadow_gate_3("UP", bn_at_t120=0.05)
    check("strong UP BN + fire UP → pass", d.passed, True)
    check("  reason = aligned", d.reason, "ok_aligned_with_strong_bn")

    # Strong UP BN, fire DN → BLOCK
    d = evaluate_shadow_gate_3("DN", bn_at_t120=0.05)
    check("strong UP BN + fire DN → BLOCK", d.passed, False)
    check("  reason = blocked", d.reason, "blocked_against_strong_bn")

    # Strong DN BN, fire DN → pass
    d = evaluate_shadow_gate_3("DN", bn_at_t120=-0.05)
    check("strong DN BN + fire DN → pass", d.passed, True)

    # Strong DN BN, fire UP → BLOCK
    d = evaluate_shadow_gate_3("UP", bn_at_t120=-0.05)
    check("strong DN BN + fire UP → BLOCK", d.passed, False)

    # Weak BN (below passthrough) → pass regardless
    d = evaluate_shadow_gate_3("UP", bn_at_t120=0.002)
    check("weak BN + fire UP → pass", d.passed, True)
    d = evaluate_shadow_gate_3("DN", bn_at_t120=0.002)
    check("weak BN + fire DN → pass", d.passed, True)

    # Medium BN (between 0.005 and 0.03) → pass, handled by existing Gate 2
    d = evaluate_shadow_gate_3("DN", bn_at_t120=0.02)
    check("medium BN + fire DN → pass", d.passed, True)
    check("  reason = medium passthrough", d.reason, "medium_bn_passthrough")

    # No BN data → pass (conservative)
    d = evaluate_shadow_gate_3("UP", bn_at_t120=None)
    check("no BN data → pass", d.passed, True)

    # ── Gate 4 tests ──
    print("\n  Gate 4 (Late BN Flip Tracker):")

    # Clean flip: BN was +0.05 at T-120, -0.05 at T-30 → detected
    tracker = LateFlipTracker()
    # Simulate tick stream: cd descends from 200 down to 0
    # Early phase: BN positive
    for cd in [200, 180, 150, 125]:
        tracker.ingest(cd, 0.05)
    # Middle phase: BN still positive
    for cd in [115, 100, 80, 60]:
        tracker.ingest(cd, 0.03)
    # Late phase: BN flips negative
    for cd in [50, 40]:
        tracker.ingest(cd, -0.02)
    for cd in [25, 15, 5]:
        tracker.ingest(cd, -0.05)
    s = tracker.summary()
    check("clean flip detected", s["late_flip_detected"], True)
    check("  early BN positive", s["early_bn"] > 0, True)
    check("  late BN negative",  s["late_bn"] < 0, True)

    # No flip: BN stays positive throughout
    tracker = LateFlipTracker()
    for cd in [200, 150, 125, 100, 60, 30, 10]:
        tracker.ingest(cd, 0.05)
    check("no flip — consistent direction", tracker.late_flip_detected, False)

    # Noise (BN near zero): should NOT trigger
    tracker = LateFlipTracker()
    for cd in [150, 125, 100, 60, 30, 10]:
        tracker.ingest(cd, 0.001)  # all below flip magnitude
    check("noise-level BN — no flip", tracker.late_flip_detected, False)

    # Sub-threshold flip (both magnitudes below flip_mag=0.010): noise reject
    tracker = LateFlipTracker()
    # Early +0.008, late -0.007 — both below 0.010 threshold
    for cd in [150, 125]:
        tracker.ingest(cd, 0.008)
    for cd in [30, 10]:
        tracker.ingest(cd, -0.007)
    check("sub-threshold flip — rejected as noise", tracker.late_flip_detected, False)

    # Missing late data: can't detect
    tracker = LateFlipTracker()
    for cd in [150, 125, 100]:
        tracker.ingest(cd, 0.05)
    # Never called for cd <= 30
    check("missing late BN — no flip claimed", tracker.late_flip_detected, False)

    # ── Summary ──
    print("\n  ─────────────────────────────────")
    total = tests_passed + tests_failed
    if tests_failed == 0:
        print(f"  ✅ ALL {total} TESTS PASSED")
        return 0
    else:
        print(f"  ❌ {tests_failed}/{total} TESTS FAILED")
        return 1

if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        sys.exit(_self_test())
    else:
        print("Use --self-test to run unit tests.")
        print("Import this module and use evaluate_shadow_gate_3() + LateFlipTracker() from bot.")
        sys.exit(0)
