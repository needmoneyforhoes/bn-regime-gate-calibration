"""Check if all required libs are installed on the VPS."""
import sys

requirements = [
    ("numpy", "Vectorized computation"),
    ("scipy", "Statistical tests (Wilson CI, p-values)"),
    ("sklearn", "Logistic/RF models, proper AUC/ROC"),
    ("matplotlib", "Plots (optional but recommended)"),
]

missing = []
print("Checking required libraries for regime_gate_pro_calibrator.py:\n")
for lib, desc in requirements:
    try:
        __import__(lib)
        mod = sys.modules[lib]
        ver = getattr(mod, '__version__', 'unknown')
        print(f"  ✓ {lib:<12} {ver:<12} — {desc}")
    except ImportError:
        print(f"  ✗ {lib:<12} MISSING       — {desc}")
        missing.append(lib)

if missing:
    print(f"\nTo install missing libraries:")
    print(f"  pip install {' '.join(missing)}")
    sys.exit(1)
else:
    print(f"\n✓ All required libraries are installed.")
