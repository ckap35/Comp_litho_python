"""Compatibility entry point. Prefer running run_mask_optimization.py."""

from run_mask_optimization import main


if __name__ == "__main__":
    raise SystemExit(main())
