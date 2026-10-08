#!/usr/bin/env python3
"""Backward-compatible wrapper for probe calibration CLI."""

from activation_drift.cli.calibrate_schwartz_probe import main


if __name__ == "__main__":
    raise SystemExit(main())