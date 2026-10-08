#!/usr/bin/env python3
"""Backward-compatible wrapper for packaged Schwartz probe evaluator CLI."""

from activation_drift.cli.eval_schwartz_probe import main


if __name__ == "__main__":
    raise SystemExit(main())
