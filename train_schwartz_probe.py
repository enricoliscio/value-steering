#!/usr/bin/env python3
"""Backward-compatible wrapper for packaged Schwartz probe trainer CLI."""

from activation_drift.cli.train_schwartz_probe import main


if __name__ == "__main__":
    raise SystemExit(main())
