#!/usr/bin/env python3
"""Generate fixed-grid ROC/PR curve CSVs from configured probability files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from qwen2_7b.shared.curve_utils import run_curve_generation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a Qwen2 RCT config file")
    args = parser.parse_args()
    result = run_curve_generation(args.config)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
