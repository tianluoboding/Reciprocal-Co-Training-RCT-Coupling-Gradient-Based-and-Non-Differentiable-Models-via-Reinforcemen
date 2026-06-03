#!/usr/bin/env python3
"""Public entry point for qwen2_llm_baseline_train."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from qwen2_7b.shared.workflows import train_supervised_lora


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a Qwen2 RCT example config file")
    args = parser.parse_args()
    result = train_supervised_lora(args.config, stage_name="qwen2_llm_baseline")
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
