#!/usr/bin/env python3
"""Add LLoPS architecture metadata to a legacy HF-format checkpoint."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--power", type=int, required=True)
    parser.add_argument("--use-linear-term", action="store_true")
    args = parser.parse_args()

    config_path = args.checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    config.update(
        architectures=["GPTSketchLMHeadModel"],
        llops_architecture="learnable_low_rank_polynomial_sketch",
        llops_power=args.power,
        llops_polynomial_degree=2 * args.power,
        llops_use_linear_term=args.use_linear_term,
    )
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Updated {config_path}")


if __name__ == "__main__":
    main()
