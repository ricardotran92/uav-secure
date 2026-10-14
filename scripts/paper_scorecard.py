"""Score the latest runs of a profile against the paper's figures (lower is better).

  python scripts/paper_scorecard.py --config configs/default.yaml configs/calibrated.yaml
Writes <output_dir>/scorecard.json and prints one line per figure.
"""
from __future__ import annotations

import argparse

from uavsec.config import add_config_args, config_from_args
from uavsec.scorecard import paper_scorecard
from uavsec.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    cfg = config_from_args(parser.parse_args())
    card = paper_scorecard(cfg.output_path)
    save_json(card, cfg.output_path / "scorecard.json")
    print(f"profile: {cfg.output_path}")
    for fig, err in sorted(card["errors"].items()):
        print(f"  {fig:14s} error {err:6.3f}  (weight {card['weights'][fig]:g})")
    print(f"  {'TOTAL':14s} error {card['total_error']:6.3f}")


if __name__ == "__main__":
    main()
