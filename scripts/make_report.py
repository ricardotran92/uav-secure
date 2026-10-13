"""Phase 5 - build outputs/report/report.md (+ figures, summary.json) from the latest runs."""
from __future__ import annotations

import argparse

from uavsec.config import add_config_args, config_from_args
from uavsec.report import build_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    report = build_report(config_from_args(parser.parse_args()))
    print(f"report written to {report}")


if __name__ == "__main__":
    main()
