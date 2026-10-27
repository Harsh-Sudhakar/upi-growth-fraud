"""Command-line entry point: ``python -m upi.cli <command>``."""
from __future__ import annotations

import argparse
import time


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="upi", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("simulate", help="generate the simulated UPI platform and build SQL marts")
    sub.add_parser("growth", help="growth analytics report")
    sub.add_parser("abtest", help="A/B testing report")
    sub.add_parser("analytics", help="simulate + growth + abtest")
    sub.add_parser("download", help="download IEEE-CIS data from Kaggle")
    fraud = sub.add_parser("fraud", help="fraud detection model + report")
    fraud.add_argument("--nrows", type=int, default=None, help="use only the first N transactions")
    args = parser.parse_args(argv)

    t0 = time.time()
    if args.command in ("simulate", "analytics"):
        from upi import db, simulate
        con = db.connect(simulate.run())
        db.build_marts(con)
        print("analytics marts built")
    if args.command in ("growth", "analytics"):
        from upi import growth
        growth.run()
        print("-> reports/growth_report.md")
    if args.command in ("abtest", "analytics"):
        from upi import abtest
        abtest.run()
        print("-> reports/abtest_report.md")
    if args.command == "download":
        from upi.fraud import data
        data.download()
    if args.command == "fraud":
        from upi.fraud import pipeline
        pipeline.run(nrows=args.nrows)
        print("-> reports/fraud_report.md")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
