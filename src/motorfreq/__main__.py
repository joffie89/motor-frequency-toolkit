"""Command line: `python -m motorfreq run` and `python -m motorfreq approve --by "Name"`."""
from __future__ import annotations

import argparse
import json

from . import pipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="motorfreq")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="run the full pipeline and write reports")
    run_p.add_argument("--config", default="configs/default.yaml")
    ap = sub.add_parser("approve", help="record a named approval for the recommended model")
    ap.add_argument("--by", required=True, help="name of the approver")
    ap.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = pipeline.load_config(args.config)
    if args.command == "run":
        r = pipeline.run(cfg)
        print(json.dumps({k: r[k] for k in ("overall_frequency", "out_of_time", "validation_comparison", "alerts")},
                         indent=2, default=str))
        print(f"Report written to {cfg['output_dir']}/report.md. Decision status: {r['decision']['status']}")
    else:
        d = pipeline.approve(cfg["output_dir"], args.by)
        print(f"Approved by {d['approved_by']}: {d['recommendation']}")


if __name__ == "__main__":
    main()
