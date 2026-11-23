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
    bp = sub.add_parser("benchmark", help="compare loop vs vectorised pandas, and string vs categorical memory")
    bp.add_argument("--config", default="configs/default.yaml")
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
    elif args.command == "benchmark":
        from pathlib import Path

        from . import benchmark, data

        d = cfg["data"]
        df = data.prepare(data.load_raw(d.get("local_path"), d.get("source", "openml")),
                          d["max_claims"], d["max_exposure"])
        Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
        print(benchmark.report(df, ["VehBrand", "Region", "Area", "VehGas"], ["VehBrand", "Region", "Area", "VehGas"],
                               Path(cfg["output_dir"]) / "benchmark.md"))
    else:
        d = pipeline.approve(cfg["output_dir"], args.by)
        print(f"Approved by {d['approved_by']}: {d['recommendation']}")


if __name__ == "__main__":
    main()
