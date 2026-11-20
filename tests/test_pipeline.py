import json
from pathlib import Path

from motorfreq import pipeline


def test_pipeline_runs_end_to_end_and_waits_for_approval(synthetic_raw, cfg):
    results = pipeline.run(cfg, raw=synthetic_raw)
    out = Path(cfg["output_dir"])
    for name in ("report.md", "results.json", "decision.json", "feature_selection.csv",
                 "monitoring.csv", "lift_chart.png"):
        assert (out / name).exists(), name
    assert results["decision"]["status"] == "awaiting_approval"
    assert results["decision"]["approved_by"] is None


def test_only_a_named_approval_changes_status(synthetic_raw, cfg):
    pipeline.run(cfg, raw=synthetic_raw)
    decision = pipeline.approve(cfg["output_dir"], "Reviewer Name")
    saved = json.loads((Path(cfg["output_dir"]) / "decision.json").read_text())
    assert decision["status"] == saved["status"] == "approved"
    assert saved["approved_by"] == "Reviewer Name"
