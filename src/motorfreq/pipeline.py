"""End-to-end pipeline: data -> checks -> features -> models -> evaluation -> report -> approval gate.

Everything is automated except promotion of a new model, which needs a named person's approval.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from . import data, evaluate, feature_selection, features, models, monitor, validate


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _spec(cfg: dict, extra_categorical: list[str] | None = None) -> models.FeatureSpec:
    f = cfg["features"]
    return models.FeatureSpec(
        categorical=list(f["categorical"]) + list(extra_categorical or []),
        binned=list(f["binned"]), log=list(f["log"]), passthrough=list(f["passthrough"]), n_bins=f["n_bins"],
    )


def build_dataset(cfg: dict, raw: pd.DataFrame | None = None) -> pd.DataFrame:
    d, s = cfg["data"], cfg["simulation"]
    df = data.prepare(raw if raw is not None else data.load_raw(d.get("local_path"), d.get("source", "openml")),
                      d["max_claims"], d["max_exposure"])
    validate.validate(df, _spec(cfg).columns + [cfg["features"]["area_grouping"]["column"]])
    df = features.add_periods(df, s["n_periods"], s["seed"])
    dr = s.get("drift", {})
    if dr.get("enabled"):
        df = features.inject_drift(df, dr["feature"], dr["level"], dr["period"], dr["uplift"], s["seed"])
    return df


def run(cfg: dict, raw: pd.DataFrame | None = None) -> dict:
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build_dataset(cfg, raw)
    train, test = evaluate.time_split(df, cfg["simulation"]["test_period"])

    # Location grouping, learned on training data only
    ag = cfg["features"]["area_grouping"]
    mapping = features.fit_area_groups(train, ag["column"], ag["n_groups"], ag["credibility_k"])
    df, train, test = (features.apply_area_groups(x, ag["column"], mapping) for x in (df, train, test))
    spec = _spec(cfg, ["AreaGroup"])
    glm_p, gbm_p = cfg["models"]["glm"], cfg["models"]["gbm"]

    results: dict = {"rows": len(df), "overall_frequency": data.overall_frequency(df)}

    # 1. Champion (GLM) vs challenger (GBM), tested on the later period
    glm = models.fit(models.build_glm(spec, **glm_p), train, spec)
    gbm = models.fit(models.build_gbm(spec, **gbm_p), train, spec)
    p_glm, p_gbm = models.predict(glm, test, spec), models.predict(gbm, test, spec)
    results["out_of_time"] = {"glm": evaluate.metrics(test, p_glm), "gbm": evaluate.metrics(test, p_gbm)}

    # 2. Why time-based validation: the same GLM on a random split hides the drifting segment
    seg_col = cfg["monitoring"]["segment"]
    seg_level = cfg["simulation"]["drift"]["level"]
    r_train, r_test = evaluate.random_split(df, 0.25, cfg["simulation"]["seed"])
    glm_r = models.fit(models.build_glm(spec, **glm_p), r_train, spec)
    p_r = models.predict(glm_r, r_test, spec)

    def seg_oe(d, p):
        m = (d[seg_col] == seg_level).to_numpy()
        return evaluate.observed_expected(d["ClaimNb"].to_numpy()[m], p[m], d["Exposure"].to_numpy()[m])

    results["validation_comparison"] = {
        "segment": f"{seg_col}={seg_level}",
        "time_split_segment_oe": seg_oe(test, p_glm),
        "random_split_segment_oe": seg_oe(r_test, p_r),
    }

    # 3. Location grouping: raw regions vs credibility-weighted groups vs no location
    area_rows = {}
    for label, s in {
        f"grouped regions ({ag['n_groups']} bands)": spec,
        f"raw {ag['column']} ({df[ag['column']].nunique()} levels)": _spec(cfg, [ag["column"]]),
        "without region (Area and Density only)": _spec(cfg),
    }.items():
        m = models.fit(models.build_glm(s, **glm_p), train, s)
        area_rows[label] = evaluate.metrics(test, models.predict(m, test, s))
    results["area_grouping"] = area_rows

    # 4. Feature selection report (validated on the later period)
    fs = feature_selection.report(df, train, test, spec, glm_p)
    fs.to_csv(out_dir / "feature_selection.csv")

    # 5. Monitoring: segment-level O/E on the latest period
    mcfg = cfg["monitoring"]
    mon = monitor.segment_oe(test, p_glm, seg_col, mcfg["min_exposure"], mcfg["oe_threshold"])
    mon.to_csv(out_dir / "monitoring.csv")
    results["alerts"] = mon[mon["alert"]].index.tolist()

    # 6. Approval gate: a better challenger is recommended, never promoted automatically
    better = (results["out_of_time"]["gbm"]["poisson_deviance"]
              < results["out_of_time"]["glm"]["poisson_deviance"])
    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "champion": "glm", "challenger": "gbm",
        "recommendation": "promote challenger" if better else "keep champion",
        "status": "awaiting_approval",
        "approved_by": None,
        "note": "Automated checks passed. A named reviewer must approve before any model is promoted.",
    }
    (out_dir / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    results["decision"] = decision

    _charts(out_dir, test, p_glm, p_gbm)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    _write_report(out_dir, results, fs, mon)
    return results


def approve(output_dir: str | Path, approver: str) -> dict:
    """Record a human approval. The only way the decision status becomes 'approved'."""
    path = Path(output_dir) / "decision.json"
    decision = json.loads(path.read_text(encoding="utf-8"))
    decision.update(status="approved", approved_by=approver,
                    approved_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return decision


def _charts(out_dir: Path, test: pd.DataFrame, p_glm, p_gbm) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c, w = test["ClaimNb"].to_numpy(), test["Exposure"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (name, p) in zip(axes, [("Poisson GLM", p_glm), ("Poisson GBM", p_gbm)]):
        t = evaluate.lift_table(c, p, w)
        ax.plot(t.index, t["observed"], marker="o", label="observed")
        ax.plot(t.index, t["predicted"], marker="x", linestyle="--", label="predicted")
        ax.set_title(f"{name}: lift by predicted-risk decile")
        ax.set_xlabel("decile (1 = lowest predicted risk)")
        ax.set_xticks(range(1, 11))
    axes[0].set_ylabel("claims per vehicle-year")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "lift_chart.png", dpi=120)
    plt.close(fig)


def _write_report(out_dir: Path, r: dict, fs: pd.DataFrame, mon: pd.DataFrame) -> None:
    oot, vc = r["out_of_time"], r["validation_comparison"]
    lines = [
        "# Model and feature report", "",
        "_Public French motor data (freMTPL2). Periods and drift are simulated: illustrative only._", "",
        f"- Policies: {r['rows']:,}",
        f"- Overall frequency: {r['overall_frequency']:.4f} claims per vehicle-year", "",
        "## Champion vs challenger (tested on a later period)", "",
        "| Model | Poisson deviance | O/E | Lift (top/bottom decile) |", "| --- | --- | --- | --- |",
    ]
    for name, m in oot.items():
        lines.append(f"| {name.upper()} | {m['poisson_deviance']:.4f} | {m['oe']:.3f} | {m['lift_top_bottom_decile']:.2f}x |")
    lines += ["", "![Lift chart](lift_chart.png)", "",
              "## Why time-based validation", "",
              f"Segment {vc['segment']} has a simulated risk increase in the latest period.", "",
              f"- Time-based split: segment O/E = **{vc['time_split_segment_oe']:.2f}** (the shift is visible)",
              f"- Random split: segment O/E = **{vc['random_split_segment_oe']:.2f}** (the shift is hidden)", "",
              "## Location grouping", "",
              "| Location feature | Poisson deviance | Lift |", "| --- | --- | --- |"]
    for name, m in r["area_grouping"].items():
        lines.append(f"| {name} | {m['poisson_deviance']:.5f} | {m['lift_top_bottom_decile']:.2f}x |")
    lines += ["", "## Feature selection", "", fs.round(4).to_markdown(), "",
              "## Monitoring (segment O/E, latest period)", "", mon.round(3).head(10).to_markdown(), "",
              f"Alerts: {', '.join(map(str, r['alerts'])) or 'none'}", "",
              "## Decision", "",
              f"Recommendation: **{r['decision']['recommendation']}**. Status: **{r['decision']['status']}** "
              "(promotion requires a named reviewer: `python -m motorfreq approve --by \"Name\"`)."]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
