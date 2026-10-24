"""
01_clean_data.py
Turn the raw Water Quality Portal and USGS daily-value downloads into small,
tidy CSV files in data/clean/.

Rules applied:
  * keep only environmental samples (quality-control blanks and replicates
    flagged as "Quality Control" are dropped)
  * one value per site, date and variable (median when a day has several
    results, for example integrated time-series subsamples)
  * method merging, in priority order:
      alk : 39086 (filtered, field IPT) > 00410 (unfiltered ANC, field)
            > 29801 (filtered, lab) > 90410 (unfiltered ANC, lab)
      pH  : 00400 (field) > 00403 (lab)
      SC  : 00095 (field) > 90095 (lab)
    the code that supplied each merged value is kept in *_src columns
  * results reported as "Not Detected" are set to half the reporting limit
    and flagged (very few cases)
  * discharge is converted from ft3/s to m3/s

Usage:
  python -I scripts/01_clean_data.py --raw-dir data/raw
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CFS_TO_CMS, CLEAN, ROOT, SITES  # noqa: E402

PRIORITY = {
    "DOC": ["00681"], "alk": ["39086", "00410", "29801", "90410"],
    "pH": ["00400", "00403"], "SC": ["00095", "90095"], "temp": ["00010"],
    "NO3": ["00631"], "TP": ["00665"], "SSC": ["80154"], "Qinst_cfs": ["00061"],
}


def clean_wq(raw_dir, site):
    f = raw_dir / f"wqp_wqx3_{site}.csv"
    d = pd.read_csv(f, low_memory=False, dtype={"USGSpcode": str})
    d["pcode"] = d["USGSpcode"].astype(str).str.zfill(5)
    d = d[~d["Activity_TypeCode"].astype(str).str.startswith("Quality Control")]
    d = d[d["Activity_Media"].astype(str).str.lower().eq("water")]
    val = pd.to_numeric(d["Result_Measure"], errors="coerce")
    nd = d["Result_ResultDetectionCondition"].astype(str).str.contains("Not Detected", na=False)
    dl = pd.to_numeric(d["DetectionLimit_MeasureA"], errors="coerce")
    val = val.where(~(nd & val.isna()), dl / 2)
    d = d.assign(value=val, censored=nd, date=pd.to_datetime(d["Activity_StartDate"]))
    d = d[d["value"].notna()]
    n_raw = len(d)

    daily = (d.groupby(["date", "pcode"])
             .agg(value=("value", "median"), censored=("censored", "max"))
             .reset_index())
    wide = daily.pivot(index="date", columns="pcode", values="value")
    cens = daily.pivot(index="date", columns="pcode", values="censored")

    out = pd.DataFrame(index=wide.index)
    for var, codes in PRIORITY.items():
        s = pd.Series(np.nan, index=wide.index)
        src = pd.Series("", index=wide.index, dtype=object)
        cflag = pd.Series(False, index=wide.index)
        for c in codes:
            if c not in wide:
                continue
            m = s.isna() & wide[c].notna()
            s[m] = wide.loc[m, c]
            src[m] = c
            cflag[m] = cens.loc[m, c].fillna(False).astype(bool)
        out[var] = s
        if len(codes) > 1:
            out[f"{var}_src"] = src
        if cflag.any():
            out[f"{var}_censored"] = cflag
    out["Qinst_m3s"] = out.pop("Qinst_cfs") * CFS_TO_CMS
    out = out.dropna(how="all", subset=list(PRIORITY.keys())[:-1] + ["Qinst_m3s"])
    out.insert(0, "site", site)
    out = out.reset_index()
    return out, n_raw


def clean_q(raw_dir, site):
    f = raw_dir / f"usgs_daily_q_{site}.csv"
    q = pd.read_csv(f, parse_dates=["date"])
    if q.empty:
        return None
    q["Q_m3s"] = pd.to_numeric(q["Q_cfs"], errors="coerce") * CFS_TO_CMS
    q = q.dropna(subset=["Q_m3s"])
    q = q[q["Q_m3s"] > 0]
    return q[["date", "Q_m3s", "approval_status"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    raw_dir = Path(ap.parse_args().raw_dir)
    summary = []
    for site, name in SITES.items():
        wq, n_raw = clean_wq(raw_dir, site)
        num = wq.select_dtypes("number").round(4)
        wq[num.columns] = num
        wq.to_csv(CLEAN / f"wq_samples_{site}.csv", index=False, date_format="%Y-%m-%d")
        for v in ["DOC", "alk", "pH", "SC", "temp", "NO3", "TP", "SSC", "Qinst_m3s"]:
            s = wq.dropna(subset=[v])
            if len(s):
                summary.append({"site": site, "site_name": name, "variable": v,
                                "n_sample_days": len(s),
                                "first_year": int(s.date.dt.year.min()),
                                "last_year": int(s.date.dt.year.max())})
        print(f"{site} {name}: {n_raw} results -> {len(wq)} sample days")
    for site in ["07374000", "07374525"]:
        q = clean_q(raw_dir, site)
        q["Q_m3s"] = q["Q_m3s"].round(1)
        q.to_csv(CLEAN / f"daily_q_{site}.csv", index=False, date_format="%Y-%m-%d")
        summary.append({"site": site, "site_name": SITES[site], "variable": "daily_Q_m3s",
                        "n_sample_days": len(q), "first_year": int(q.date.dt.year.min()),
                        "last_year": int(q.date.dt.year.max())})
        print(f"daily Q {site}: {len(q)} days {q.date.min().date()} to {q.date.max().date()}")
    pd.DataFrame(summary).to_csv(CLEAN / "data_inventory.csv", index=False)
    log = json.loads((raw_dir / "download_log.json").read_text())
    (CLEAN / "download_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
