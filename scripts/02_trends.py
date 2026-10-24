"""
02_trends.py
Seasonal Mann-Kendall trend tests with Sen slopes, raw and flow-adjusted.

Approach
  1. Samples are reduced to one value per calendar quarter and year (median),
     because sampling frequency changed over time (about monthly before 1996,
     about 12 to 20 samples per year after). Quarters are the seasons.
  2. Raw trend: seasonal Mann-Kendall (pymannkendall.seasonal_test, period 4)
     on the quarterly medians. The Sen slope is in units per year.
  3. Flow-adjusted trend: within the analysis window a LOWESS curve
     (statsmodels, frac = 0.5) of concentration against ln(discharge) is fit
     to all samples; the residuals are reduced to quarterly medians and
     tested in the same way.
  4. Windows:
       * "1997-2025": every St. Francisville sample has a paired discharge,
         so raw and flow-adjusted trends are both reported.
       * "1980-2025": long record, raw trend only (discharge is not paired
         with most samples before 1997). DOC is excluded because it was
         sampled only sporadically before 1996.
     Other sites use the 1997-2025 window when they have enough data.
  5. A series is tested only if at least 60 percent of the quarter-year cells
     in the window have data and at least 15 years are represented.

Outputs: results/tables/trends_seasonal_mk.csv, results/figures/trend_*.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pymannkendall as mk
from statsmodels.nonparametric.smoothers_lowess import lowess

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIGS, MAIN_SITE, SITES, TABLES, VARS, attach_q, load_samples, set_style  # noqa: E402

plt = set_style()
WINDOWS = {"1997-2025": (1997, 2025), "1980-2025": (1980, 2025)}
MIN_COVER, MIN_YEARS = 0.60, 15


def quarterly(df, col, y0, y1):
    d = df[["date", col]].dropna()
    d = d[(d.date.dt.year >= y0) & (d.date.dt.year <= y1)]
    g = d.groupby([d.date.dt.year.rename("yr"), d.date.dt.quarter.rename("q")])[col].median()
    idx = pd.MultiIndex.from_product([range(y0, y1 + 1), [1, 2, 3, 4]], names=["yr", "q"])
    return g.reindex(idx)


def flow_adjust(df, col):
    d = df[["date", col, "Q_m3s"]].dropna()
    lnq = np.log(d["Q_m3s"].values)
    fit = lowess(d[col].values, lnq, frac=0.5, return_sorted=False)
    out = df.copy()
    out[col + "_FA"] = np.nan
    out.loc[d.index, col + "_FA"] = d[col].values - fit
    return out, d, fit


def run_test(series):
    x = series.values.astype(float)
    cover = np.isfinite(x).mean()
    nyears = series.dropna().index.get_level_values("yr").nunique()
    if cover < MIN_COVER or nyears < MIN_YEARS:
        return None, cover, nyears
    r = mk.seasonal_test(x, period=4)
    return r, cover, nyears


def main():
    rows = []
    fa_store = {}
    for site in ["07373420", "07374000", "07374525", "07295100"]:
        d = attach_q(load_samples(site), site)
        for wname, (y0, y1) in WINDOWS.items():
            if wname == "1980-2025" and site != MAIN_SITE:
                continue
            dw = d[(d.date.dt.year >= y0) & (d.date.dt.year <= y1)].copy()
            for v in VARS:
                if v not in dw or dw[v].notna().sum() < 40:
                    continue
                if wname == "1980-2025" and v == "DOC":
                    continue
                med = dw[v].median()
                kinds = [("raw", v)]
                if wname == "1997-2025":
                    qfrac = dw.dropna(subset=[v])["Q_m3s"].notna().mean()
                    if qfrac >= 0.9:
                        dw, dd, fit = flow_adjust(dw, v)
                        kinds.append(("flow_adjusted", v + "_FA"))
                        if site == MAIN_SITE:
                            fa_store[v] = (dw.copy(), dd, fit)
                for kind, col in kinds:
                    s = quarterly(dw, col, y0, y1)
                    r, cover, ny = run_test(s)
                    row = {"site": site, "site_name": SITES[site], "window": wname,
                           "variable": v, "units": VARS[v][0], "type": kind,
                           "n_samples": int(dw[col].notna().sum()),
                           "n_years": ny, "quarter_coverage": round(cover, 2),
                           "median_conc": round(med, 4)}
                    if r is None:
                        row.update({"tested": False})
                    else:
                        row.update({"tested": True, "trend": r.trend, "p_value": r.p,
                                    "tau": r.Tau, "sen_slope_per_yr": r.slope,
                                    "slope_pct_of_median_per_yr": 100 * r.slope / med if med else np.nan,
                                    "change_over_window": r.slope * (y1 - y0)})
                    rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(TABLES / "trends_seasonal_mk.csv", index=False, float_format="%.5g")
    show = res[res.tested == True]  # noqa: E712
    print(show[["site_name", "window", "variable", "type", "n_samples", "trend", "p_value",
                "sen_slope_per_yr", "slope_pct_of_median_per_yr"]].to_string(index=False))

    # --- Figure 1: St. Francisville concentrations with raw Sen trend line
    vs = ["DOC", "alk", "pH", "SC", "NO3", "TP", "SSC", "temp"]
    fig, axs = plt.subplots(4, 2, figsize=(10, 10), sharex=True)
    d = attach_q(load_samples(MAIN_SITE), MAIN_SITE)
    d = d[(d.date.dt.year >= 1980) & (d.date.dt.year <= 2025)]
    for ax, v in zip(axs.flat, vs):
        ax.plot(d.date, d[v], ".", ms=2.5, color="0.45")
        for w, colr in [("1997-2025", "C3"), ("1980-2025", "C0")]:
            r = res[(res.site == MAIN_SITE) & (res.window == w) & (res.variable == v)
                    & (res.type == "raw") & (res.tested == True)]  # noqa: E712
            if len(r):
                y0 = WINDOWS[w][0]
                dd = d[(d.date.dt.year >= y0)][["date", v]].dropna()
                t = dd.date.dt.year + dd.date.dt.day_of_year / 365.25
                slope = r.sen_slope_per_yr.iloc[0]
                icpt = np.median(dd[v] - slope * t)
                tt = np.array([y0, 2025.99])
                lab = f"{w}: {slope:+.3g}/yr, p={r.p_value.iloc[0]:.3f}"
                ax.plot(pd.to_datetime([f"{y0}-01-01", "2025-12-31"]), icpt + slope * tt,
                        color=colr, lw=1.6, label=lab)
        ax.set_title(f"{VARS[v][1]} ({VARS[v][0]})", fontsize=9)
        ax.legend(fontsize=7, loc="upper left", frameon=False)
    fig.suptitle("Mississippi River near St. Francisville (USGS 07373420): samples and seasonal Sen slopes", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "trend_timeseries_stfrancisville.png")
    plt.close(fig)

    # --- Figure 2: concentration-discharge LOWESS and flow-adjusted residuals
    vs2 = ["DOC", "alk", "NO3", "TP"]
    fig, axs = plt.subplots(2, 4, figsize=(12, 5.5))
    for j, v in enumerate(vs2):
        if v not in fa_store:
            continue
        dw, dd, fit = fa_store[v]
        o = np.argsort(dd["Q_m3s"].values)
        axs[0, j].scatter(dd["Q_m3s"], dd[v], s=5, c=dd.date.dt.year, cmap="viridis")
        axs[0, j].plot(dd["Q_m3s"].values[o], fit[o], color="C3", lw=1.8)
        axs[0, j].set_xscale("log")
        axs[0, j].set_xlabel("Discharge (m3/s)")
        axs[0, j].set_title(f"{v} ({VARS[v][0]})", fontsize=9)
        r = res[(res.site == MAIN_SITE) & (res.window == "1997-2025") & (res.variable == v)
                & (res.type == "flow_adjusted")].iloc[0]
        axs[1, j].plot(dw.date, dw[v + "_FA"], ".", ms=3, color="0.4")
        axs[1, j].axhline(0, color="k", lw=0.6)
        axs[1, j].set_title(f"flow-adjusted: {r.sen_slope_per_yr:+.3g}/yr, p={r.p_value:.3f}", fontsize=8)
        axs[1, j].set_ylabel("LOWESS residual")
    fig.suptitle("St. Francisville 1997-2025: concentration vs discharge (top) and flow-adjusted residuals (bottom)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "trend_flow_adjustment.png")
    plt.close(fig)

    # --- Figure 3: summary of percent-per-year slopes
    s = show[(show.window == "1997-2025")].copy()
    s["lab"] = s.site_name + " " + s.variable
    piv = s.pivot_table(index="lab", columns="type", values="slope_pct_of_median_per_yr")
    pv = s.pivot_table(index="lab", columns="type", values="p_value")
    piv = piv.sort_index()
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(piv) + 1.2))
    y = np.arange(len(piv))
    for k, (col, off, c) in enumerate([("raw", -0.18, "C0"), ("flow_adjusted", 0.18, "C1")]):
        if col not in piv:
            continue
        vals = piv[col].values
        sig = pv[col].reindex(piv.index).values < 0.05
        ax.barh(y + off, vals, height=0.34, color=c, alpha=0.9, label=col.replace("_", " "))
        ax.barh(y[~sig] + off, vals[~sig], height=0.34, color="white", alpha=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(piv.index, fontsize=7)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Sen slope, percent of window median per year (faded bars: p >= 0.05)")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Seasonal Mann-Kendall trends, 1997-2025", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "trend_summary_1997_2025.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
