"""
05_loads.py
Annual loads at St. Francisville from a rating regression and daily
discharge at Baton Rouge (USGS 07374000), water years 2005-2026.

Model (the LOADEST "model 7" form, fit to concentration):
  ln C = b0 + b1 lnQc + b2 lnQc^2 + b3 sin(2 pi T) + b4 cos(2 pi T) + b5 Tc + b6 Tc^2
  lnQc = ln Q minus its mean over the samples, T = decimal year, Tc = T
  minus its mean. Fitted by ordinary least squares (statsmodels) on
  sample dates from 2004-03-17 to 2026-09-30. Daily concentrations are
  back-transformed with the Duan smearing estimator and multiplied by daily
  discharge. Uncertainty: 500 bootstrap resamples of the calibration
  samples (refit and re-predict), giving 5th and 95th percentile annual loads.

Constituents: nitrate plus nitrite (as N), DOC, total phosphorus,
suspended sediment, and alkalinity converted to bicarbonate carbon
(1 meq alkalinity = 1 mmol C, see 04_carbonate_box_model.py).

Outputs: results/tables/loads_annual.csv, loads_regression_fit.csv,
         results/figures/loads_annual.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIGS, MAIN_SITE, SEED, TABLES, load_daily_q, load_samples, set_style, water_year  # noqa: E402

plt = set_style()
rng = np.random.default_rng(SEED)
N_BOOT = 500
# name: (column, multiplier from mg/L to output mass per m3, output unit, scale to unit)
CONST = {
    "NO3_N": ("NO3", 1.0, "Gg N", 1e9),
    "DOC": ("DOC", 1.0, "Gg C", 1e9),
    "TP": ("TP", 1.0, "Gg P", 1e9),
    "SSC": ("SSC", 1.0, "Tg", 1e12),
    "HCO3_C": ("alk", 12.011 / 50.04, "Tg C", 1e12),
}


def design(lnq, t, lnq_mean, t_mean):
    lq = lnq - lnq_mean
    tc = t - t_mean
    return np.column_stack([np.ones_like(lq), lq, lq**2, np.sin(2 * np.pi * t),
                            np.cos(2 * np.pi * t), tc, tc**2])


def main():
    q = load_daily_q("07374000").interpolate(limit=10)
    q = q[q.index <= "2026-09-30"]
    tq = q.index.year + (q.index.day_of_year - 0.5) / 365.25
    wy = water_year(pd.Series(q.index)).values
    s = load_samples(MAIN_SITE).set_index("date")
    s = s[(s.index >= q.index[0]) & (s.index <= q.index[-1])]
    fit_rows, ann_rows = [], []
    for name, (col, mult, unit, scale) in CONST.items():
        d = s[[col]].dropna()
        d = d[d[col] > 0]
        lnq_s = np.log(q.reindex(d.index).values)
        t_s = d.index.year + (d.index.day_of_year - 0.5) / 365.25
        ok = np.isfinite(lnq_s)
        y = np.log(d[col].values[ok])
        lnq_s, t_s = lnq_s[ok], np.asarray(t_s)[ok]
        lm, tm = lnq_s.mean(), t_s.mean()
        X = design(lnq_s, t_s, lm, tm)
        res = sm.OLS(y, X).fit()
        smear = np.mean(np.exp(res.resid))
        Xd = design(np.log(q.values), np.asarray(tq), lm, tm)
        conc = np.exp(Xd @ res.params) * smear * mult           # g per m3
        daily = conc * q.values * 86400                         # g per day
        fit_rows.append({"constituent": name, "n_samples": len(y), "R2_lnC": res.rsquared,
                         "resid_sd_lnC": np.sqrt(res.scale), "smearing": smear,
                         **{f"b{i}": v for i, v in enumerate(res.params)},
                         **{f"p_b{i}": v for i, v in enumerate(res.pvalues)}})
        # bootstrap
        boot = np.empty((N_BOOT, len(q)))
        for b in range(N_BOOT):
            ii = rng.integers(0, len(y), len(y))
            rb = sm.OLS(y[ii], X[ii]).fit()
            sb = np.mean(np.exp(rb.resid))
            boot[b] = np.exp(Xd @ rb.params) * sb * mult * q.values * 86400
        for yr in range(2005, 2027):
            m = wy == yr
            if m.sum() < 360:
                continue
            bl = boot[:, m].sum(axis=1) / scale
            ann_rows.append({"constituent": name, "unit_per_yr": unit, "water_year": yr,
                             "load": daily[m].sum() / scale, "load_p05": np.percentile(bl, 5),
                             "load_p95": np.percentile(bl, 95),
                             "flow_weighted_conc_mgL": daily[m].sum() / (q.values[m].sum() * 86400) / mult,
                             "n_samples_in_wy": int(((d.index >= f"{yr-1}-10-01") & (d.index <= f"{yr}-09-30")).sum())})
    fit = pd.DataFrame(fit_rows)
    fit.to_csv(TABLES / "loads_regression_fit.csv", index=False, float_format="%.4g")
    ann = pd.DataFrame(ann_rows)
    ann.to_csv(TABLES / "loads_annual.csv", index=False, float_format="%.4g")
    print(fit[["constituent", "n_samples", "R2_lnC", "resid_sd_lnC", "smearing", "b5", "p_b5"]].to_string(index=False))
    summ = (ann[ann.water_year <= 2025].groupby(["constituent", "unit_per_yr"])
            .load.agg(["mean", "min", "max"]).reset_index())
    summ.to_csv(TABLES / "loads_summary_wy2005_2025.csv", index=False, float_format="%.4g")
    print(summ.to_string(index=False))

    # compare with box model HCO3-C load if available
    bx = TABLES / "box_model_annual_HCO3C_load.csv"
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    for ax, name in zip(axs.flat, CONST):
        a = ann[ann.constituent == name]
        ax.bar(a.water_year, a.load, color="C0", alpha=0.8, label="rating regression")
        ax.errorbar(a.water_year, a.load, yerr=[a.load - a.load_p05, a.load_p95 - a.load],
                    fmt="none", ecolor="k", elinewidth=0.7)
        if name == "HCO3_C" and bx.exists():
            b = pd.read_csv(bx)
            ax.plot(b.water_year, b.HCO3_C_load_TgC_median, "o-", color="C1", ms=3, lw=1,
                    label="box model (median)")
            ax.legend(fontsize=7, frameon=False)
        ax.set_title(f"{name} ({a.unit_per_yr.iloc[0]}/yr)", fontsize=9)
    ax = axs.flat[-1]
    wq = pd.Series(q.values, index=q.index).groupby(wy).mean()
    wq = wq[(wq.index >= 2005) & (wq.index <= 2026)]
    ax.bar(wq.index, wq.values, color="0.5")
    ax.set_title("Mean discharge, Baton Rouge (m3/s)", fontsize=9)
    fig.suptitle("Annual loads past St. Francisville, water years 2005-2026 (bars: 90% bootstrap range; WY2026 provisional)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "loads_annual.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
