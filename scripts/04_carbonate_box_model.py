"""
04_carbonate_box_model.py
Part A. Carbonate-system pCO2 and DIC from pH, alkalinity and temperature.
Part B. A two-end-member mixing box model for alkalinity, calibrated by
        Monte Carlo rejection sampling, used to estimate daily alkalinity
        and annual bicarbonate-carbon loads with parameter uncertainty.

PART A
  For St. Francisville samples (1997-2025) with field or lab pH, alkalinity
  and water temperature, pCO2 (uatm), DIC (umol/L) and dissolved CO2 are
  computed with freshwater constants (K1, K2: Plummer and Busenberg 1982;
  K0: Weiss 1974; Kw: Harned and Owen form). Measurement uncertainty is
  propagated by Monte Carlo (1000 draws per sample) with pH sd = 0.05,
  alkalinity sd = 2 percent and temperature sd = 0.2 deg C.
  Atmospheric reference: a linear ramp from 364 uatm (1997) to 426 uatm
  (2025), close to the NOAA global annual mean CO2 mole fraction; this is an
  approximation used only to report the fraction of supersaturated samples.

PART B: the box model
  Daily discharge Q (Baton Rouge, USGS 07374000) is split into a slow
  component Qb and a quick component Qq = Q - Qb with the Eckhardt (2005)
  recursive digital filter (parameters a and BFImax). Each component has a
  fixed alkalinity (A_b, A_q). The two flows mix and pass through one
  well-mixed box with residence time tau (days), representing channel and
  floodplain storage between the basin and the gauge:

      C_mix(t) = (Qb A_b + Qq A_q) / Q
      dC/dt    = (C_mix - C) / tau      (solved exactly for a daily step)

  Five parameters (A_b, A_q, BFImax, a, tau) get uniform priors. 20,000
  parameter sets are run; the 1 percent with the lowest RMSE against
  observed alkalinity in the calibration period (2004-07-01 to 2016-12-31)
  are accepted (rejection sampling with a quantile threshold). Accepted
  sets are evaluated on 2017-2026 samples that were not used for
  calibration. The 5th to 95th percentile of the accepted ensemble gives
  the parameter uncertainty band.

  Annual bicarbonate-carbon load: alkalinity (mg/L as CaCO3) / 50.04 gives
  meq/L; at the observed pH (median near 7.9) nearly all alkalinity is
  HCO3-, so 1 meq of alkalinity is taken as 1 mmol C (12.011 mg C).

Outputs: results/tables/carbonate_*.csv, box_model_*.csv,
         results/figures/carbonate_pco2.png, box_model_*.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pymannkendall as mk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (FIGS, MAIN_SITE, SEED, TABLES, carbonate_from_ph_alk,  # noqa: E402
                    load_daily_q, load_samples, set_style, water_year)

plt = set_style()
rng = np.random.default_rng(SEED)

PRIORS = {"A_b": (100.0, 250.0), "A_q": (10.0, 150.0), "BFImax": (0.20, 0.95),
          "a": (0.95, 0.995), "tau": (0.5, 60.0)}
N_RUNS, ACCEPT_FRAC = 20000, 0.01
CAL_START, CAL_END = "2004-07-01", "2016-12-31"


def part_a():
    d = load_samples(MAIN_SITE)
    d = d[(d.date.dt.year >= 1997) & (d.date.dt.year <= 2025)].dropna(subset=["pH", "alk", "temp"])
    pco2, dic, co2 = carbonate_from_ph_alk(d.pH, d.alk, d.temp)
    d = d.assign(pCO2_uatm=pco2, DIC_umolL=dic, CO2_umolL=co2)
    # Monte Carlo measurement uncertainty
    nd = 1000
    ph = d.pH.values[:, None] + rng.normal(0, 0.05, (len(d), nd))
    al = d.alk.values[:, None] * (1 + rng.normal(0, 0.02, (len(d), nd)))
    tt = d.temp.values[:, None] + rng.normal(0, 0.2, (len(d), nd))
    pc, _, _ = carbonate_from_ph_alk(ph, al, tt)
    d["pCO2_p05"] = np.percentile(pc, 5, axis=1)
    d["pCO2_p95"] = np.percentile(pc, 95, axis=1)
    yr = d.date.dt.year + d.date.dt.day_of_year / 365.25
    d["pCO2_atm_ref"] = 364 + (426 - 364) * (yr - 1997) / (2025 - 1997)
    d["supersaturated"] = d.pCO2_uatm > d.pCO2_atm_ref
    cols = ["date", "pH", "pH_src", "alk", "alk_src", "temp", "pCO2_uatm", "pCO2_p05",
            "pCO2_p95", "DIC_umolL", "CO2_umolL", "pCO2_atm_ref", "supersaturated"]
    d[cols].to_csv(TABLES / "carbonate_pco2_samples.csv", index=False, float_format="%.4g",
                   date_format="%Y-%m-%d")

    rel_unc = ((d.pCO2_p95 - d.pCO2_p05) / 2 / d.pCO2_uatm).median()
    month = d.groupby(d.date.dt.month).pCO2_uatm.agg(["count", "median",
                                                       lambda s: s.quantile(0.25),
                                                       lambda s: s.quantile(0.75)])
    month.columns = ["n", "median_pCO2", "q25", "q75"]
    month.to_csv(TABLES / "carbonate_pco2_by_month.csv", float_format="%.1f")
    # seasonal Mann-Kendall on quarterly medians of pCO2 (same design as 02_trends)
    g = d.groupby([d.date.dt.year, d.date.dt.quarter]).pCO2_uatm.median()
    idx = pd.MultiIndex.from_product([range(1997, 2026), [1, 2, 3, 4]])
    r = mk.seasonal_test(g.reindex(idx).values, period=4)
    summ = pd.DataFrame([{
        "n_samples": len(d), "first_date": d.date.min().date(), "last_date": d.date.max().date(),
        "pH_field_fraction": (d.pH_src == "00400").mean(),
        "median_pCO2_uatm": d.pCO2_uatm.median(), "q25_pCO2": d.pCO2_uatm.quantile(0.25),
        "q75_pCO2": d.pCO2_uatm.quantile(0.75), "min_pCO2": d.pCO2_uatm.min(),
        "max_pCO2": d.pCO2_uatm.max(),
        "fraction_supersaturated": d.supersaturated.mean(),
        "median_DIC_umolL": d.DIC_umolL.median(),
        "median_ratio_pCO2_to_atm": (d.pCO2_uatm / d.pCO2_atm_ref).median(),
        "median_MC_half_width_90pct_relative": rel_unc,
        "smk_trend": r.trend, "smk_p": r.p, "smk_sen_slope_uatm_per_yr": r.slope,
        "median_pH": d.pH.median()}])
    summ.to_csv(TABLES / "carbonate_pco2_summary.csv", index=False, float_format="%.4g")
    print(summ.T.to_string())

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
    axs[0].errorbar(d.date, d.pCO2_uatm, yerr=[d.pCO2_uatm - d.pCO2_p05, d.pCO2_p95 - d.pCO2_uatm],
                    fmt=".", ms=3, color="C0", ecolor="0.75", elinewidth=0.6)
    axs[0].plot(d.date, d.pCO2_atm_ref, "k--", lw=1, label="approx. atmospheric pCO2")
    axs[0].set_yscale("log")
    axs[0].set_ylabel("pCO2 (uatm)")
    axs[0].legend(frameon=False, fontsize=8)
    axs[0].set_title("St. Francisville: pCO2 from pH, alkalinity and temperature (bars: 90% MC range)", fontsize=9)
    bp = [d.loc[d.date.dt.month == m, "pCO2_uatm"].values for m in range(1, 13)]
    axs[1].boxplot(bp, showfliers=False)
    axs[1].set_xticks(range(1, 13))
    axs[1].set_xticklabels(list("JFMAMJJASOND"))
    axs[1].set_title("By month, 1997-2025", fontsize=9)
    axs[1].set_ylabel("pCO2 (uatm)")
    fig.tight_layout()
    fig.savefig(FIGS / "carbonate_pco2.png")
    plt.close(fig)
    return d


def eckhardt_box(Q, P, save_idx=None, full=False):
    """Run the filter + mixing + box for parameter arrays P (dict of arrays,
    length n). Q is a 1-D numpy array of daily discharge. Returns
    concentrations at save_idx (n x len(save_idx)) or the full series."""
    n = len(P["A_b"])
    a, bfi = P["a"], P["BFImax"]
    k = 1 - np.exp(-1.0 / P["tau"])
    b = np.full(n, Q[0] * bfi)
    C = (b * P["A_b"] + (Q[0] - b) * P["A_q"]) / Q[0]
    out = np.empty((n, len(Q) if full else len(save_idx)))
    pos = {t: j for j, t in enumerate(save_idx)} if not full else None
    for t in range(len(Q)):
        if t > 0:
            b = ((1 - bfi) * a * b + (1 - a) * bfi * Q[t]) / (1 - a * bfi)
            b = np.minimum(b, Q[t])
        cmix = (b * P["A_b"] + (Q[t] - b) * P["A_q"]) / Q[t]
        C = C + (cmix - C) * k if t > 0 else cmix
        if full:
            out[:, t] = C
        elif t in pos:
            out[:, pos[t]] = C
    return out


def part_b():
    q = load_daily_q("07374000").interpolate(limit=10)
    q = q[q.index <= "2026-09-30"]
    Q = q.values
    s = load_samples(MAIN_SITE).dropna(subset=["alk"])
    s = s[(s.date >= q.index[0]) & (s.date <= q.index[-1])]
    s = s.groupby("date").alk.median()
    idx = np.array([q.index.get_loc(t) for t in s.index])
    cal = (s.index >= CAL_START) & (s.index <= CAL_END)
    val = s.index > CAL_END
    obs = s.values

    P = {k: rng.uniform(lo, hi, N_RUNS) for k, (lo, hi) in PRIORS.items()}
    sim = eckhardt_box(Q, P, save_idx=list(idx))
    rmse_cal = np.sqrt(np.mean((sim[:, cal] - obs[cal]) ** 2, axis=1))
    thr = np.quantile(rmse_cal, ACCEPT_FRAC)
    acc = rmse_cal <= thr
    rmse_val = np.sqrt(np.mean((sim[:, val] - obs[val]) ** 2, axis=1))

    def nse(o, p):
        return 1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2)

    # parameter table: prior vs accepted
    prow = []
    for k, (lo, hi) in PRIORS.items():
        v = P[k][acc]
        prow.append({"parameter": k, "prior_low": lo, "prior_high": hi,
                     "accepted_p05": np.percentile(v, 5), "accepted_median": np.median(v),
                     "accepted_p95": np.percentile(v, 95),
                     "spearman_with_rmse_all_runs": pd.Series(P[k]).corr(pd.Series(rmse_cal), method="spearman")})
    ptab = pd.DataFrame(prow)
    ptab.to_csv(TABLES / "box_model_parameters.csv", index=False, float_format="%.4g")
    print(ptab.to_string(index=False))

    # full daily series for accepted runs
    Pa = {k: v[acc] for k, v in P.items()}
    full = eckhardt_box(Q, Pa, full=True)
    med = np.median(full, axis=0)
    lo, hi = np.percentile(full, [5, 95], axis=0)
    med_at = med[idx]
    lo_at, hi_at = lo[idx], hi[idx]
    best = np.argmin(rmse_cal)
    # benchmarks: power-law C-Q rating fit on calibration samples
    lq = np.log(Q[idx])
    coef = np.polyfit(lq[cal], np.log(obs[cal]), 1)
    rating = np.exp(np.polyval(coef, lq))
    skill = []
    for per, m in [("calibration 2004-07 to 2016", cal), ("validation 2017 to 2026-09", val)]:
        skill.append({"period": per, "n_samples": int(m.sum()),
                      "ensemble_median_RMSE": np.sqrt(np.mean((med_at[m] - obs[m]) ** 2)),
                      "ensemble_median_NSE": nse(obs[m], med_at[m]),
                      "ensemble_median_bias": np.mean(med_at[m] - obs[m]),
                      "best_run_RMSE": np.sqrt(np.mean((sim[best, m] - obs[m]) ** 2)),
                      "accepted_RMSE_p05": np.percentile((rmse_cal if m is cal else rmse_val)[acc], 5),
                      "accepted_RMSE_p95": np.percentile((rmse_cal if m is cal else rmse_val)[acc], 95),
                      "obs_within_90pct_band": np.mean((obs[m] >= lo_at[m]) & (obs[m] <= hi_at[m])),
                      "powerlaw_rating_RMSE": np.sqrt(np.mean((rating[m] - obs[m]) ** 2)),
                      "powerlaw_rating_NSE": nse(obs[m], rating[m]),
                      "constant_mean_RMSE": np.sqrt(np.mean((obs[cal].mean() - obs[m]) ** 2))})
    sk = pd.DataFrame(skill)
    sk.to_csv(TABLES / "box_model_skill.csv", index=False, float_format="%.4g")
    print(sk.T.to_string())
    print(f"acceptance threshold RMSE = {thr:.2f} mg/L; accepted {acc.sum()} of {N_RUNS}")

    # annual loads of bicarbonate carbon, water years 2005-2026 (2026 through Sep 30)
    gC_per_m3 = full / 50.04 * 12.011           # mg/L CaCO3 -> meq/L -> g C per m3
    daily_load = gC_per_m3 * Q[None, :] * 86400  # g C per day, per run
    wy = water_year(pd.Series(q.index)).values
    rows = []
    for y in range(2005, 2027):
        m = wy == y
        if m.sum() < 360:
            continue
        L = daily_load[:, m].sum(axis=1) / 1e12      # Tg C per year
        rows.append({"water_year": y, "days": int(m.sum()),
                     "mean_Q_m3s": Q[m].mean(),
                     "HCO3_C_load_TgC_median": np.median(L),
                     "HCO3_C_load_TgC_p05": np.percentile(L, 5),
                     "HCO3_C_load_TgC_p95": np.percentile(L, 95)})
    lt = pd.DataFrame(rows)
    lt.to_csv(TABLES / "box_model_annual_HCO3C_load.csv", index=False, float_format="%.4g")
    print(lt.to_string(index=False))
    pd.DataFrame({"date": q.index, "Q_m3s": Q, "alk_median": med, "alk_p05": lo,
                  "alk_p95": hi}).to_csv(TABLES / "box_model_daily_alkalinity.csv", index=False,
                                         float_format="%.2f", date_format="%Y-%m-%d")

    # figures
    fig, axs = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                            gridspec_kw={"height_ratios": [2, 1]})
    axs[0].fill_between(q.index, lo, hi, color="C1", alpha=0.35, lw=0, label="accepted ensemble 5-95%")
    axs[0].plot(q.index, med, color="C1", lw=0.9, label="ensemble median")
    axs[0].plot(s.index[cal], obs[cal], "k.", ms=4, label="observed (calibration)")
    axs[0].plot(s.index[val], obs[val], "o", mfc="none", mec="C3", ms=3.5, label="observed (validation)")
    axs[0].axvline(pd.Timestamp(CAL_END), color="0.4", ls=":", lw=1)
    axs[0].set_ylabel("Alkalinity (mg/L as CaCO3)")
    axs[0].legend(fontsize=7, frameon=False, ncol=4, loc="lower left")
    axs[0].set_title("Two-end-member mixing box model of alkalinity at St. Francisville, "
                     "Monte Carlo rejection calibration", fontsize=10)
    axs[1].plot(q.index, Q, color="C0", lw=0.7)
    axs[1].set_ylabel("Q, Baton Rouge (m3/s)")
    fig.tight_layout()
    fig.savefig(FIGS / "box_model_alkalinity.png")
    plt.close(fig)

    fig, axs = plt.subplots(1, 5, figsize=(13, 2.8))
    for ax, (k, (lo_, hi_)) in zip(axs, PRIORS.items()):
        ax.hist(P[k], bins=30, range=(lo_, hi_), color="0.8", density=True, label="prior")
        ax.hist(P[k][acc], bins=30, range=(lo_, hi_), color="C1", alpha=0.8, density=True,
                label="accepted")
        ax.set_title(k, fontsize=9)
        ax.set_yticks([])
    axs[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Box model parameters: uniform priors vs accepted (best 1% of 20,000 runs)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "box_model_posteriors.png")
    plt.close(fig)
    return lt


if __name__ == "__main__":
    part_a()
    part_b()
