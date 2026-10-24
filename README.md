# Lower Mississippi River Carbon and Water Chemistry: Trends, Machine Learning and a Box Model

Anamika Dristi, Ph.D.

## Summary

This repository analyzes long-term changes in carbon and water chemistry of the Lower Mississippi River with public U.S. Geological Survey (USGS) data only. It combines four pieces of work on the same data:

1. Seasonal Mann-Kendall trend tests with Sen slopes, raw and flow-adjusted, for dissolved organic carbon (DOC), alkalinity, pH, specific conductance, water temperature, nitrate, total phosphorus and suspended sediment.
2. Machine-learning models (random forest and gradient boosting) compared with a linear baseline and a monthly climatology for predicting nitrate and DOC from discharge, antecedent flow, season and water temperature, scored with time-blocked cross-validation.
3. A carbonate-system estimate of river pCO2 from pH, alkalinity and temperature, and a two-end-member mixing box model of alkalinity calibrated by Monte Carlo rejection sampling.
4. Annual loads of nitrate, DOC, total phosphorus, suspended sediment and bicarbonate carbon from a rating regression and daily discharge.

Every number in this README was produced by the scripts in this repository on the access date given below.

**Independence statement.** This is an independent analysis of public USGS data. It is separate from the author's LSU dissertation datasets: no field measurements from the LSU Watershed Hydrology Lab are used, copied or referenced here.

## Data sources

All data were downloaded on **2026-10-23** (UTC) without an API key.

| Source | Service | Link |
|---|---|---|
| Water Quality Portal (USGS NWIS discrete samples) | WQX 3.0 Result service, `basicPhysChem` profile | https://www.waterqualitydata.us/ |
| USGS Water Data APIs (daily mean discharge, parameter 00060, statistic 00003) | OGC API `daily` collection | https://api.waterdata.usgs.gov/ |

Sites (monitoring location pages follow the form `https://waterdata.usgs.gov/monitoring-location/USGS-<site number>/`):

| USGS site | Name | Used for |
|---|---|---|
| 07373420 | Mississippi River near St. Francisville, LA | main water-quality site (all analyses) |
| 07374000 | Mississippi River at Baton Rouge, LA | daily discharge (2004-03-17 onward); trends where coverage allows |
| 07374525 | Mississippi River at Belle Chasse, LA | trends; daily discharge (2008-10-29 onward) |
| 07295100 | Mississippi River at Tarbert Landing, MS | suspended sediment trend only (no USGS daily discharge series is served for this site) |

USGS parameter codes requested: 00681 (DOC, filtered), 39086 / 00410 / 29801 / 90410 (alkalinity or ANC as CaCO3), 00400 / 00403 (pH, field / lab), 00095 / 90095 (specific conductance, field / lab), 00010 (water temperature), 00631 (nitrate plus nitrite as N, filtered), 00665 (total phosphorus), 80154 (suspended sediment concentration), 00061 (instantaneous discharge at sampling) and 00060 (daily mean discharge).

Sample days available after cleaning (from `data/clean/data_inventory.csv`):

| Variable | St. Francisville | Baton Rouge | Belle Chasse | Tarbert Landing |
|---|---|---|---|---|
| DOC | 467 (1978-2026) | 208 (1991-2017) | 288 (1978-2026) | - |
| Alkalinity / ANC | 946 (1964-2026) | 180 (1975-2017) | 424 (1977-2026) | - |
| pH | 1220 (1954-2026) | 578 (1975-2023) | 424 (1977-2026) | - |
| Specific conductance | 1237 (1954-2026) | 588 (1975-2023) | 424 (1977-2026) | - |
| Water temperature | 828 (1968-2026) | 582 (1975-2023) | 420 (1977-2026) | 4 |
| Nitrate plus nitrite | 583 (1973-2026) | 594 (1991-2023) | 392 (1979-2026) | - |
| Total phosphorus | 673 (1974-2026) | 176 (1975-2016) | 407 (1977-2026) | - |
| Suspended sediment | 626 (1978-2026) | 311 (1975-2026) | 344 (2006-2026) | 701 (1975-2025) |
| Discharge at sampling | 438 (1967-2026) | 307 (1975-2026) | 525 (1978-2026) | 996 (1978-2026) |

Daily mean discharge: Baton Rouge 8,252 days (2004-03-17 to 2026-10-22); Belle Chasse 6,242 days (2008-10-29 to 2026-10-22). Recent values are provisional.

## Methods

**Cleaning (`01_clean_data.py`).** Quality-control samples are dropped, results are reduced to one value per site, date and variable (median), and alternative methods are merged in a fixed priority order (for alkalinity: filtered field inflection-point titration, then unfiltered field ANC, then laboratory methods; for pH and conductance: field, then laboratory). The source code of each merged value is kept in a `*_src` column. Results reported as not detected (one sample day at St. Francisville and two at Baton Rouge) are set to half the reporting limit and flagged.

**Trends (`02_trends.py`).** Samples are reduced to quarterly medians (quarters are the seasons) because sampling frequency changed over time. Raw trends use the seasonal Mann-Kendall test with Sen slope (`pymannkendall.seasonal_test`, period 4). Flow-adjusted trends use residuals from a LOWESS fit (statsmodels, frac 0.5) of concentration against ln(discharge) within the window, tested the same way. Discharge for each sample is the instantaneous value measured with the sample when present, otherwise the daily mean at the paired gauge. Two windows are used at St. Francisville: 1997-2025, when nearly every sample has a paired discharge (raw and flow-adjusted), and 1980-2025 (raw only; DOC excluded because it was sampled only sporadically before 1996). A series is tested only if at least 60 percent of quarter-year cells have data and at least 15 years are represented.

**Machine learning (`03_ml_model.py`).** Targets: nitrate and DOC at St. Francisville, 2004-2026 (322 and 320 samples). Predictors: ln daily discharge at Baton Rouge on the sample day; ln mean discharge over the previous 7, 30 and 90 days; the 7-day change in ln discharge; sine and cosine of day of year; water temperature. Models: ordinary least squares (linear baseline), random forest (500 trees), gradient boosting (300 trees, depth 3) and a monthly-mean climatology. Hyperparameters were fixed in advance, not tuned on test data. Validation: leave-one-block-out with contiguous 3-year blocks, plus a forward test (train 2004-2019, test 2020-2026). Importance: permutation importance (increase in RMSE) on each held-out block. A variant with a decimal-year predictor is also reported in `ml_skill.csv`.

**Carbonate system (`04_carbonate_box_model.py`, part A).** pCO2, DIC and dissolved CO2 are computed from pH, alkalinity and temperature with freshwater constants (K1 and K2 from Plummer and Busenberg 1982, K0 from Weiss 1974). Measurement uncertainty is propagated by Monte Carlo (1,000 draws per sample; pH sd 0.05, alkalinity sd 2 percent, temperature sd 0.2 deg C). Supersaturation is judged against an approximate atmospheric pCO2 that rises linearly from 364 uatm (1997) to 426 uatm (2025).

**Box model (`04_carbonate_box_model.py`, part B).** Daily discharge is split into slow and quick components with the Eckhardt (2005) recursive digital filter. Each component carries a fixed alkalinity (A_b, A_q); the mixture passes through one well-mixed box with residence time tau. Five parameters (A_b, A_q, BFImax, filter constant a, tau) have uniform priors. Of 20,000 random parameter sets, the 1 percent with the lowest RMSE against observed alkalinity in 2004-07-01 to 2016-12-31 are accepted; samples from 2017 to September 2026 are held out for validation. The accepted ensemble gives the 5th to 95th percentile band. Daily alkalinity is converted to bicarbonate carbon (1 meq alkalinity taken as 1 mmol C, reasonable at the observed median pH of 7.8) and summed to water-year loads.

**Loads (`05_loads.py`).** A LOADEST-style regression, ln C = b0 + b1 lnQ + b2 lnQ^2 + b3 sin(2 pi t) + b4 cos(2 pi t) + b5 t + b6 t^2 (centered), is fit by OLS to samples from 2004-03-17 to 2026-09-30, back-transformed with the Duan smearing factor, and applied to daily discharge at Baton Rouge. The 90 percent range comes from 500 bootstrap refits.

## Repository structure

```
mississippi-carbon-trends-ml-boxmodel/
  README.md  LICENSE  requirements.txt  .gitignore  run_all.sh
  scripts/
    common.py                    paths, site table, carbonate functions
    00_download_data.py          Water Quality Portal and USGS daily-value downloads
    01_clean_data.py             tidy one-row-per-sample-day tables
    02_trends.py                 seasonal Mann-Kendall, Sen slope, LOWESS flow adjustment
    03_ml_model.py               linear, random forest, gradient boosting, blocked CV
    04_carbonate_box_model.py    pCO2 from pH and alkalinity; Monte Carlo box model
    05_loads.py                  rating-regression annual loads with bootstrap
  data/clean/                    small cleaned public CSVs, inventory and download log
  results/tables/                all result tables (CSV)
  results/figures/               all figures (PNG)
```

Raw downloads go to `data/raw/`, which is excluded from version control and re-created by the download script.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh                 # downloads, cleans and runs every analysis
SKIP_DOWNLOAD=1 bash run_all.sh # re-run analyses on existing data/raw
```

Scripts are run with `python -I -B` (isolated mode, no bytecode files). A full run took about 6 minutes on a laptop, most of it in the download and the cross-validation. Downloading again on a later date will add new samples and may change results slightly.

## Key findings

All results are for St. Francisville (USGS 07373420) unless another site is named. Units per year are Sen slopes.

**Trends, 1997-2025 (29 years, 417 to 432 samples per variable).**

- Alkalinity increased: +0.41 mg/L as CaCO3 per year raw (p = 0.006) and +0.49 per year flow-adjusted (p < 0.001), about 0.4 to 0.5 percent of the 106 mg/L median per year.
- Nitrate plus nitrite decreased in raw concentration, -0.0079 mg N/L per year (p = 0.015, -0.6 percent per year), but the flow-adjusted trend was not significant (-0.0062 per year, p = 0.11). Part of the raw decline may reflect flow conditions rather than a change in sources.
- Suspended sediment decreased, -1.34 mg/L per year raw (p = 0.005) and -1.35 flow-adjusted (p < 0.001). At Tarbert Landing the decline was -4.0 mg/L per year raw and -3.2 flow-adjusted (both p < 0.001).
- DOC showed no trend (raw p = 0.96, flow-adjusted p = 0.71; median 3.6 mg/L). Total phosphorus, pH and specific conductance also showed no significant trend in this window.
- Over 1980-2025 (raw only), pH increased by +0.0051 units per year (p < 0.001), alkalinity by +0.16 mg/L per year (p = 0.035) and total phosphorus by +0.0011 mg/L per year (p = 0.005); nitrate decreased by -0.004 mg N/L per year (p = 0.032) and suspended sediment by -2.4 mg/L per year (p < 0.001).
- Belle Chasse (lower coverage, about 70 percent of quarter-years) showed decreasing pH (-0.010 per year, p < 0.001) and raw nitrate (-0.015 mg N/L per year, p = 0.011) and increasing water temperature (+0.077 deg C per year, p = 0.023).

**Machine learning (time-blocked cross-validation).**

- Nitrate (322 samples): random forest R2 = 0.28 (RMSE 0.44 mg N/L), monthly climatology 0.23, gradient boosting 0.20, linear baseline 0.13. On the 2020-2026 forward test (93 samples) random forest reached R2 = 0.23 and all models over-predicted by +0.20 to +0.30 mg N/L, consistent with recent nitrate being lower than the 2004-2019 relations imply.
- DOC (320 samples): the linear baseline did best, R2 = 0.24 (RMSE 0.42 mg/L), ahead of random forest (0.17), gradient boosting (0.14) and climatology (0.12). On the forward test linear R2 was 0.19 and gradient boosting was below zero (-0.21).
- Seasonal timing (day-of-year terms) was the most important predictor of nitrate, followed by 90-day antecedent discharge. For DOC, water temperature and 7-day antecedent discharge mattered most.
- In short, hydrology and season explain only about a quarter of sample-to-sample variance in nitrate and DOC at this site, and the tree models gave little or no gain over simple baselines.

**Carbonate system (428 samples, 1997-2025).** Computed pCO2 had a median of 2,082 uatm (interquartile range 1,446 to 2,965), about 5.2 times the approximate atmospheric value; every sample was supersaturated. Monthly medians peaked in May and June (2,809 and 2,897 uatm) and were lowest in January (1,361) and October (1,453). There was no significant trend (p = 0.26). The median Monte Carlo 90 percent half-width from measurement uncertainty was 19 percent of the estimate.

**Box model.** 200 of 20,000 parameter sets were accepted (calibration RMSE at most 13.8 mg/L). The ensemble median gave NSE = 0.54 (RMSE 12.3 mg/L) in calibration and NSE = 0.50 (RMSE 13.3 mg/L, bias -2.6 mg/L) in validation, better than a power-law concentration-discharge rating (NSE 0.43 and 0.44) and a constant mean (RMSE 18.1 and 19.0). The slow-flow alkalinity A_b (median 172, 90 percent range 132 to 240 mg/L) and quick-flow alkalinity A_q (41, range 12 to 74 mg/L) were constrained by the data; the filter constant a and residence time tau were not. Mean bicarbonate-carbon export for water years 2005-2025 was 12.8 Tg C per year (mean of annual medians; mean annual 90 percent band 12.2 to 13.3), ranging from 7.4 (2006) to 21.5 Tg C per year (2019).

**Loads, water years 2005-2025 (rating regression, mean and range of annual values).** Nitrate plus nitrite 688 Gg N per year (413 to 1,011); DOC 1,830 Gg C per year (977 to 3,244); total phosphorus 113 Gg P per year (60 to 176); suspended sediment 71 Tg per year (38 to 103); bicarbonate carbon 12.7 Tg C per year (7.9 to 19.8). The regression and box-model bicarbonate loads agreed closely year to year (correlation 0.99, mean difference under 0.1 percent, largest annual difference 9 percent), which is expected because both are driven by the same discharge record. The 2019 flood year had the largest loads of every constituent.

## Caveats and limitations

- **Sampling design.** Sampling frequency, methods and laboratories changed over seven decades (for example, unfiltered ANC before 2004 versus filtered alkalinity after 1996, and lab versus field pH). The method priority keeps the record consistent where possible, but method changes can produce step changes that a trend test will read as trend, especially in the 1980-2025 window.
- **Statistics.** About 50 trend tests were run at alpha 0.05, so two or three significant results are expected by chance. The seasonal Mann-Kendall test here does not correct for serial correlation between years. Sen slopes of exactly zero for pH and DOC reflect values reported to 0.1 units, which produces many ties.
- **Flow adjustment.** LOWESS against same-day discharge does not capture hysteresis or lagged basin responses; flow-adjusting water temperature is not physically meaningful and is reported only for completeness.
- **Discharge.** Daily discharge at Baton Rouge starts in 2004, so the machine-learning, box-model and load analyses cover 2004-2026 only. Baton Rouge flow is used for St. Francisville, about 60 river km upstream, with no major tributary in between. Both sites are downstream of the Old River Control Structure, so loads describe the Mississippi main stem only and exclude flow diverted to the Atchafalaya River. Water year 2026 and recent discharge values are provisional.
- **Machine learning.** Sample sizes are small (about 320) and skill is modest. Permutation importance is unstable across blocks (see `sd_across_blocks`) and correlated predictors share importance, so the rankings describe these models, not causation.
- **Carbonate calculation.** pCO2 from pH and alkalinity is sensitive to pH error (a 0.05 pH error changes pCO2 by about 12 percent) and can be biased high where organic acids contribute to alkalinity; this bias is likely small here because alkalinity is high (median 106 mg/L as CaCO3), but it was not tested. The atmospheric reference is an approximation.
- **Box model.** The model is a two-end-member simplification with constant end-member chemistry; it ignores carbonate weathering dynamics, in-river calcite precipitation, CO2 evasion and seasonal shifts in source areas. The accepted-parameter band covered only 50 percent (calibration) and 44 percent (validation) of observations because it reflects parameter uncertainty only, not structural or measurement error. Accepting the best 1 percent is an informal (GLUE-like) rejection rule, not a formal Bayesian posterior.
- **Loads.** Bootstrap ranges reflect regression-coefficient uncertainty only and are narrower than the true uncertainty. Bicarbonate carbon is approximated from alkalinity and is not the same as total DIC (dissolved CO2 adds a few percent at pH 7.8). The rating regression explains 22 percent (DOC) to 56 percent (alkalinity) of the variance in ln concentration.
- These results have not been peer reviewed.

## Related published work (context only)

The questions here relate to the author's published research on carbon in the Mississippi-Atchafalaya system and coastal Louisiana. None of the data or results from these papers are used in this repository.

- Dristi, A., and Xu, Y. J. (2023). *Water*, 15(14), 2621.
- Dristi, A., and Xu, Y. J. (2025). *Water*, 17(17), 2589.
- Dristi, A., et al. (2025). *Estuarine, Coastal and Shelf Science*, 324, 109465.

## Selected method references

- Eckhardt, K. (2005). How to construct recursive digital filters for baseflow separation. *Hydrological Processes*, 19, 507-515.
- Hirsch, R. M., Slack, J. R., and Smith, R. A. (1982). Techniques of trend analysis for monthly water quality data. *Water Resources Research*, 18, 107-121.
- Plummer, L. N., and Busenberg, E. (1982). The solubilities of calcite, aragonite and vaterite in CO2-H2O solutions between 0 and 90 C. *Geochimica et Cosmochimica Acta*, 46, 1011-1040.
- Runkel, R. L., Crawford, C. G., and Cohn, T. A. (2004). Load Estimator (LOADEST). USGS Techniques and Methods 4-A5.
- Weiss, R. F. (1974). Carbon dioxide in water and seawater: the solubility of a non-ideal gas. *Marine Chemistry*, 2, 203-215.

## License

Code: MIT License (see `LICENSE`). Data in `data/clean/` are derived from public-domain USGS records; please cite the USGS and the Water Quality Portal when reusing them.

Contact: anamikadristi@gmail.com
