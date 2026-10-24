"""Shared paths, constants and helper functions."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"
for _d in (CLEAN, TABLES, FIGS):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
CFS_TO_CMS = 0.0283168466

SITES = {
    "07373420": "St. Francisville",
    "07374000": "Baton Rouge",
    "07374525": "Belle Chasse",
    "07295100": "Tarbert Landing",
}
MAIN_SITE = "07373420"      # longest carbon and nutrient record
Q_SITE_FOR = {"07373420": "07374000", "07374000": "07374000",
              "07374525": "07374525", "07295100": None}

# column name, units, label for figures
VARS = {
    "DOC": ("mg/L", "Dissolved organic carbon"),
    "alk": ("mg/L as CaCO3", "Alkalinity / ANC"),
    "pH": ("units", "pH"),
    "SC": ("uS/cm", "Specific conductance"),
    "temp": ("deg C", "Water temperature"),
    "NO3": ("mg/L as N", "Nitrate plus nitrite"),
    "TP": ("mg/L as P", "Total phosphorus"),
    "SSC": ("mg/L", "Suspended sediment"),
}


def set_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3})
    return plt


def load_samples(site):
    f = CLEAN / f"wq_samples_{site}.csv"
    d = pd.read_csv(f, parse_dates=["date"], dtype={"site": str, "alk_src": str,
                                                         "pH_src": str, "SC_src": str})
    return d


def load_daily_q(site):
    d = pd.read_csv(CLEAN / f"daily_q_{site}.csv", parse_dates=["date"])
    return d.set_index("date")["Q_m3s"].asfreq("D")


def attach_q(samples, site):
    """Discharge for each sample: instantaneous Q measured with the sample if
    present, otherwise the daily mean Q of the paired gauge on that date."""
    d = samples.copy()
    d["Q_m3s"] = d["Qinst_m3s"]
    d["Q_source"] = np.where(d["Q_m3s"].notna(), "instantaneous", "")
    qs = Q_SITE_FOR.get(site)
    if qs is not None:
        q = load_daily_q(qs)
        fill = d["date"].map(q)
        m = d["Q_m3s"].isna() & fill.notna()
        d.loc[m, "Q_m3s"] = fill[m]
        d.loc[m, "Q_source"] = f"daily_{qs}"
    return d


# ---------------------------------------------------------------------------
# Carbonate chemistry for fresh water (salinity ~0)
# ---------------------------------------------------------------------------
def k0_weiss(temp_c):
    """CO2 solubility K0 (mol L-1 atm-1), Weiss (1974), S = 0."""
    T = np.asarray(temp_c, float) + 273.15
    lnk0 = -58.0931 + 90.5069 * (100 / T) + 22.2940 * np.log(T / 100)
    return np.exp(lnk0)


def k1_k2_fresh(temp_c):
    """First and second dissociation constants of carbonic acid in fresh
    water, Plummer and Busenberg (1982). Returns (K1, K2) in mol/L."""
    T = np.asarray(temp_c, float) + 273.15
    logk1 = -356.3094 - 0.06091964 * T + 21834.37 / T + 126.8339 * np.log10(T) - 1684915 / T**2
    logk2 = -107.8871 - 0.03252849 * T + 5151.79 / T + 38.92561 * np.log10(T) - 563713.9 / T**2
    return 10**logk1, 10**logk2


def kw_fresh(temp_c):
    T = np.asarray(temp_c, float) + 273.15
    return 10 ** (-(4470.99 / T - 6.0875 + 0.01706 * T))


def carbonate_from_ph_alk(pH, alk_mgL_caco3, temp_c):
    """Speciate the carbonate system from pH, alkalinity and temperature,
    treating alkalinity as carbonate alkalinity plus OH- minus H+ and using
    concentrations in place of activities (fresh water, low ionic strength).
    Returns pCO2 (uatm), DIC (umol/L) and CO2* (umol/L)."""
    pH = np.asarray(pH, float)
    H = 10 ** (-pH)
    K1, K2 = k1_k2_fresh(temp_c)
    Kw = kw_fresh(temp_c)
    alk_eq = np.asarray(alk_mgL_caco3, float) / 50.04 * 1e-3   # eq/L
    carb_alk = alk_eq - Kw / H + H
    # carbonate alkalinity = DIC * (K1 H + 2 K1 K2) / (H^2 + K1 H + K1 K2)
    denom = H**2 + K1 * H + K1 * K2
    dic = carb_alk * denom / (K1 * H + 2 * K1 * K2)
    co2 = dic * H**2 / denom
    pco2 = co2 / k0_weiss(temp_c) * 1e6
    return pco2, dic * 1e6, co2 * 1e6


def water_year(dates):
    dates = pd.to_datetime(dates)
    return dates.dt.year + (dates.dt.month >= 10).astype(int)
