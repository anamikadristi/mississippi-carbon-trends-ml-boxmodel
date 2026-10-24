"""
03_ml_model.py
Predict nitrate (NO3, mg/L as N) and dissolved organic carbon (DOC, mg/L) at
St. Francisville from hydrologic and seasonal predictors, comparing a linear
baseline, a random forest and gradient boosting.

Period: samples from 2004-03-17 onward, when daily mean discharge at Baton
Rouge (USGS 07374000, about 60 river km downstream with no major tributary
in between) is available for every sample date.

Predictors (no information from the target is used):
  lnQ        ln daily mean discharge on the sample day
  lnQ_7d     ln mean discharge over the previous 7 days
  lnQ_30d    ln mean discharge over the previous 30 days
  lnQ_90d    ln mean discharge over the previous 90 days
  dlnQ_7d    lnQ minus ln discharge 7 days earlier (rising vs falling limb)
  sin_doy, cos_doy  seasonal harmonics
  temp       water temperature measured with the sample (deg C)
  trend_yr   decimal year (lets models absorb slow drift; reported both
             with and without it)

Validation:
  * leave-one-block-out cross-validation with contiguous 3-year blocks, so
    no fold is scored on years adjacent to its training data within a block
  * a forward holdout: train on 2004-2019, test on 2020-2026
Skill: R2 (out of sample, equals Nash-Sutcliffe efficiency), RMSE, MAE, bias.
Variable importance: permutation importance (increase in RMSE when a
predictor is shuffled) computed on each held-out block and averaged,
weighted by block size.

Outputs: results/tables/ml_*.csv, results/figures/ml_*.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIGS, MAIN_SITE, SEED, TABLES, load_daily_q, load_samples, set_style  # noqa: E402

plt = set_style()
BASE_FEATS = ["lnQ", "lnQ_7d", "lnQ_30d", "lnQ_90d", "dlnQ_7d", "sin_doy", "cos_doy", "temp"]
TARGETS = {"NO3": "Nitrate plus nitrite (mg/L as N)", "DOC": "Dissolved organic carbon (mg/L)"}


def build_features():
    q = load_daily_q("07374000")
    lnq = np.log(q)
    f = pd.DataFrame({
        "lnQ": lnq,
        "lnQ_7d": np.log(q.shift(1).rolling(7, min_periods=6).mean()),
        "lnQ_30d": np.log(q.shift(1).rolling(30, min_periods=25).mean()),
        "lnQ_90d": np.log(q.shift(1).rolling(90, min_periods=75).mean()),
        "dlnQ_7d": lnq - lnq.shift(7),
    })
    s = load_samples(MAIN_SITE).set_index("date")
    d = s.join(f, how="inner")
    doy = d.index.day_of_year
    d["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    d["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    d["trend_yr"] = d.index.year + (doy - 0.5) / 365.25
    return d


def models():
    return {
        "linear": make_pipeline(StandardScaler(), LinearRegression()),
        "random_forest": RandomForestRegressor(n_estimators=500, min_samples_leaf=3,
                                               max_features=0.5, random_state=SEED, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                       max_depth=3, subsample=0.8,
                                                       random_state=SEED),
    }


def metrics(y, p):
    return {"n": len(y), "R2": r2_score(y, p), "RMSE": mean_squared_error(y, p) ** 0.5,
            "MAE": mean_absolute_error(y, p), "bias": float(np.mean(p - y))}


def main():
    d = build_features()
    rows, imp_rows, preds_all = [], [], []
    for target in TARGETS:
        for fs_name, feats in [("hydro_season_temp", BASE_FEATS),
                               ("plus_trend_year", BASE_FEATS + ["trend_yr"])]:
            dd = d.dropna(subset=feats + [target]).copy()
            y = dd[target].values
            X = dd[feats].values
            yrs = dd.index.year.values
            block = (yrs - 2004) // 3
            # climatology baseline: monthly mean of training data
            month = dd.index.month.values
            for name in ["climatology"] + list(models()):
                pred = np.full(len(y), np.nan)
                for b in np.unique(block):
                    te, tr = block == b, block != b
                    if name == "climatology":
                        clim = pd.Series(y[tr]).groupby(month[tr]).mean()
                        pred[te] = pd.Series(month[te]).map(clim).fillna(y[tr].mean()).values
                        continue
                    m = models()[name].fit(X[tr], y[tr])
                    pred[te] = m.predict(X[te])
                    if fs_name == "hydro_season_temp" and te.sum() >= 10:
                        pi = permutation_importance(m, X[te], y[te], n_repeats=20,
                                                    random_state=SEED,
                                                    scoring="neg_root_mean_squared_error")
                        for k, ft in enumerate(feats):
                            imp_rows.append({"target": target, "model": name, "block": int(b),
                                             "feature": ft, "importance": pi.importances_mean[k],
                                             "weight": int(te.sum())})
                r = {"target": target, "features": fs_name, "model": name,
                     "validation": "leave-one-3yr-block-out", **metrics(y, pred)}
                rows.append(r)
                if fs_name == "hydro_season_temp":
                    preds_all.append(pd.DataFrame({"date": dd.index, "target": target,
                                                   "model": name, "obs": y, "pred": pred}))
                # forward holdout
                tr, te = yrs <= 2019, yrs >= 2020
                if name == "climatology":
                    clim = pd.Series(y[tr]).groupby(month[tr]).mean()
                    p2 = pd.Series(month[te]).map(clim).values
                else:
                    p2 = models()[name].fit(X[tr], y[tr]).predict(X[te])
                rows.append({"target": target, "features": fs_name, "model": name,
                             "validation": "train 2004-2019, test 2020-2026",
                             **metrics(y[te], p2)})
    res = pd.DataFrame(rows)
    res.to_csv(TABLES / "ml_skill.csv", index=False, float_format="%.4f")
    print(res.to_string(index=False))

    imp = pd.DataFrame(imp_rows)
    impw = (imp.assign(wi=imp.importance * imp.weight)
            .groupby(["target", "model", "feature"])
            .apply(lambda g: pd.Series({"perm_importance_RMSE_increase": g.wi.sum() / g.weight.sum(),
                                        "sd_across_blocks": g.importance.std()}),
                   include_groups=False)
            .reset_index())
    impw.to_csv(TABLES / "ml_permutation_importance.csv", index=False, float_format="%.4f")
    print(impw[impw.model != "linear"].sort_values(["target", "model", "perm_importance_RMSE_increase"],
                                                   ascending=[True, True, False]).to_string(index=False))
    pa = pd.concat(preds_all)
    pa.to_csv(TABLES / "ml_cv_predictions.csv", index=False, float_format="%.4f",
              date_format="%Y-%m-%d")

    # Figures: observed vs CV prediction time series, and importance bars
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, t in zip(axs, TARGETS):
        p = pa[pa.target == t]
        o = p[p.model == "linear"]
        ax.plot(o.date, o.obs, "k.", ms=3, label="observed")
        for mname, c in [("linear", "C0"), ("gradient_boosting", "C3")]:
            q = p[p.model == mname]
            sk = res[(res.target == t) & (res.model == mname) & (res.features == "hydro_season_temp")
                     & (res.validation == "leave-one-3yr-block-out")].iloc[0]
            ax.plot(q.date, q.pred, "-", lw=0.9, color=c, alpha=0.85,
                    label=f"{mname.replace('_', ' ')} (CV R2={sk.R2:.2f}, RMSE={sk.RMSE:.3f})")
        ax.set_ylabel(TARGETS[t], fontsize=8)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    axs[0].set_title("St. Francisville: out-of-sample predictions from leave-one-3-year-block-out CV", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "ml_cv_predictions.png")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, t in zip(axs, TARGETS):
        g = impw[(impw.target == t) & (impw.model.isin(["random_forest", "gradient_boosting"]))]
        piv = g.pivot(index="feature", columns="model", values="perm_importance_RMSE_increase")
        piv = piv.loc[piv.mean(axis=1).sort_values().index]
        piv.plot.barh(ax=ax, width=0.75, legend=(t == "NO3"))
        ax.set_xlabel("Mean increase in held-out RMSE (mg/L) when feature is permuted")
        ax.set_title(t, fontsize=10)
        ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIGS / "ml_permutation_importance.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
