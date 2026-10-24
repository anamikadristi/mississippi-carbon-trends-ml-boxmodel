"""
00_download_data.py
Download public USGS water-quality samples and daily discharge for the
Lower Mississippi River.

Sources (no API key is used or needed):
  * Water Quality Portal, WQX 3.0 Result service (USGS NWIS samples):
      https://www.waterqualitydata.us/wqx3/Result/search
  * USGS Water Data OGC API, daily values collection:
      https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items

Raw files are written to a separate directory (default data/raw, which is
listed in .gitignore). Nothing in that directory is executed.

Usage:
  python -I scripts/00_download_data.py --raw-dir data/raw
"""
import argparse
import datetime as dt
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]

# Water-quality sampling sites (USGS station numbers)
WQ_SITES = {
    "07373420": "Mississippi River near St. Francisville, LA",
    "07374000": "Mississippi River at Baton Rouge, LA",
    "07374525": "Mississippi River at Belle Chasse, LA",
    "07295100": "Mississippi River at Tarbert Landing, MS",
}
# Sites with a USGS daily mean discharge record (00060, statistic 00003)
Q_SITES = ["07374000", "07374525", "07295100", "07373420"]

PCODES = [
    "00681",  # dissolved organic carbon, filtered, mg/L
    "00410",  # ANC, unfiltered, fixed endpoint, field, mg/L as CaCO3
    "90410",  # ANC, unfiltered, fixed endpoint, laboratory
    "39086",  # alkalinity, filtered, inflection point, field
    "29801",  # alkalinity, filtered, fixed endpoint, laboratory
    "00400",  # pH, field
    "00403",  # pH, laboratory
    "00095",  # specific conductance, field, uS/cm at 25 C
    "90095",  # specific conductance, laboratory
    "00010",  # water temperature, deg C
    "00631",  # nitrate plus nitrite, filtered, mg/L as N
    "00665",  # total phosphorus, unfiltered, mg/L as P
    "80154",  # suspended sediment concentration, mg/L
    "00061",  # instantaneous discharge at time of sampling, ft3/s
]

WQP_URL = "https://www.waterqualitydata.us/wqx3/Result/search"
DV_URL = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items"


def get_with_retry(url, params=None, timeout=600, tries=4):
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            print(f"  HTTP {r.status_code}; retrying")
        except requests.RequestException as e:
            print(f"  {type(e).__name__}; retrying")
        time.sleep(10 * (k + 1))
    raise RuntimeError(f"failed after {tries} tries: {url}")


def download_wq(site, raw_dir):
    params = [("siteid", f"USGS-{site}"), ("mimeType", "csv"), ("zip", "no"),
              ("dataProfile", "basicPhysChem")] + [("pCode", p) for p in PCODES]
    r = get_with_retry(WQP_URL, params=params)
    out = raw_dir / f"wqp_wqx3_{site}.csv"
    out.write_bytes(r.content)
    n = max(r.text.count("\n") - 1, 0)
    print(f"  WQP {site}: {n} result rows -> {out.name}")
    return n


def download_daily_q(site, raw_dir):
    params = {"monitoring_location_id": f"USGS-{site}", "parameter_code": "00060",
              "statistic_id": "00003", "limit": 10000, "f": "json"}
    url, rows = DV_URL, []
    while url:
        r = get_with_retry(url, params=params)
        js = r.json()
        for f in js.get("features", []):
            p = f["properties"]
            rows.append((p["time"], p["value"], p.get("approval_status", ""),
                         p.get("qualifier") or ""))
        nxt = [l["href"] for l in js.get("links", []) if l.get("rel") == "next"]
        url, params = (nxt[0], None) if nxt and js.get("features") else (None, None)
        time.sleep(1)
    out = raw_dir / f"usgs_daily_q_{site}.csv"
    with out.open("w") as fh:
        fh.write("date,Q_cfs,approval_status,qualifier\n")
        for t, v, a, q in sorted(rows):
            q = "|".join(q) if isinstance(q, list) else str(q)
            fh.write(f"{t},{v},{a},{q}\n")
    print(f"  daily Q {site}: {len(rows)} days -> {out.name}")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    a = ap.parse_args()
    raw_dir = Path(a.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    log = {"access_date_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"),
           "wqp_url": WQP_URL, "dv_url": DV_URL, "pcodes": PCODES,
           "wq_rows": {}, "daily_q_days": {}}
    for s in WQ_SITES:
        log["wq_rows"][s] = download_wq(s, raw_dir)
    for s in Q_SITES:
        log["daily_q_days"][s] = download_daily_q(s, raw_dir)
    (raw_dir / "download_log.json").write_text(json.dumps(log, indent=2))
    print("done:", json.dumps(log["daily_q_days"]))


if __name__ == "__main__":
    main()
