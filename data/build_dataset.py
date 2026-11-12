#!/usr/bin/env python3
"""Build the QSIP dataset: real weather (Open-Meteo archive) + FAO-56 crop/soil tables
-> weekly crop water demand, and QAOA-ready allocation instances.

Usage:  python build_dataset.py            # downloads real weather (needs internet)
        python build_dataset.py --selftest # pipeline check with dummy weather (NOT real data)
"""
import sys, time, json, urllib.request, urllib.parse
import numpy as np, pandas as pd

START, END = "2020-01-01", "2024-12-31"
PLANT_YEARS = [2021, 2022, 2023]          # planting dates: 1st of every month
FIELD_HA = 1.0                            # demand is computed per hectare
N_INSTANCES, CROPS_PER_INSTANCE, SEED = 3000, 4, 42
RAIN_EFF = 0.8; RAIN_MIN_MM = 5.0         # effective rainfall = 0.8*P if P>5 mm/day else 0

def fetch(lat, lon):
    q = urllib.parse.urlencode(dict(latitude=lat, longitude=lon, start_date=START, end_date=END,
        daily="et0_fao_evapotranspiration,precipitation_sum,temperature_2m_max,temperature_2m_min",
        timezone="auto"))
    url = "https://archive-api.open-meteo.com/v1/archive?" + q
    for k in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                d = json.load(r)["daily"]
            return pd.DataFrame({"date": pd.to_datetime(d["time"]), "et0_mm": d["et0_fao_evapotranspiration"],
                "rain_mm": d["precipitation_sum"], "tmax_c": d["temperature_2m_max"], "tmin_c": d["temperature_2m_min"]})
        except Exception as e:
            print("retry", k, e); time.sleep(5 * (k + 1))
    raise SystemExit(f"could not download weather for {lat},{lon}")

def dummy(lat, lon):  # selftest only
    rng = np.random.default_rng(int(abs(lat * lon)))
    d = pd.date_range(START, END)
    et = 4 + 2 * np.sin((np.asarray(d.dayofyear) - 80) / 365 * 2 * np.pi) + rng.normal(0, .3, len(d))
    return pd.DataFrame({"date": d, "et0_mm": et.clip(0.5), "rain_mm": rng.gamma(.3, 8, len(d)),
                         "tmax_c": 30 + 0 * et, "tmin_c": 20 + 0 * et})

def kc_curve(c):
    i, dv, m, l = c.len_ini_d, c.len_dev_d, c.len_mid_d, c.len_late_d
    kc, stage = [], []
    for day in range(i + dv + m + l):
        if day < i: kc.append(c.kc_ini); stage.append("initial")
        elif day < i + dv: kc.append(c.kc_ini + (c.kc_mid - c.kc_ini) * (day - i) / dv); stage.append("development")
        elif day < i + dv + m: kc.append(c.kc_mid); stage.append("mid")
        else: kc.append(c.kc_mid + (c.kc_end - c.kc_mid) * (day - i - dv - m) / l); stage.append("late")
    return np.array(kc), np.array(stage)

def main():
    selftest = "--selftest" in sys.argv
    crops = pd.read_csv("crop_reference.csv"); locs = pd.read_csv("locations.csv")
    soils = pd.read_csv("soil_reference.csv"); eff = pd.read_csv("efficiency_reference.csv")
    wx_all, weekly = [], []
    for L in locs.itertuples():
        w = (dummy if selftest else fetch)(L.latitude, L.longitude).set_index("date")
        w = w.fillna({"rain_mm": 0}).interpolate()
        wx = w.reset_index(); wx.insert(0, "location", L.location); wx_all.append(wx)
        w["rain_eff_mm"] = np.where(w.rain_mm > RAIN_MIN_MM, RAIN_EFF * w.rain_mm, 0.0)
        for c in crops.itertuples():
            kc, stg = kc_curve(c); n = len(kc)
            for y in PLANT_YEARS:
                for mth in range(1, 13):
                    p = pd.Timestamp(y, mth, 1); days = pd.date_range(p, periods=n)
                    if days[-1] > w.index[-1]: continue
                    s = w.loc[days]
                    etc = kc * s.et0_mm.values; pe = s.rain_eff_mm.values
                    df = pd.DataFrame({"week": np.arange(n) // 7, "stage": stg, "kc": kc, "et0": s.et0_mm.values,
                                       "etc": etc, "rain": s.rain_mm.values, "pe": np.minimum(pe, etc)})
                    g = df.groupby("week").agg(stage=("stage", lambda x: x.mode()[0]), kc_mean=("kc", "mean"),
                        et0_mm=("et0", "sum"), etc_mm=("etc", "sum"), rain_mm=("rain", "sum"), eff_rain_mm=("pe", "sum")).reset_index()
                    g["net_irrig_mm"] = (g.etc_mm - g.eff_rain_mm).clip(lower=0)
                    g.insert(0, "planting_date", p.date()); g.insert(0, "crop", c.crop); g.insert(0, "location", L.location)
                    g["week_start"] = (p + pd.to_timedelta(g.week * 7, unit="D")).dt.date
                    weekly.append(g)
    pd.concat(wx_all).to_csv("weather_daily.csv", index=False)
    W = pd.concat(weekly, ignore_index=True)
    W.round(3).to_csv("weekly_crop_water_demand.csv", index=False)
    print("weekly rows:", len(W))

    # QAOA-ready allocation instances (scenario generation: random crop mix, soil, method, supply)
    rng = np.random.default_rng(SEED); rows = []
    sw = crops.set_index("crop")
    W["stage_w"] = [sw.loc[c, f"stage_weight_{ 'ini' if s=='initial' else 'dev' if s=='development' else s}"] for c, s in zip(W.crop, W.stage)]
    Wg = W[W.net_irrig_mm > 0]
    for k in range(N_INSTANCES):
        loc = rng.choice(locs.location); pool = Wg[Wg.location == loc]
        picks = []
        for cr in rng.choice(crops.crop, CROPS_PER_INSTANCE, replace=False):
            sub = pool[pool.crop == cr]
            if len(sub): picks.append(sub.iloc[rng.integers(len(sub))])
        if len(picks) < 3: continue
        meth = eff.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
        soil = rng.choice(soils.soil_type); area = rng.choice([0.5, 1, 2, 3, 5])
        D = [round(r.net_irrig_mm * area * 10 / meth.application_efficiency, 1) for r in picks]  # m3 (1 mm on 1 ha = 10 m3)
        Wsup = round(sum(D) * rng.uniform(0.4, 0.9), 1)
        for r, d in zip(picks, D):
            rows.append(dict(instance_id=k, location=loc, soil_type=soil, irrigation_method=meth.irrigation_method,
                efficiency=meth.application_efficiency, field_area_ha=area, crop=r.crop, stage=r.stage,
                week_start=r.week_start, net_irrig_mm=round(r.net_irrig_mm, 2), demand_m3=d,
                priority_weight=r.stage_w, supply_m3=Wsup))
    pd.DataFrame(rows).to_csv("allocation_instances.csv", index=False)
    print("instances:", len(rows) // CROPS_PER_INSTANCE, "(approx)")
main()
