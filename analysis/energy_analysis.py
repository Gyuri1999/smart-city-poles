#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Energy-saving analysis for the Smart City Poles experiment.
Ανάλυση εξοικονόμησης ενέργειας για το πείραμα των έξυπνων κολονών.

Pipeline / Ροή:
  raw data (InfluxDB or CSV)  ->  measurement blocks  ->  statistics  ->  model  ->  figures

  1. Power of the luminaires (Shelly EM), scenario (input_select.kolones_mode) and
     detections (counter.kolones_passes_night) are read from InfluxDB or from CSV.
  2. Every period with the lights on and one scenario active becomes a "block";
     its mean power = energy / duration (step-hold integration of the samples).
  3. Baseline vs adaptive: saving per traffic rate, with bootstrap 95% CIs.
  4. Analytical model (Poisson arrivals) -> predicted saving, compared with the
     measurements.
  5. Figures for the paper + CSV/Markdown tables.

Modes / Τρόποι χρήσης (see analysis/README.md):
  python3 analysis/energy_analysis.py influx --host http://homeassistant.local:8086 \\
          --power-entity shellyem_xxxxxxxxxxxx_channel_1_power --start 2026-09-14
  python3 analysis/energy_analysis.py csv  --dir analysis/output/raw
  python3 analysis/energy_analysis.py demo --nights 60

Requires: Python 3.9+, numpy, matplotlib  (pip install numpy matplotlib)
"""
import argparse
import base64
import bisect
import csv
import datetime as dt
import getpass
import json
import math
import os
import sys
import urllib.parse
import urllib.request

import numpy as np

# Measured on the testbed (README, energy experiment) - total of the 3 luminaires, W.
DEFAULT_STANDBY = 3.19
DEFAULT_LEVELS = (16.4, 19.1, 21.0)   # base 75 %, wave 88 %, full 100 %


# =============================================================================
#  1. DATA LOADING
# =============================================================================
def parse_time(s):
    """Epoch seconds, or ISO-8601 (with or without timezone; naive = UTC)."""
    s = str(s).strip()
    try:
        return float(s)
    except ValueError:
        pass
    t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.timestamp()


def iso(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%d %H:%M")


def read_series_csv(path, numeric=True):
    """CSV with columns time,value (or time,state). Returns sorted (times, values)."""
    if not os.path.exists(path):
        return [], []
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            v = r.get("value", r.get("state"))
            if v is None or v == "":
                continue
            if numeric:
                try:
                    v = float(v)
                except ValueError:
                    continue      # "unavailable", "unknown" ...
            rows.append((parse_time(r["time"]), v))
    rows.sort(key=lambda x: x[0])
    return [r[0] for r in rows], [r[1] for r in rows]


def write_series_csv(path, times, values, col="value"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", col])
        for t, v in zip(times, values):
            w.writerow([f"{t:.3f}", v])


class Influx:
    """Minimal InfluxDB v1 client (HTTP /query, no extra dependencies)."""

    def __init__(self, host, db, user, password):
        self.host, self.db, self.user, self.password = host.rstrip("/"), db, user, password

    def query(self, q):
        url = f"{self.host}/query?" + urllib.parse.urlencode({"db": self.db, "q": q, "epoch": "s"})
        req = urllib.request.Request(url)
        if self.user:
            token = base64.b64encode(f"{self.user}:{self.password}".encode()).decode()
            req.add_header("Authorization", "Basic " + token)
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        res = data["results"][0]
        if "error" in res:
            raise RuntimeError(res["error"])
        if not res.get("series"):
            return []
        return res["series"][0]["values"]

    def series(self, measurement, field, entity, start, end, numeric=True, chunk_days=7):
        """Fetch one entity in weekly chunks (a Shelly writes ~10k points a day)."""
        times, values = [], []
        t = start
        while t < end:
            t2 = min(end, t + chunk_days * 86400)
            q = (f'SELECT "{field}" FROM "{measurement}" WHERE "entity_id" = \'{entity}\' '
                 f"AND time >= {int(t)}s AND time < {int(t2)}s")
            for ts, v in self.query(q):
                if v is None:
                    continue
                if numeric:
                    try:
                        v = float(v)
                    except (TypeError, ValueError):
                        continue
                times.append(float(ts))
                values.append(v)
            t = t2
        return times, values


def load_influx(a):
    pw = os.environ.get(a.password_env) or (getpass.getpass("InfluxDB password: ") if a.user else "")
    cli = Influx(a.host, a.db, a.user, pw)
    start = parse_time(a.start)
    end = parse_time(a.end) if a.end else dt.datetime.now(dt.timezone.utc).timestamp()
    print(f"Λήψη δεδομένων από {a.host} ({iso(start)} -> {iso(end)}) ...")
    power = cli.series(a.power_measurement, "value", a.power_entity, start, end)
    mode = cli.series("state", "state", a.mode_entity, start, end, numeric=False)
    passes = cli.series("state", "value", a.passes_entity, start, end) if a.passes_entity else ([], [])
    print(f"  ισχύς: {len(power[0])} δείγματα · σενάριο: {len(mode[0])} αλλαγές · "
          f"ανιχνεύσεις: {len(passes[0])} τιμές")
    # Keep a copy of the raw data: the analysis can then be re-run (and published) from CSV.
    raw = os.path.join(a.out, "raw")
    write_series_csv(os.path.join(raw, "power.csv"), *power)
    write_series_csv(os.path.join(raw, "mode.csv"), *mode, col="state")
    write_series_csv(os.path.join(raw, "passes.csv"), *passes)
    print(f"  τα ακατέργαστα δεδομένα αποθηκεύτηκαν στο {raw}/")
    return power, mode, passes


def load_csv(a):
    power = read_series_csv(os.path.join(a.dir, "power.csv"))
    mode = read_series_csv(os.path.join(a.dir, "mode.csv"), numeric=False)
    passes = read_series_csv(os.path.join(a.dir, "passes.csv"))
    if not power[0]:
        sys.exit(f"Δεν βρέθηκε {a.dir}/power.csv")
    return power, mode, passes


# =============================================================================
#  2. SIMULATOR (demo mode) - reproduces the Home Assistant automation exactly
# =============================================================================
def neighbours(n):
    return [[j for j in (i - 1, i + 1) if 0 <= j < n] for i in range(n)]


def simulate(a, rng):
    """Synthetic data with the same format as InfluxDB: alternating nights,
    Poisson traffic, lamp levels following kolones_automation.yaml."""
    n, T = a.poles, a.hold
    ps, (pb, pw, pf) = a.standby, a.levels
    lamp = {"b": ps / n + (pb - ps) / n, "w": ps / n + (pw - ps) / n,
            "f": ps / n + (pf - ps) / n, "off": ps / n}
    nb = neighbours(n)
    rates = a.rates or [0.0]
    t0 = parse_time("2026-09-14T17:00:00+00:00")
    night_s = a.night_hours * 3600
    p_t, p_v, m_t, m_v, c_t, c_v = [], [], [], [], [], []

    def sample(change_t, change_p, start, end):
        # Shelly-like readings every 10 s of the true (step) power + noise.
        ts = np.arange(start, end, 10.0)
        idx = np.searchsorted(change_t, ts, side="right") - 1
        vals = np.asarray(change_p)[idx] + rng.normal(0, a.noise, len(ts))
        p_t.extend(ts.tolist())
        p_v.extend(np.round(vals, 2).tolist())

    for k in range(a.nights):
        start = t0 + k * 86400
        end = start + night_s
        scen = "baseline" if k % 2 == 0 else "adaptive"
        r = rates[(k // 2) % len(rates)]
        m_t.append(start - 60)
        m_v.append(scen)
        # the day after the night: lights off (standby only)
        sample([end], [ps], end, start + 86400)
        if scen == "baseline":
            sample([start], [n * lamp["f"]], start, end)
            continue
        # ---- adaptive: generate detections -------------------------------
        lam = r / 3600.0
        det = []
        if lam > 0:
            if a.traffic == "walk":
                t = start + rng.exponential(1 / lam)
                while t < end:
                    order = range(n) if rng.random() < 0.5 else range(n - 1, -1, -1)
                    for j, pole in enumerate(order):
                        det.append((t + j * a.gap, pole))
                    t += rng.exponential(1 / lam)
            else:
                for pole in range(n):
                    t = start + rng.exponential(1 / lam)
                    while t < end:
                        det.append((t, pole))
                        t += rng.exponential(1 / lam)
        det = sorted(d for d in det if d[0] < end)
        # ---- automation semantics -----------------------------------------
        level = ["b"] * n
        timer = [None] * n
        ch_t, ch_p = [start], [sum(lamp[x] for x in level)]

        def expire_until(tt):
            while True:
                due = [(timer[i], i) for i in range(n) if timer[i] is not None and timer[i] <= tt]
                if not due:
                    return
                te, i = min(due)
                level[i], timer[i] = "b", None
                ch_t.append(te)
                ch_p.append(sum(lamp[x] for x in level))

        c_t.append(start)
        c_v.append(0.0)
        for cnt, (td, i) in enumerate(det, 1):
            expire_until(td)
            level[i] = "f"                              # the pole that detected -> 100 %
            for j in nb[i]:
                if level[j] == "b":                     # neighbours below 88 % -> 88 %
                    level[j] = "w"
            for j in [i] + nb[i]:
                timer[j] = td + T                       # (re)start the 60 s timers
            ch_t.append(td)
            ch_p.append(sum(lamp[x] for x in level))
            c_t.append(td)
            c_v.append(float(cnt))
        expire_until(end)
        sample(ch_t, ch_p, start, end)
    order = np.argsort(p_t)
    power = (np.asarray(p_t)[order].tolist(), np.asarray(p_v)[order].tolist())
    raw = os.path.join(a.out, "raw")
    write_series_csv(os.path.join(raw, "power.csv"), *power)
    write_series_csv(os.path.join(raw, "mode.csv"), m_t, m_v, col="state")
    write_series_csv(os.path.join(raw, "passes.csv"), c_t, c_v)
    print(f"Συνθετικά δεδομένα: {a.nights} νύχτες, κυκλοφορία «{a.traffic}», ρυθμοί {rates} -> {raw}/")
    return power, (m_t, m_v), (c_t, c_v)


# =============================================================================
#  3. BLOCKS
# =============================================================================
def step_mean(times, values, a, b):
    """Time-weighted mean of a step-held series on [a, b)."""
    i = max(bisect.bisect_right(times, a) - 1, 0)
    total, t, v = 0.0, a, values[i]
    i += 1
    while i < len(times) and times[i] < b:
        total += v * (times[i] - t)
        t, v = times[i], values[i]
        i += 1
    total += v * (b - t)
    return total / (b - a)


def counter_increase(times, values, a, b):
    """Sum of positive steps of a counter on (a, b] - resets to 0 are ignored."""
    if not times:
        return 0.0
    i = bisect.bisect_right(times, a)
    prev = values[i - 1] if i > 0 else values[0]
    inc = 0.0
    while i < len(times) and times[i] <= b:
        if values[i] > prev:
            inc += values[i] - prev
        prev = values[i]
        i += 1
    return inc


def lit_intervals(times, values, thr):
    out, start = [], None
    for t, v in zip(times, values):
        if v > thr and start is None:
            start = t
        elif v <= thr and start is not None:
            out.append((start, t))
            start = None
    if start is not None:
        out.append((start, times[-1]))
    return out


def build_blocks(a, power, mode, passes):
    pt, pv = power
    mt, mv = mode
    if a.blocks_csv:                       # manual block list overrides the automatic split
        blocks = []
        with open(a.blocks_csv, newline="") as fh:
            for r in csv.DictReader(fh):
                blocks.append({"start": parse_time(r["start"]), "end": parse_time(r["end"]),
                               "scenario": r["scenario"].strip(),
                               "nominal": float(r["rate"]) if r.get("rate") else None})
    else:
        if not mt:
            sys.exit("Δεν βρέθηκαν αλλαγές σεναρίου (mode). Έλεγξε το --mode-entity ή δώσε --blocks-csv.")
        thr = a.on_threshold if a.on_threshold else a.standby + 0.5 * (a.levels[0] - a.standby)
        trim = a.trim_min * 60
        blocks = []
        for s, e in lit_intervals(pt, pv, thr):
            cuts = [s] + [t for t in mt if s < t < e] + [e]
            for x, y in zip(cuts[:-1], cuts[1:]):
                i = bisect.bisect_right(mt, x + 1e-6) - 1
                if i < 0:
                    continue
                x, y = x + trim, y - trim
                if y - x < a.min_block_h * 3600:
                    continue
                step = a.split_h * 3600 if a.split_h else y - x
                k = int((y - x) // step) if a.split_h else 1
                for j in range(k):
                    blocks.append({"start": x + j * step, "end": x + (j + 1) * step,
                                   "scenario": str(mv[i]).strip(), "nominal": None})
    keep = {"baseline", "adaptive"}
    blocks = [b for b in blocks if b["scenario"] in keep]
    for b in blocks:
        hours = (b["end"] - b["start"]) / 3600
        b["hours"] = hours
        b["mean_w"] = step_mean(pt, pv, b["start"], b["end"])
        det = counter_increase(*passes, b["start"], b["end"])
        b["detections"] = det
        b["rate"] = det / (a.poles * hours)    # detections per pole per hour
    return blocks


def assign_rate_groups(blocks, rates):
    for b in blocks:
        r = b["nominal"] if b["nominal"] is not None else b["rate"]
        if rates:
            r = min(rates, key=lambda x: abs(x - r))
        b["group"] = round(r, 1)


# =============================================================================
#  4. MODEL
# =============================================================================
def lamp_fractions_independent(r, T, k):
    """Exact fractions of time at full / wave level for one lamp with k neighbours,
    when every pole sees independent Poisson detections at r per hour.
    A lamp leaves the base level at any detection of itself or a neighbour, stays
    up while relevant detections keep coming within T, and is at full level from
    its own first detection in that busy period (renewal-reward argument)."""
    lam = r / 3600.0
    if lam <= 0:
        return 0.0, 0.0
    mu = (k + 1) * lam
    q = 1 - math.exp(-mu * T)                         # P(next relevant detection within T)
    p = 1.0 / (k + 1)                                 # P(a relevant detection is its own)
    m_lt = 1 / mu - T * (1 - q) / q                   # E[gap | gap < T]
    a_ = (1 - p) * q
    e_wave = m_lt * a_ / (1 - a_) + T * (1 - q) * (1 - p) / (1 - a_)
    e_busy = q / (mu * (1 - q))
    f_up = q
    f_wave = f_up * e_wave / e_busy
    return f_up - f_wave, f_wave


def lamp_fractions_walk(r, T, gap, p_before, p_after):
    """Approximation for pedestrians walking the street: every pass triggers each
    pole once, `gap` seconds apart. A later neighbour restarts the timer (+gap at
    full level), an earlier neighbour pre-lights the lamp for `gap` seconds."""
    rho = r / 3600.0
    l_full = T + gap * p_after
    l_wave = gap * p_before
    f_full = 1 - math.exp(-rho * l_full)
    f_wave = math.exp(-rho * l_full) * (1 - math.exp(-rho * l_wave))
    return f_full, f_wave


def model_power(r, a, T=None):
    """Expected total power (W) of the n luminaires at r detections per pole per hour."""
    T = a.hold if T is None else T
    n = a.poles
    ps, (pb, pw, pf) = a.standby, a.levels
    nb = neighbours(n)
    total = ps
    for i in range(n):
        if a.model == "walk":
            # ends: half of the passes have no earlier (or later) neighbour
            p_b = 0.5 if len(nb[i]) == 1 else (1.0 if nb[i] else 0.0)
            ff, fw = lamp_fractions_walk(r, T, a.gap, p_b, p_b)
        else:
            ff, fw = lamp_fractions_independent(r, T, len(nb[i]))
        fb = 1 - ff - fw
        total += ((pb - ps) * fb + (pw - ps) * fw + (pf - ps) * ff) / n
    return total


def model_saving(r, a, T=None, net=False):
    p = model_power(r, a, T)
    pf, ps = a.levels[2], a.standby
    if net:
        return 1 - (p - ps) / (pf - ps)
    return 1 - p / pf


# =============================================================================
#  5. STATISTICS
# =============================================================================
def boot_mean_ci(x, rng, n_boot):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan")
    m = rng.choice(x, (n_boot, len(x))).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_saving_ci(base, adapt, rng, n_boot, standby=0.0):
    base, adapt = np.asarray(base, float), np.asarray(adapt, float)
    if len(base) < 2 or len(adapt) < 2:
        return float("nan"), float("nan")
    mb = rng.choice(base, (n_boot, len(base))).mean(axis=1) - standby
    ma = rng.choice(adapt, (n_boot, len(adapt))).mean(axis=1) - standby
    s = 1 - ma / mb
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def summarise(blocks, a, rng):
    base = [b["mean_w"] for b in blocks if b["scenario"] == "baseline"]
    if not base:
        sys.exit("Δεν βρέθηκε κανένα μπλοκ baseline - δεν γίνεται σύγκριση.")
    mb = float(np.mean(base))
    rows = [{"scenario": "baseline", "group": "", "n": len(base), "rate": float("nan"),
             "mean_w": mb, "w_lo": boot_mean_ci(base, rng, a.boot)[0],
             "w_hi": boot_mean_ci(base, rng, a.boot)[1]}]
    for g in sorted({b["group"] for b in blocks if b["scenario"] == "adaptive"}):
        ad = [b for b in blocks if b["scenario"] == "adaptive" and b["group"] == g]
        w = [b["mean_w"] for b in ad]
        rate = float(np.mean([b["rate"] if b["nominal"] is None else b["nominal"] for b in ad]))
        lo, hi = boot_mean_ci(w, rng, a.boot)
        s = 1 - np.mean(w) / mb
        s_lo, s_hi = boot_saving_ci(base, w, rng, a.boot)
        sn = 1 - (np.mean(w) - a.standby) / (mb - a.standby)
        sn_lo, sn_hi = boot_saving_ci(base, w, rng, a.boot, a.standby)
        rows.append({"scenario": "adaptive", "group": g, "n": len(w), "rate": rate,
                     "mean_w": float(np.mean(w)), "w_lo": lo, "w_hi": hi,
                     "saving": s, "s_lo": s_lo, "s_hi": s_hi,
                     "saving_net": sn, "sn_lo": sn_lo, "sn_hi": sn_hi,
                     "model": model_saving(rate, a), "model_net": model_saving(rate, a, net=True)})
    return rows


# =============================================================================
#  6. OUTPUT
# =============================================================================
def pct(x):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}"


def write_outputs(blocks, rows, a):
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "blocks.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["start_utc", "end_utc", "scenario", "hours", "mean_w", "detections",
                    "rate_per_pole_h", "group"])
        for b in blocks:
            w.writerow([iso(b["start"]), iso(b["end"]), b["scenario"], f"{b['hours']:.2f}",
                        f"{b['mean_w']:.3f}", int(b["detections"]), f"{b['rate']:.2f}", b["group"]])
    base = rows[0]
    lines = [
        "# Energy saving — summary",
        "",
        f"Baseline: {base['n']} blocks, mean power {base['mean_w']:.2f} W "
        f"(95% CI {base['w_lo']:.2f}–{base['w_hi']:.2f} W). "
        f"Model: {a.model}, hold time {a.hold:.0f} s, standby {a.standby} W, "
        f"levels {a.levels[0]}/{a.levels[1]}/{a.levels[2]} W.",
        "",
        "| Rate (det./pole/h) | Blocks | Mean power (W) | Saving (%) [95% CI] | "
        "Net saving (%) [95% CI] | Model (%) | Model − measured (pp) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows[1:]:
        diff = 100 * (r["model"] - r["saving"])
        lines.append(
            f"| {r['rate']:.1f} | {r['n']} | {r['mean_w']:.2f} | {pct(r['saving'])} "
            f"[{pct(r['s_lo'])}–{pct(r['s_hi'])}] | {pct(r['saving_net'])} "
            f"[{pct(r['sn_lo'])}–{pct(r['sn_hi'])}] | {pct(r['model'])} | {diff:+.1f} |")
    errs = [abs(r["model"] - r["saving"]) for r in rows[1:]]
    if errs:
        lines += ["", f"Model error: MAE {100 * np.mean(errs):.2f} pp, max {100 * max(errs):.2f} pp."]
    gap = (base["mean_w"] - a.levels[2]) / a.levels[2]
    if abs(gap) > 0.05:
        lines += ["", f"WARNING: measured baseline power ({base['mean_w']:.2f} W) differs "
                      f"{100 * gap:+.1f}% from the full-level power used by the model "
                      f"({a.levels[2]} W). Check --levels / --standby and the block trimming."]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(a.out, "summary.md"), "w") as fh:
        fh.write(text)
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as fh:
        keys = ["scenario", "group", "n", "rate", "mean_w", "w_lo", "w_hi", "saving", "s_lo",
                "s_hi", "saving_net", "sn_lo", "sn_hi", "model", "model_net"]
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print()
    print(text)


def make_figures(blocks, rows, a):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Το matplotlib δεν είναι εγκατεστημένο - παραλείπονται τα σχήματα "
              "(pip install matplotlib).")
        return
    grey, blue = "#8c8c8c", "#1f5fae"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    meas = rows[1:]
    r_max = max([60.0] + [r["rate"] * 1.15 for r in meas])
    rr = np.linspace(0, r_max, 300)

    # --- Fig. A: measured saving vs model --------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(rr, [100 * model_saving(x, a) for x in rr], color=grey, lw=1.6,
            label=f"Model ({a.model}, T = {a.hold:.0f} s)")
    if meas:
        x = [r["rate"] for r in meas]
        y = [100 * r["saving"] for r in meas]
        lo = [100 * (r["saving"] - r["s_lo"]) if not math.isnan(r["s_lo"]) else 0 for r in meas]
        hi = [100 * (r["s_hi"] - r["saving"]) if not math.isnan(r["s_hi"]) else 0 for r in meas]
        ax.errorbar(x, y, yerr=[lo, hi], fmt="o", color=blue, ms=5, capsize=3,
                    label="Measured (mean, 95% CI)")
    ax.set_xlabel("Detections per pole per hour")
    ax.set_ylabel("Energy saving vs. 100 % baseline (%)")
    ax.set_xlim(0, r_max)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.set_title("Adaptive lighting saving vs. traffic", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_saving_vs_rate.png"), dpi=200)
    fig.savefig(os.path.join(a.out, "fig_saving_vs_rate.pdf"))
    plt.close(fig)

    # --- Fig. B: design chart, hold time T ------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    hold_times = sorted({30, 60, 120, 300, int(a.hold)})
    for i, T in enumerate(hold_times):
        ys = [100 * model_saving(x, a, T=T) for x in rr]
        own = T == int(a.hold)
        col = blue if own else grey
        alpha = 1.0 if own else 0.35 + 0.5 * i / max(len(hold_times) - 1, 1)
        ax.plot(rr, ys, color=col, alpha=alpha, lw=2.0 if own else 1.4)
        ax.text(rr[-1] * 1.01, max(ys[-1], 0.8), f"T = {T} s", color=col, va="center", fontsize=9,
                fontweight="bold" if own else "normal")   # direct labels, no legend
    ax.set_xlabel("Detections per pole per hour")
    ax.set_ylabel("Predicted energy saving (%)")
    ax.set_xlim(0, r_max * 1.15)
    ax.set_ylim(bottom=0)
    ax.set_title("Model: effect of the hold time", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_design_chart.png"), dpi=200)
    fig.savefig(os.path.join(a.out, "fig_design_chart.pdf"))
    plt.close(fig)

    # --- Fig. C: every block over time (quality check) -------------------------
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for scen, col in (("baseline", grey), ("adaptive", blue)):
        bs = [b for b in blocks if b["scenario"] == scen]
        ax.plot([dt.datetime.fromtimestamp(b["start"], dt.timezone.utc) for b in bs],
                [b["mean_w"] for b in bs], "o", color=col, ms=4, label=scen)
    ax.set_ylabel("Mean power per block (W)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Measurement blocks", loc="left", fontsize=11)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_blocks.png"), dpi=200)
    plt.close(fig)
    print(f"Σχήματα: {a.out}/fig_saving_vs_rate.(png|pdf), fig_design_chart.(png|pdf), fig_blocks.png")


# =============================================================================
#  CLI
# =============================================================================
def floats(s):
    return [float(x) for x in s.split(",") if x.strip()] if s else []


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="source", required=True)

    common = argparse.ArgumentParser(add_help=False)
    g = common.add_argument_group("analysis")
    g.add_argument("--out", default="analysis/output", help="output folder")
    g.add_argument("--poles", type=int, default=3, help="number of poles in a row (default 3)")
    g.add_argument("--hold", type=float, default=60, help="hold time T in seconds (default 60)")
    g.add_argument("--standby", type=float, default=DEFAULT_STANDBY, help="standby power, W")
    g.add_argument("--levels", type=floats, default=list(DEFAULT_LEVELS),
                   help="total power at base,wave,full level, W (default 16.4,19.1,21.0)")
    g.add_argument("--model", choices=["independent", "walk"], default="walk",
                   help="traffic model: independent detections per pole, or pedestrians "
                        "walking past all poles (default walk)")
    g.add_argument("--gap", type=float, default=8, help="walk model: seconds between poles")
    g.add_argument("--rates", type=floats, default=[],
                   help="nominal traffic rates (detections/pole/h), e.g. 0,5,12,30,60; "
                        "blocks are grouped to the nearest one")
    g.add_argument("--blocks-csv", help="manual block list: start,end,scenario[,rate]")
    g.add_argument("--on-threshold", type=float, help="power above which the lights count as on, W")
    g.add_argument("--trim-min", type=float, default=5, help="minutes cut from both ends of a block")
    g.add_argument("--min-block-h", type=float, default=1.0, help="drop blocks shorter than this, h")
    g.add_argument("--split-h", type=float, default=0, help="split long blocks into chunks of h hours")
    g.add_argument("--boot", type=int, default=10000, help="bootstrap resamples")
    g.add_argument("--seed", type=int, default=1, help="random seed (bootstrap, demo)")

    pi = sub.add_parser("influx", parents=[common], help="read from InfluxDB v1")
    pi.add_argument("--host", required=True, help="e.g. http://homeassistant.local:8086")
    pi.add_argument("--db", default="kolones")
    pi.add_argument("--user", default="homeassistant")
    pi.add_argument("--password-env", default="INFLUX_PASSWORD",
                    help="environment variable with the password (otherwise you are asked)")
    pi.add_argument("--power-entity", required=True,
                    help="Shelly power entity without domain, e.g. shellyem_xxxx_channel_1_power")
    pi.add_argument("--power-measurement", default="W")
    pi.add_argument("--mode-entity", default="kolones_mode")
    pi.add_argument("--passes-entity", default="kolones_passes_night",
                    help="counter of detections ('' to skip)")
    pi.add_argument("--start", required=True, help="e.g. 2026-09-14")
    pi.add_argument("--end", help="default: now")

    pc = sub.add_parser("csv", parents=[common], help="read power.csv, mode.csv, passes.csv")
    pc.add_argument("--dir", required=True)

    pd = sub.add_parser("demo", parents=[common], help="synthetic data from a simulator of the automation")
    pd.add_argument("--nights", type=int, default=60)
    pd.add_argument("--night-hours", type=float, default=10)
    pd.add_argument("--traffic", choices=["walk", "independent"], default="walk")
    pd.add_argument("--noise", type=float, default=0.15, help="meter noise, W (sd)")

    a = p.parse_args(argv)
    if len(a.levels) != 3:
        p.error("--levels needs three values: base,wave,full")
    rng = np.random.default_rng(a.seed)

    if a.source == "influx":
        power, mode, passes = load_influx(a)
    elif a.source == "csv":
        power, mode, passes = load_csv(a)
    else:
        if not a.rates:
            a.rates = [0, 5, 12, 30, 60]
        power, mode, passes = simulate(a, rng)

    blocks = build_blocks(a, power, mode, passes)
    if not blocks:
        sys.exit("Δεν βγήκε κανένα μπλοκ μέτρησης. Έλεγξε το --on-threshold και τα δεδομένα.")
    assign_rate_groups(blocks, a.rates)
    nb = sum(b["scenario"] == "baseline" for b in blocks)
    print(f"Μπλοκ μέτρησης: {len(blocks)} ({nb} baseline, {len(blocks) - nb} adaptive)")
    rows = summarise(blocks, a, rng)
    write_outputs(blocks, rows, a)
    make_figures(blocks, rows, a)


if __name__ == "__main__":
    main()
