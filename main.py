"""gta6-trophy-rarity-forecast-01 entry: a two-component rarity forecaster.

A trophy's rarity is a ratio, earners over tracked players: r = 100 * E / N.
In launch week the two sides of that ratio move very differently. The tracked
player count N explodes as new players are picked up, which dilutes every
rate; the earner count E for a trophy grows as the tracked population
progresses through the game, which pushes rates back up. A persistence
forecast misses both motions, so this program models the two sides
separately on a log-time axis and recombines them:

  * Time axis: h = hours since release (2026-11-19T00:00:00Z), x = ln(1 + h).
    Growth in both N and E is fast early and slows, so log time linearises it.
  * N model: recency-weighted least squares of ln N on x, slope capped at
    1.0; the target-time estimate averages the full-slope and half-slope
    extrapolations geometrically and never drops below the last observed N.
  * Earner model: recency-weighted least squares of ln(E + 0.5) on x with the
    slope damped by 0.7 (earner growth saturates), extrapolated to the
    target and clamped between the last observed earner count and N*.
  * A direct logit-space trend fit of the rate itself captures whatever the
    ratio model misses; its weighted RMS residual measures local noise.
  * The median blends the three views on the logit scale (ratio model 0.15,
    trend 0.25, last observed rate 0.60). The blend leans on the last rate
    because both extrapolations are fitted to saturating curves: a synthetic
    launch-week simulator (earner S-curves plus exploding N, with harsh,
    mild and no dilution) favoured these weights and the narrower spread
    below over heavier extrapolation in every world tested. The spread
    combines a random-walk term, the trend fit's residual, and a
    dilution-uncertainty term driven by the N slope, floored at 0.25
    logit units.

The program reads every *.in case in $MAKEFUN_INPUT_DIR (default /input) and
writes <stem>.out in $MAKEFUN_OUTPUT_DIR (default /output). Python 3.13,
standard library only, fully deterministic: no randomness, no wall clock,
no network; two runs write identical bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves this file's directory off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Release moment: the time axis counts hours from here.
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Model constants (see the module docstring for the rationale).
N_SLOPE_CAP = 1.0        # ln N may grow at most this fast per unit of x.
E_SLOPE_DAMP = 0.7       # earner-growth slopes are damped by this factor.
W_MODEL = 0.15           # median blend weight: ratio model.
W_TREND = 0.25           # median blend weight: logit trend.
W_LAST = 0.60            # median blend weight: last observed rate.
RW_SIGMA = 0.35          # random-walk spread per square-root day (logit).
DILUTION_SIGMA = 0.35    # spread per unit of N slope per square-root day.
SIGMA_FLOOR = 0.25       # the spread never goes below this (logit units).
SATURATED_ROW = [99.0, 99.5, 99.9, 99.97, 99.99]


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def logistic_pct(v: float) -> float:
    """Inverse logit, returned as a percent; stable for large |v|."""
    if v >= 0:
        return 100.0 / (1.0 + math.exp(-v))
    e = math.exp(v)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """Round to 4 decimals, half up; whole numbers come back as int."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def hours_since_release(stamp: str) -> float:
    """Hours between the release moment and an ISO-8601 UTC timestamp."""
    t = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    return (t - RELEASE).total_seconds() / 3600.0


def time_axis(stamp: str) -> float:
    """x = ln(1 + h) for a timestamp; the guard keeps pre-release reads sane."""
    return math.log(max(1.0 + hours_since_release(stamp), 1e-6))


def wls(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Recency-weighted least squares of y on x; weights 1..n in list order.

    Returns (intercept, slope, weighted RMS residual). With fewer than two
    points, or a degenerate x spread, the slope is 0 and the intercept is
    the weighted mean.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    sw = sx = sy = sxx = sxy = 0.0
    for j in range(n):
        w = float(j + 1)
        sw += w
        sx += w * xs[j]
        sy += w * ys[j]
        sxx += w * xs[j] * xs[j]
        sxy += w * xs[j] * ys[j]
    denom = sw * sxx - sx * sx
    if n < 2 or abs(denom) < 1e-12:
        intercept, slope = sy / sw, 0.0
    else:
        slope = (sw * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / sw
    sse = 0.0
    for j in range(n):
        w = float(j + 1)
        resid = ys[j] - (intercept + slope * xs[j])
        sse += w * resid * resid
    return intercept, slope, math.sqrt(sse / sw)


def fit_player_count(history: list[dict], x_target: float) -> tuple[float, float]:
    """Extrapolate the tracked-player count to the target time.

    Fits ln N on x (recency-weighted, slope capped), then takes the
    geometric mean of the full-slope and half-slope extrapolations, floored
    at the last observed count. Returns (N*, capped slope of ln N).
    """
    xs = [time_axis(read["at"]) for read in history]
    ys = [math.log(max(float(read["players_tracked"]), 1.0)) for read in history]
    intercept, slope, _ = wls(xs, ys)
    slope = min(slope, N_SLOPE_CAP)
    n_full = math.exp(intercept + slope * x_target)
    n_half = math.exp(intercept + 0.5 * slope * x_target)
    n_star = math.sqrt(n_full * n_half)
    n_last = float(history[-1]["players_tracked"])
    return max(n_star, n_last), slope


def forecast_trophy(
    obs: list[tuple[float, float, float]],
    n_star: float,
    slope_ln_n: float,
    horizon_days: float,
    x_target: float,
) -> list:
    """Forecast one trophy's five quantiles (percent).

    `obs` holds one (x, rate, players_tracked) triple per read that listed
    the trophy, oldest first.
    """
    if not obs:
        # Never observed: no signal at all. A token floor row keeps the case
        # valid; such a trophy cannot occur in a well-formed case.
        obs = [(x_target, 0.01, n_star)]
    last_rate = obs[-1][1]
    if last_rate >= 99.99:
        return list(SATURATED_ROW)

    if len(obs) < 2:
        rate_model = rate_trend = last_rate
        rms = 0.0
    else:
        xs = [o[0] for o in obs]
        # Earner model: fit ln(E + 0.5) on x, damp the slope, extrapolate,
        # and keep the earner count between its last observed value and N*.
        e_last = obs[-1][1] * obs[-1][2] / 100.0
        ln_e = [math.log(o[1] * o[2] / 100.0 + 0.5) for o in obs]
        a_e, b_e, _ = wls(xs, ln_e)
        e_star = math.exp(a_e + E_SLOPE_DAMP * b_e * x_target) - 0.5
        e_star = min(max(e_star, min(e_last, n_star)), n_star)
        rate_model = 100.0 * e_star / n_star
        # Logit trend: fit logit(rate) on x and extrapolate directly.
        lg = [logit(o[1]) for o in obs]
        a_l, b_l, rms = wls(xs, lg)
        rate_trend = logistic_pct(a_l + b_l * x_target)

    median_logit = (
        W_MODEL * logit(rate_model)
        + W_TREND * logit(rate_trend)
        + W_LAST * logit(last_rate)
    )
    root_days = math.sqrt(horizon_days)
    sigma = math.sqrt(
        (RW_SIGMA * root_days) ** 2
        + rms**2
        + (DILUTION_SIGMA * abs(slope_ln_n) * root_days) ** 2
    )
    sigma = max(sigma, SIGMA_FLOOR)

    row = []
    for z in Z:
        row.append(round4(clamp_pct(logistic_pct(median_logit + z * sigma))))
    # Rounding is monotone, but enforce the contract explicitly.
    for k in range(1, len(row)):
        if row[k] < row[k - 1]:
            row[k] = row[k - 1]
    return row


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    x_target = time_axis(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    n_star, slope_ln_n = fit_player_count(history, x_target)

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        obs = []
        for read in history:
            rate = read["rates"][i]
            if rate is not None:
                obs.append(
                    (time_axis(read["at"]), float(rate), float(read["players_tracked"]))
                )
        q = forecast_trophy(obs, n_star, slope_ln_n, horizon_days, x_target)
        rows.append({"id": trophy["id"], "q": q})
    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
