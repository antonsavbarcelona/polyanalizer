"""Stage B trade-flow incremental-value analysis.

Answers exactly one question per baseline momentum config: does adding a
Binance aggressive-flow confirmation (direction-normalized signed_flow >=
threshold) improve the response beyond what the momentum-only signal
already gets -- not "what's the best absolute config."

Everything is computed as a DELTA against the exact-matching flow=OFF
baseline (same momentum_window_ms, z_threshold, volatility_window_ms,
asset, latency_ms, size_shares, horizon_ms) -- never as a standalone
leaderboard. A flow variant that shrinks the sample to a lucky handful of
signals is not a win even if its raw uplift looks bigger; sample and
market retention are load-bearing metrics here, not a footnote.

Sections:
  1. Delta table: every (baseline, flow_window, threshold) triple vs its
     OFF baseline -- Δmean_uplift, signal_retention, market_retention, at
     the canonical slice (asset=ALL, latency=250ms, size=100, horizon=2000ms).
  2. Per-baseline best flow variant, eligible only if it clears sample AND
     market retention floors (never picked on raw delta alone).
  3. Region ranking: (flow_window, flow_threshold) pairs aggregated ACROSS
     all 24 baselines and ranked on all four axes at once -- incremental
     improvement, sample retention, market coverage, neighbor consistency
     (grid-adjacent regions also net-positive). This -- not any single
     per-baseline "best" -- is what picks the "leading flow region" used
     below; a region has to win broadly, not just in one lucky baseline
     (the flow_window=3000/threshold=.6/n=24 anti-pattern the plan calls
     out is exactly what the coverage + neighbor-consistency floors below
     are built to reject).
  4. Signed-flow bucket analysis (<0, 0-.1, .1-.2, ..., .6+), computed
     directly from the 24 baseline configs' raw signals (pooled BTC/ETH/
     SOL) using the wide, config-independent flow_1s snapshot field --
     independent of any flow_window_ms/threshold choice. Checks whether
     response scales monotonically with signed flow (real information) or
     is scattered until one lucky threshold (selection noise).
  5. Δuplift shape across horizons (500ms/1s/2s/3s/5s) for the leading
     region -- distinguishes "flow flags a faster impulse" (effect
     concentrated at short horizons) from "flow adds durable information"
     (effect holds/grows at longer horizons).
  6. Per-asset (BTC/ETH/SOL) breakdown of the leading region's delta --
     flags ASSET_SPECIFIC if sign disagrees across assets or the spread
     across assets dwarfs the pooled ALL effect.
  7. Final verdict: FLOW_ADDS_VALUE = YES / NO / INCONCLUSIVE, plus the
     robust flow region if not a hard NO.

Design choices worth flagging explicitly (none of these are dictated by
the plan verbatim -- they're this script's concrete interpretation of it):
  - BUCKET_FLOW_FIELD = flow_1s: the plan's own worked examples use "flow
    1s >= .2"; flow_1s also sits mid-grid among the swept flow_window_ms
    values (250-5000ms) and is captured on EVERY signal regardless of a
    config's own flow settings (detect.py's FLOW_WINDOWS_MS snapshot), so
    bucketing on it doesn't depend on section 3/5's flow_window_ms choice.
  - MIN_BASELINE_COVERAGE / MIN_REGION_POSITIVE_FRACTION: no exact
    percentages are given in the plan beyond "not a single lucky config" --
    50%/60% are this script's floors for what counts as "broadly robust".
  - The reported CI is an APPROXIMATION: bootstrap_ci95_uplift_low/high in
    signal_response_summary is each side's own CI on (signal - matched
    control), not a CI on the OFF-vs-flow delta. This script combines the
    two sides' half-widths as if independent (SE_delta = sqrt(SE_on^2 +
    SE_off^2)), which is conservative/wider than the true delta CI since
    flow's signals are a strict subset of OFF's (correlated, not
    independent draws). A proper paired-bootstrap delta CI is not
    implemented -- flagged here rather than silently presented as exact.

Run: python -m research.analyze_stage_b
Requires Stage B's run to have reached its own final pooling pass (asset=
ALL rows only exist after the whole 3528-unit run completes, not
incrementally -- see research/discovery_experiment.py's pooling comment).
Sections that need per-region cross-baseline data degrade gracefully to
"no leading region" on a partially-complete DB rather than crashing.
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
from collections import defaultdict

from research.discovery.aggregation import compute_signal_response_summary
from research.discovery.reconstruct import control_response_from_row, signal_response_from_row
from research.discovery_types import ALL_ASSET

DB = "data/stage_b_flow_results.db"
EXPERIMENT_ID = None  # filled in from the DB at runtime (single experiment per db)

CANON_LATENCY = 250
CANON_SIZE = 100.0
CANON_HORIZON = 2000
HORIZONS = (500, 1000, 2000, 3000, 5000)
LATENCIES = (100, 250, 500)
ASSETS = ("BTC", "ETH", "SOL")
FLOW_WINDOWS = (250, 500, 750, 1000, 1500, 2000, 3000, 5000)
FLOW_THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)
SIGNED_FLOW_BUCKETS = [
    (None, 0.0, "<0"), (0.0, 0.1, "0-.1"), (0.1, 0.2, ".1-.2"), (0.2, 0.3, ".2-.3"),
    (0.3, 0.4, ".3-.4"), (0.4, 0.5, ".4-.5"), (0.5, 0.6, ".5-.6"), (0.6, None, ".6+"),
]
BUCKET_FLOW_FIELD = "flow_1s"
BUCKET_MIN_N = 20  # buckets thinner than this are excluded from the monotonicity check

OUT_PATH = "research/stage_b_analysis_report.md"

# Minimum sample size to trust a flow variant at all (below this, retention
# math is meaningless noise regardless of how good the uplift looks).
MIN_SIGNAL_COUNT = 50
# Minimum retention -- a variant that keeps less than this fraction of the
# baseline's signals/markets is disqualified from "best flow variant" even
# if its delta looks good, per the .6+/n=24 anti-pattern called out in the plan.
MIN_SIGNAL_RETENTION = 0.25
MIN_MARKET_RETENTION = 0.5

# A (flow_window, flow_threshold) region only counts as "robust" if at least
# this fraction of the 24 baselines are individually eligible AND at least
# this fraction of THOSE have a positive delta -- one lucky baseline is not
# a region win.
MIN_BASELINE_COVERAGE = 0.5
MIN_REGION_POSITIVE_FRACTION = 0.6


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con


def load_configs(con: sqlite3.Connection) -> dict[str, dict]:
    return {r["signal_config_id"]: json.loads(r["config_json"])
            for r in con.execute("SELECT signal_config_id, config_json FROM signal_configs")}


def fmt(v, pct=False, digits=4):
    if v is None:
        return "—"
    if pct:
        return f"{v * 100:.2f}%"
    return f"{v:.{digits}f}"


def response_row(con, experiment_id, cfg_id, asset, latency, horizon):
    return con.execute(
        """SELECT * FROM signal_response_summary
           WHERE experiment_id=? AND signal_config_id=? AND asset=? AND latency_ms=?
             AND size_shares=? AND horizon_ms=?""",
        (experiment_id, cfg_id, asset, latency, CANON_SIZE, horizon),
    ).fetchone()


def config_summary_row(con, experiment_id, cfg_id, asset, latency):
    return con.execute(
        """SELECT * FROM signal_config_summary
           WHERE experiment_id=? AND signal_config_id=? AND asset=? AND latency_ms=? AND size_shares=?""",
        (experiment_id, cfg_id, asset, latency, CANON_SIZE),
    ).fetchone()


def build_baseline_map(configs: dict[str, dict]) -> dict[tuple, dict]:
    by_baseline: dict[tuple, dict] = {}
    for cfg_id, cfg in configs.items():
        key = (cfg["momentum_window_ms"], cfg["z_threshold"], cfg["volatility_window_ms"])
        entry = by_baseline.setdefault(key, {"off": None, "on": {}})
        if cfg["flow_window_ms"] is None:
            entry["off"] = cfg_id
        else:
            entry["on"][(cfg["flow_window_ms"], cfg["flow_threshold"])] = cfg_id
    return by_baseline


def _region_neighbors(window: int, thr: float) -> list[tuple[int, float]]:
    """Adjacent grid points along the flow_window and flow_threshold axes,
    one axis at a time -- same neighbor definition as plateau.py (differ in
    exactly one axis, by one adjacent grid step)."""
    ws, ts = sorted(FLOW_WINDOWS), sorted(FLOW_THRESHOLDS)
    wi, ti = ws.index(window), ts.index(thr)
    out = []
    if wi > 0:
        out.append((ws[wi - 1], thr))
    if wi < len(ws) - 1:
        out.append((ws[wi + 1], thr))
    if ti > 0:
        out.append((window, ts[ti - 1]))
    if ti < len(ts) - 1:
        out.append((window, ts[ti + 1]))
    return out


# ---------------------------------------------------------------------------
# Sections 1-2: delta table + per-baseline eligible best variant
# ---------------------------------------------------------------------------

def compute_deltas(con, experiment_id, by_baseline, emit):
    emit("## Per-baseline: best flow variant vs OFF (canonical slice)")
    emit()
    emit(f"Selection rule: among flow variants with signal_count >= {MIN_SIGNAL_COUNT} AND "
         f"signal_retention >= {MIN_SIGNAL_RETENTION:.0%} AND market_retention >= "
         f"{MIN_MARKET_RETENTION:.0%}, pick the max Δuplift_mean. A variant failing retention is "
         "never selected even with a bigger raw delta.")
    emit()
    emit("| baseline (mom/z/vol) | OFF n | OFF uplift@2s | best flow | flow n | flow uplift@2s | Δuplift | "
         "sig_retain | mkt_retain |")
    emit("|---|---|---|---|---|---|---|---|---|")

    best_per_baseline = {}
    region_candidates: dict[tuple, list] = defaultdict(list)

    for key, entry in sorted(by_baseline.items()):
        off_id = entry["off"]
        if off_id is None:
            continue
        off_resp = response_row(con, experiment_id, off_id, ALL_ASSET, CANON_LATENCY, CANON_HORIZON)
        off_cs = config_summary_row(con, experiment_id, off_id, ALL_ASSET, CANON_LATENCY)
        if off_resp is None or off_cs is None or off_cs["signal_count"] == 0:
            emit(f"| {key} | — | no OFF data | | | | | | |")
            continue

        candidates = []
        for (window, thr), cfg_id in entry["on"].items():
            resp = response_row(con, experiment_id, cfg_id, ALL_ASSET, CANON_LATENCY, CANON_HORIZON)
            cs = config_summary_row(con, experiment_id, cfg_id, ALL_ASSET, CANON_LATENCY)
            if resp is None or cs is None or resp["uplift_mean_response"] is None:
                continue
            sig_retain = cs["signal_count"] / off_cs["signal_count"] if off_cs["signal_count"] else 0
            mkt_retain = cs["market_count"] / off_cs["market_count"] if off_cs["market_count"] else 0
            delta = resp["uplift_mean_response"] - off_resp["uplift_mean_response"]
            c = {"window": window, "thr": thr, "cfg_id": cfg_id, "n": cs["signal_count"],
                 "markets": cs["market_count"], "uplift": resp["uplift_mean_response"],
                 "off_uplift": off_resp["uplift_mean_response"], "delta": delta,
                 "sig_retain": sig_retain, "mkt_retain": mkt_retain, "baseline_key": key, "off_id": off_id}
            candidates.append(c)
            region_candidates[(window, thr)].append(c)

        eligible = [c for c in candidates if c["n"] >= MIN_SIGNAL_COUNT
                    and c["sig_retain"] >= MIN_SIGNAL_RETENTION
                    and c["mkt_retain"] >= MIN_MARKET_RETENTION]
        best = max(eligible, key=lambda c: c["delta"]) if eligible else None
        best_per_baseline[key] = {"off_id": off_id, "off_n": off_cs["signal_count"],
                                   "off_uplift": off_resp["uplift_mean_response"], "best": best,
                                   "all_candidates": candidates}

        if best is None:
            emit(f"| {key} | {off_cs['signal_count']} | {fmt(off_resp['uplift_mean_response'])} | "
                 f"none eligible | | | | | |")
        else:
            emit(f"| {key} | {off_cs['signal_count']} | {fmt(off_resp['uplift_mean_response'])} | "
                 f"{best['window']}ms/{best['thr']} | {best['n']} | {fmt(best['uplift'])} | "
                 f"{fmt(best['delta'])} | {fmt(best['sig_retain'], pct=True)} | {fmt(best['mkt_retain'], pct=True)} |")
    emit()
    return best_per_baseline, region_candidates


# ---------------------------------------------------------------------------
# Section 3: region ranking (pooled across baselines, 4 axes)
# ---------------------------------------------------------------------------

def rank_regions(region_candidates, emit):
    region_stats: dict[tuple, dict] = {}
    for region, cands in region_candidates.items():
        eligible = [c for c in cands if c["n"] >= MIN_SIGNAL_COUNT
                    and c["sig_retain"] >= MIN_SIGNAL_RETENTION
                    and c["mkt_retain"] >= MIN_MARKET_RETENTION]
        if not eligible:
            continue
        n_eligible = len(eligible)
        region_stats[region] = {
            "n_baselines": len(cands), "n_eligible": n_eligible,
            "mean_delta": sum(c["delta"] for c in eligible) / n_eligible,
            "frac_positive": sum(1 for c in eligible if c["delta"] > 0) / n_eligible,
            "mean_sig_retain": sum(c["sig_retain"] for c in eligible) / n_eligible,
            "mean_mkt_retain": sum(c["mkt_retain"] for c in eligible) / n_eligible,
            "mean_off_uplift": sum(c["off_uplift"] for c in eligible) / n_eligible,
            "mean_flow_uplift": sum(c["uplift"] for c in eligible) / n_eligible,
        }

    ranked = []
    for region, stats in region_stats.items():
        coverage = stats["n_eligible"] / 24.0
        if coverage < MIN_BASELINE_COVERAGE or stats["frac_positive"] < MIN_REGION_POSITIVE_FRACTION:
            continue
        neighbor_stats = [region_stats[n] for n in _region_neighbors(*region) if n in region_stats]
        stats["neighbor_positive"] = sum(1 for s in neighbor_stats if s["mean_delta"] > 0)
        stats["neighbor_total"] = len(neighbor_stats)
        ranked.append((region, stats))

    ranked.sort(key=lambda t: t[1]["mean_delta"] * t[1]["frac_positive"], reverse=True)

    emit("## Region ranking (pooled across baselines, 4-axis)")
    emit()
    emit(f"A region qualifies only if >= {MIN_BASELINE_COVERAGE:.0%} of the 24 baselines are "
         f"individually eligible (n>={MIN_SIGNAL_COUNT}, sig_retain>={MIN_SIGNAL_RETENTION:.0%}, "
         f"mkt_retain>={MIN_MARKET_RETENTION:.0%}) AND >= {MIN_REGION_POSITIVE_FRACTION:.0%} of those "
         "have a positive Δuplift. Ranked by mean_delta x frac_positive; the neighbor column shows how "
         "many of the up-to-4 adjacent (window, threshold) grid points are ALSO net-positive regions.")
    emit()
    emit("| flow window | threshold | baselines eligible | mean Δuplift | frac positive | "
         "mean sig_retain | mean mkt_retain | neighbors positive |")
    emit("|---|---|---|---|---|---|---|---|")
    for region, stats in ranked[:10]:
        emit(f"| {region[0]}ms | {region[1]} | {stats['n_eligible']}/24 | {fmt(stats['mean_delta'])} | "
             f"{fmt(stats['frac_positive'], pct=True)} | {fmt(stats['mean_sig_retain'], pct=True)} | "
             f"{fmt(stats['mean_mkt_retain'], pct=True)} | {stats['neighbor_positive']}/{stats['neighbor_total']} |")
    emit()

    if not ranked:
        emit("No (flow_window, flow_threshold) region cleared the robustness bar across baselines.")
        emit()
        return None, None
    return ranked[0]


# ---------------------------------------------------------------------------
# Section 4: signed-flow bucket analysis
# ---------------------------------------------------------------------------

def bucket_of(signed_flow: float) -> str | None:
    for lo, hi, label in SIGNED_FLOW_BUCKETS:
        if lo is not None and signed_flow < lo:
            continue
        if hi is not None and signed_flow >= hi:
            continue
        return label
    return None


def _bucket_monotonicity(bucket_uplifts, bucket_signal_count, order) -> str:
    values = []
    for label in order:
        if bucket_signal_count.get(label, 0) < BUCKET_MIN_N:
            continue
        v = bucket_uplifts.get(label, {}).get(CANON_HORIZON)
        if v is None:
            continue
        values.append(v)
    if len(values) < 3:
        return f"INCONCLUSIVE (fewer than 3 buckets with >= {BUCKET_MIN_N} signals)"
    steps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    frac_nondecreasing = sum(1 for s in steps if s >= 0) / len(steps)
    if frac_nondecreasing >= 0.8:
        return "YES"
    if frac_nondecreasing >= 0.5:
        return "PARTIAL"
    return "NO"


def signed_flow_buckets(con, experiment_id, off_ids, emit):
    emit(f"## Signed-flow bucket analysis (baseline momentum signals, {BUCKET_FLOW_FIELD}, "
         f"latency={CANON_LATENCY}ms, size={CANON_SIZE:.0f} shares)")
    emit()
    emit("Buckets the raw signed-flow value at signal time for all 24 baseline (flow=OFF) configs' "
         "signals, pooled across BTC/ETH/SOL -- independent of any flow_window_ms/threshold config "
         "choice. Checks whether response scales monotonically with signed flow (real information) or "
         "is flat/scattered until one lucky threshold (selection noise).")
    emit()

    if not off_ids:
        emit("No OFF baseline configs found -- skipped.")
        emit()
        return {}, "INCONCLUSIVE (no OFF configs)"

    id_ph = ",".join("?" * len(off_ids))
    sig_rows = con.execute(
        f"""SELECT signal_id, market_id, direction, snapshot_json FROM signals
            WHERE experiment_id=? AND signal_config_id IN ({id_ph})""",
        (experiment_id, *off_ids),
    ).fetchall()

    bucket_of_signal: dict[str, str] = {}
    bucket_markets: dict[str, set] = defaultdict(set)
    bucket_signal_count: dict[str, int] = defaultdict(int)
    for r in sig_rows:
        snap = json.loads(r["snapshot_json"])
        flow_val = snap.get(BUCKET_FLOW_FIELD)
        if flow_val is None:
            continue
        sign = 1.0 if r["direction"] == "UP" else -1.0
        label = bucket_of(sign * flow_val)
        if label is None:
            continue
        bucket_of_signal[r["signal_id"]] = label
        bucket_markets[label].add(r["market_id"])
        bucket_signal_count[label] += 1

    resp_rows = con.execute(
        f"""SELECT * FROM signal_response WHERE signal_config_id IN ({id_ph})
            AND latency_ms=? AND size_shares=?""",
        (*off_ids, CANON_LATENCY, CANON_SIZE),
    ).fetchall()
    responses = [signal_response_from_row(r) for r in resp_rows]

    ctrl_rows = con.execute(
        f"""SELECT control_id, source_signal_id FROM controls WHERE signal_config_id IN ({id_ph})""",
        off_ids,
    ).fetchall()
    control_bucket_of: dict[str, str] = {
        r["control_id"]: bucket_of_signal[r["source_signal_id"]]
        for r in ctrl_rows if r["source_signal_id"] in bucket_of_signal
    }

    cresp_rows = con.execute(
        f"""SELECT * FROM control_response WHERE signal_config_id IN ({id_ph})
            AND latency_ms=? AND size_shares=?""",
        (*off_ids, CANON_LATENCY, CANON_SIZE),
    ).fetchall()
    control_responses = [control_response_from_row(r) for r in cresp_rows]

    resp_by_bucket_h: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for r in responses:
        label = bucket_of_signal.get(r.signal_id)
        if label is not None:
            resp_by_bucket_h[label][r.horizon_ms].append(r)
    ctrl_by_bucket_h: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for r in control_responses:
        label = control_bucket_of.get(r.control_id)
        if label is not None:
            ctrl_by_bucket_h[label][r.horizon_ms].append(r)

    header = ["signed_flow", "signals", "markets"] + [f"uplift@{h}ms" for h in HORIZONS]
    emit("| " + " | ".join(header) + " |")
    emit("|" + "---|" * len(header))

    bucket_uplifts: dict[str, dict[int, float | None]] = {}
    order = [label for _, _, label in SIGNED_FLOW_BUCKETS]
    for label in order:
        n = bucket_signal_count.get(label, 0)
        markets = len(bucket_markets.get(label, ()))
        row_uplifts: dict[int, float | None] = {}
        for h in HORIZONS:
            rs = resp_by_bucket_h.get(label, {}).get(h, [])
            cs = ctrl_by_bucket_h.get(label, {}).get(h, [])
            if not rs:
                row_uplifts[h] = None
                continue
            mkt_of = {id(x): x.market_id for x in rs}
            c_mkt_of = {id(x): x.market_id for x in cs}
            summary = compute_signal_response_summary(
                "stage_b_bucket", ALL_ASSET, CANON_LATENCY, CANON_SIZE, h, rs, mkt_of, cs, c_mkt_of,
                bootstrap_iterations=0,
            )
            row_uplifts[h] = summary.uplift_mean_response
        bucket_uplifts[label] = row_uplifts
        cells = [label, str(n), str(markets)] + [fmt(row_uplifts.get(h)) for h in HORIZONS]
        emit("| " + " | ".join(cells) + " |")
    emit()

    monotonic = _bucket_monotonicity(bucket_uplifts, bucket_signal_count, order)
    emit(f"Bucket monotonicity @{CANON_HORIZON}ms: **{monotonic}** "
         f"(buckets with < {BUCKET_MIN_N} signals excluded from the check).")
    emit()
    return bucket_uplifts, monotonic


# ---------------------------------------------------------------------------
# Sections 5-6: horizon shape + per-asset breakdown for the leading region
# ---------------------------------------------------------------------------

def horizon_shape(con, experiment_id, by_baseline, leading_region, emit):
    emit("## Δuplift shape across horizons — leading region")
    emit()
    if leading_region is None:
        emit("No leading region identified (see region ranking above) -- skipped.")
        emit()
        return None

    window, thr = leading_region
    emit(f"Leading region: flow_window={window}ms, signed_flow >= {thr} "
         f"(asset=ALL, latency={CANON_LATENCY}ms, size={CANON_SIZE:.0f} shares; mean Δuplift is an "
         "unweighted average across every baseline with data for both OFF and this flow variant, "
         "which can include baselines that didn't clear section 1-2's eligibility floor).")
    emit()
    emit("| horizon | mean Δuplift | baselines contributing |")
    emit("|---|---|---|")
    shape = {}
    for h in HORIZONS:
        deltas = []
        for entry in by_baseline.values():
            off_id, on_id = entry["off"], entry["on"].get((window, thr))
            if off_id is None or on_id is None:
                continue
            off_resp = response_row(con, experiment_id, off_id, ALL_ASSET, CANON_LATENCY, h)
            on_resp = response_row(con, experiment_id, on_id, ALL_ASSET, CANON_LATENCY, h)
            if off_resp is None or on_resp is None:
                continue
            if off_resp["uplift_mean_response"] is None or on_resp["uplift_mean_response"] is None:
                continue
            deltas.append(on_resp["uplift_mean_response"] - off_resp["uplift_mean_response"])
        mean_delta = sum(deltas) / len(deltas) if deltas else None
        shape[h] = mean_delta
        emit(f"| {h}ms | {fmt(mean_delta)} | {len(deltas)}/24 |")
    emit()
    return shape


def asset_breakdown(con, experiment_id, by_baseline, leading_region, emit):
    emit("## Per-asset breakdown — leading region")
    emit()
    if leading_region is None:
        emit("No leading region identified -- skipped.")
        emit()
        return None, "N/A"

    window, thr = leading_region
    emit(f"Δuplift@{CANON_HORIZON}ms by asset (latency={CANON_LATENCY}ms, size={CANON_SIZE:.0f} shares), "
         "unweighted average across baselines with data for both sides.")
    emit()
    emit("| asset | mean Δuplift | baselines contributing |")
    emit("|---|---|---|")
    per_asset = {}
    for asset in (*ASSETS, ALL_ASSET):
        deltas = []
        for entry in by_baseline.values():
            off_id, on_id = entry["off"], entry["on"].get((window, thr))
            if off_id is None or on_id is None:
                continue
            off_resp = response_row(con, experiment_id, off_id, asset, CANON_LATENCY, CANON_HORIZON)
            on_resp = response_row(con, experiment_id, on_id, asset, CANON_LATENCY, CANON_HORIZON)
            if off_resp is None or on_resp is None:
                continue
            if off_resp["uplift_mean_response"] is None or on_resp["uplift_mean_response"] is None:
                continue
            deltas.append(on_resp["uplift_mean_response"] - off_resp["uplift_mean_response"])
        mean_delta = sum(deltas) / len(deltas) if deltas else None
        per_asset[asset] = mean_delta
        emit(f"| {asset} | {fmt(mean_delta)} | {len(deltas)}/24 |")
    emit()

    valid = {a: per_asset.get(a) for a in ASSETS if per_asset.get(a) is not None}
    verdict = "N/A"
    if len(valid) == len(ASSETS):
        signs_positive = [v > 0 for v in valid.values()]
        all_ref = per_asset.get(ALL_ASSET) or 0.0
        if all(signs_positive) or not any(signs_positive):
            spread = max(valid.values()) - min(valid.values())
            verdict = "ASSET_SPECIFIC" if spread > 2 * abs(all_ref) + 1e-9 else "GLOBAL"
        else:
            verdict = "ASSET_SPECIFIC"  # signs disagree across assets
    emit(f"Cross-asset consistency: **{verdict}**")
    emit()
    return per_asset, verdict


def _latency_breakdown(con, experiment_id, by_baseline, leading_region):
    window, thr = leading_region
    out = {}
    for latency in LATENCIES:
        deltas = []
        for entry in by_baseline.values():
            off_id, on_id = entry["off"], entry["on"].get((window, thr))
            if off_id is None or on_id is None:
                continue
            off_resp = response_row(con, experiment_id, off_id, ALL_ASSET, latency, CANON_HORIZON)
            on_resp = response_row(con, experiment_id, on_id, ALL_ASSET, latency, CANON_HORIZON)
            if off_resp is None or on_resp is None:
                continue
            if off_resp["uplift_mean_response"] is None or on_resp["uplift_mean_response"] is None:
                continue
            deltas.append(on_resp["uplift_mean_response"] - off_resp["uplift_mean_response"])
        out[latency] = sum(deltas) / len(deltas) if deltas else None
    return out


def _approx_delta_ci(con, experiment_id, by_baseline, leading_region):
    """See module docstring's CI caveat -- this treats the OFF and flow
    uplift-vs-matched-control estimates as independent, which is
    conservative (wider) since flow's signals are a strict subset of OFF's."""
    window, thr = leading_region
    deltas, half_widths = [], []
    for entry in by_baseline.values():
        off_id, on_id = entry["off"], entry["on"].get((window, thr))
        if off_id is None or on_id is None:
            continue
        off_resp = response_row(con, experiment_id, off_id, ALL_ASSET, CANON_LATENCY, CANON_HORIZON)
        on_resp = response_row(con, experiment_id, on_id, ALL_ASSET, CANON_LATENCY, CANON_HORIZON)
        if off_resp is None or on_resp is None:
            continue
        if off_resp["uplift_mean_response"] is None or on_resp["uplift_mean_response"] is None:
            continue
        deltas.append(on_resp["uplift_mean_response"] - off_resp["uplift_mean_response"])
        off_lo, off_hi = off_resp["bootstrap_ci95_uplift_low"], off_resp["bootstrap_ci95_uplift_high"]
        on_lo, on_hi = on_resp["bootstrap_ci95_uplift_low"], on_resp["bootstrap_ci95_uplift_high"]
        if None in (off_lo, off_hi, on_lo, on_hi):
            continue
        se_off = (off_hi - off_lo) / (2 * 1.96)
        se_on = (on_hi - on_lo) / (2 * 1.96)
        half_widths.append(1.96 * math.sqrt(se_off ** 2 + se_on ** 2))

    if not deltas or not half_widths:
        return None
    mean_delta = sum(deltas) / len(deltas)
    mean_half_width = sum(half_widths) / len(half_widths)
    return mean_delta - mean_half_width, mean_delta + mean_half_width


# ---------------------------------------------------------------------------
# Section 7: final verdict
# ---------------------------------------------------------------------------

def final_verdict(region_stats, leading_region, bucket_monotonic, shape, per_asset, asset_verdict,
                   delta_ci, latency_deltas, emit) -> str:
    emit("## Final verdict")
    emit()

    if leading_region is None:
        emit("FLOW_ADDS_VALUE = NO")
        emit()
        emit("No (flow_window, flow_threshold) region cleared the cross-baseline robustness bar "
             f"(>= {MIN_BASELINE_COVERAGE:.0%} of the 24 baselines individually eligible, "
             f">= {MIN_REGION_POSITIVE_FRACTION:.0%} of those net-positive). Flow mostly reduces "
             "sample size without a stable incremental response. Do not include flow; proceed to Stage C.")
        return "NO"

    window, thr = leading_region
    canon_delta = region_stats["mean_delta"]
    ok_bucket = bucket_monotonic in ("YES", "PARTIAL")
    ok_delta = canon_delta is not None and canon_delta > 0
    verdict = "YES" if (ok_bucket and ok_delta) else ("INCONCLUSIVE" if ok_delta else "NO")

    emit(f"FLOW_ADDS_VALUE = {verdict}")
    emit()
    if verdict != "YES":
        if not ok_delta:
            reason = "the leading region's pooled Δuplift is not positive"
            treat_as = "a hard NO -- do not include flow."
        else:
            reason = f"the signed-flow bucket table is not monotonic ({bucket_monotonic}, selection-noise risk)"
            treat_as = "unresolved -- needs a human look at the bucket table and region ranking above."
        emit(f"Region {window}ms / signed_flow>={thr} passed the coverage/positivity bar, but {reason}. "
             f"Treat as {treat_as}")
        emit()

    emit("Robust flow region:" if verdict == "YES" else "Candidate flow region (not adopted as-is):")
    emit()
    emit("    baseline: momentum 500-1000ms, z 2.25-3.00, vol 30-60s")
    emit(f"    incremental confirmation: flow window {window}ms, signed_flow >= {thr}")
    emit()
    emit("Effect:")
    label_width = len(f"baseline blended uplift@{CANON_HORIZON}ms")
    emit(f"    {f'baseline blended uplift@{CANON_HORIZON}ms'.ljust(label_width)} = "
         f"{fmt(region_stats.get('mean_off_uplift'))}")
    emit(f"    {'with flow'.ljust(label_width)} = {fmt(region_stats.get('mean_flow_uplift'))}")
    emit(f"    {'incremental'.ljust(label_width)} = {fmt(canon_delta)}")
    emit()
    emit(f"    signal retention = {fmt(region_stats['mean_sig_retain'], pct=True)}")
    emit(f"    market retention = {fmt(region_stats['mean_mkt_retain'], pct=True)}")
    emit(f"    baselines eligible = {region_stats['n_eligible']}/24, "
         f"frac positive = {fmt(region_stats['frac_positive'], pct=True)}")
    emit()
    if per_asset:
        for asset in ASSETS:
            emit(f"    {asset} = {fmt(per_asset.get(asset))}")
    emit(f"    cross-asset consistency = {asset_verdict}")
    emit()
    emit(f"    neighbor configs positive = {region_stats.get('neighbor_positive', 0)}/"
         f"{region_stats.get('neighbor_total', 0)}")
    emit(f"    bucket monotonicity = {bucket_monotonic}")
    emit()
    if latency_deltas:
        emit("    latency:")
        for latency in LATENCIES:
            emit(f"      {latency}ms: {fmt(latency_deltas.get(latency))}")
        emit()
    if shape:
        emit("    horizon shape (Δuplift):")
        for h in HORIZONS:
            emit(f"      {h}ms: {fmt(shape.get(h))}")
        emit()
    if delta_ci:
        emit(f"    CI (approx 95%, treats OFF/flow uplift estimates as independent -- see module "
             f"docstring): [{fmt(delta_ci[0])}, {fmt(delta_ci[1])}]")
    else:
        emit("    CI: not available (missing bootstrap CI data for one or more contributing baselines)")
    emit()
    return verdict


def main() -> None:
    global EXPERIMENT_ID
    con = connect()
    EXPERIMENT_ID = con.execute("SELECT experiment_id FROM experiments").fetchone()["experiment_id"]
    configs = load_configs(con)
    out: list[str] = []

    def emit(line=""):
        out.append(line)

    by_baseline = build_baseline_map(configs)
    missing_off = [k for k, v in by_baseline.items() if v["off"] is None]

    emit(f"# Stage B analysis — {EXPERIMENT_ID}")
    emit(f"Canonical slice: asset=ALL, latency={CANON_LATENCY}ms, size=100 shares, horizon={CANON_HORIZON}ms")
    emit("24 baseline configs x 49 flow variants (1 OFF + 8 windows x 6 thresholds)")
    emit()
    if missing_off:
        emit(f"WARNING: {len(missing_off)} baseline keys have no OFF config -- grid mismatch: {missing_off}")
        emit()

    best_per_baseline, region_candidates = compute_deltas(con, EXPERIMENT_ID, by_baseline, emit)
    leading_region, region_winner_stats = rank_regions(region_candidates, emit)

    off_ids = [v["off"] for v in by_baseline.values() if v["off"] is not None]
    bucket_uplifts, bucket_monotonic = signed_flow_buckets(con, EXPERIMENT_ID, off_ids, emit)

    shape = horizon_shape(con, EXPERIMENT_ID, by_baseline, leading_region, emit)
    per_asset, asset_verdict = asset_breakdown(con, EXPERIMENT_ID, by_baseline, leading_region, emit)

    delta_ci = _approx_delta_ci(con, EXPERIMENT_ID, by_baseline, leading_region) if leading_region else None
    latency_deltas = _latency_breakdown(con, EXPERIMENT_ID, by_baseline, leading_region) if leading_region else None

    final_verdict(region_winner_stats, leading_region, bucket_monotonic, shape, per_asset, asset_verdict,
                  delta_ci, latency_deltas, emit)

    con.close()
    text = "\n".join(out)
    # Write the report to disk BEFORE touching stdout: a non-UTF8 console
    # codepage (default on plain Windows terminals) raises UnicodeEncodeError
    # on the Δ/— characters used throughout -- that must never cost us the
    # already-computed report after a multi-day run.
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(sys.stdout.encoding or "ascii", errors="replace").decode(sys.stdout.encoding or "ascii"))
    print(f"\n[report written to {OUT_PATH}, {len(out)} lines]", file=sys.stderr)


if __name__ == "__main__":
    main()
