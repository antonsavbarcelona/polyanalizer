"""Stage B winning-region validation: does flow_window=5000ms, signed_flow
>= 0.6 add INDEPENDENT information beyond momentum, or does it just
duplicate it (momentum-strong signals are already flow-strong almost by
construction)?

Triggered by a real discrepancy the user caught in the first Stage B
report: the signed-flow bucket table's ".6+" bucket held 9002/9880 (~91%)
of raw baseline signals, while the delta table's mean signal_retention for
this exact region showed 72.5%. Both numbers are real and both are
correctly computed -- but over DIFFERENT universes:
  - the 91% figure used flow_1s (analyze_stage_b.py's BUCKET_FLOW_FIELD,
    a deliberate simplification flagged in that script's own docstring) and
    pooled ALL 24 baselines' raw signal counts together;
  - the 72.5% figure used flow_5s (the winning region's actual window) and
    is the MEAN of 24 separate per-baseline ratios (flow_n / off_n), which
    is not the same arithmetic object as one pooled ratio.
This script recomputes everything on ONE universe -- flow_5s, the actual
window the winning region uses -- and reports numerator/denominator
explicitly so this kind of mismatch cannot hide again.

Sections:
  1. Reconciliation: flow_5s retention computed two ways (pooled ratio,
     mean of per-baseline ratios) side by side with the original flow_1s
     number, so the discrepancy has an explicit paper trail.
  2. Fine-grained (11-bucket) signed_flow_5s table with per-asset
     breakdown -- same sanity check as before, but on the actual winning
     window instead of a fixed 1s proxy.
  3. Paired RETAINED-vs-REJECTED comparison: classify every baseline OFF
     signal by its own flow_5s value (post-hoc, no re-detection needed --
     flow_5s is captured on every signal regardless of config) and compare
     response between the two groups directly. This is the central
     question: are REJECTED signals actually worse, or about the same?
  4. Per-baseline Δuplift table (baseline vs the flow_5s>=0.6 variant
     specifically, not each baseline's own "best" variant) with a
     24-baseline summary (count positive, median, P25/P75).
  5. Proper PAIRED bootstrap CI95 for the incremental Δuplift (resamples
     market_id once per iteration, applies that same draw to flow-signal,
     flow-control, off-signal, and off-control simultaneously) -- replaces
     the earlier "combine two independent CIs" approximation.
  6. corr(z_score, signed flow_5s) + flow_5s distribution within baseline
     signals -- direct evidence for "is flow just a relabeling of
     momentum."
  7. Verdict: FLOW_ADDS_INDEPENDENT_VALUE = YES / WEAK / NO /
     REDUNDANT_CONFIRMATION.

Run: python -m research.validate_stage_b_flow
"""
from __future__ import annotations

import json
import math
import random
import sqlite3
import sys
from collections import defaultdict
from statistics import median

from research.discovery.aggregation import compute_signal_response_summary
from research.discovery.reconstruct import control_response_from_row, signal_response_from_row
from research.discovery_types import ALL_ASSET

DB = "data/stage_b_flow_results.db"
EXPERIMENT_ID = None

CANON_LATENCY = 250
CANON_SIZE = 100.0
CANON_HORIZON = 2000
HORIZONS = (500, 1000, 2000, 3000, 5000)
ASSETS = ("BTC", "ETH", "SOL")

WINNER_WINDOW_MS = 5000
WINNER_THRESHOLD = 0.6
WINNER_FLOW_FIELD = "flow_5s"
OLD_BUCKET_FLOW_FIELD = "flow_1s"  # what analyze_stage_b.py's bucket table used -- kept for the reconciliation

FINE_BUCKETS = [
    (None, 0.0, "<0"), (0.0, 0.1, "0-.1"), (0.1, 0.2, ".1-.2"), (0.2, 0.3, ".2-.3"),
    (0.3, 0.4, ".3-.4"), (0.4, 0.5, ".4-.5"), (0.5, 0.6, ".5-.6"), (0.6, 0.7, ".6-.7"),
    (0.7, 0.8, ".7-.8"), (0.8, 0.9, ".8-.9"), (0.9, None, ".9+"),
]
BUCKET_MIN_N = 20
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 1337

OUT_PATH = "research/stage_b_flow_validation_report.md"

REDUNDANT_THRESHOLD = 0.85  # if >=85% of raw baseline signals already clear the winning flow threshold,
                             # the "confirmation" isn't really confirming much


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con


def fmt(v, pct=False, digits=4):
    if v is None:
        return "—"
    if pct:
        return f"{v * 100:.2f}%"
    return f"{v:.{digits}f}"


def bucket_of(signed_flow: float) -> str | None:
    for lo, hi, label in FINE_BUCKETS:
        if lo is not None and signed_flow < lo:
            continue
        if hi is not None and signed_flow >= hi:
            continue
        return label
    return None


def load_configs(con):
    return {r["signal_config_id"]: json.loads(r["config_json"])
            for r in con.execute("SELECT signal_config_id, config_json FROM signal_configs")}


def build_baseline_map(configs):
    by_baseline = {}
    for cfg_id, cfg in configs.items():
        key = (cfg["momentum_window_ms"], cfg["z_threshold"], cfg["volatility_window_ms"])
        entry = by_baseline.setdefault(key, {"off": None, "on": {}})
        if cfg["flow_window_ms"] is None:
            entry["off"] = cfg_id
        else:
            entry["on"][(cfg["flow_window_ms"], cfg["flow_threshold"])] = cfg_id
    return by_baseline


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


# ---------------------------------------------------------------------------
# Load every raw OFF-baseline signal with its snapshot-derived features
# ---------------------------------------------------------------------------

def load_off_signals(con, experiment_id, off_ids):
    id_ph = ",".join("?" * len(off_ids))
    rows = con.execute(
        f"""SELECT signal_id, signal_config_id, market_id, asset, direction, snapshot_json FROM signals
            WHERE experiment_id=? AND signal_config_id IN ({id_ph})""",
        (experiment_id, *off_ids),
    ).fetchall()
    out = []
    for r in rows:
        snap = json.loads(r["snapshot_json"])
        sign = 1.0 if r["direction"] == "UP" else -1.0
        flow_5s = snap.get(WINNER_FLOW_FIELD)
        flow_1s = snap.get(OLD_BUCKET_FLOW_FIELD)
        out.append({
            "signal_id": r["signal_id"], "signal_config_id": r["signal_config_id"],
            "market_id": r["market_id"], "asset": r["asset"], "direction": r["direction"],
            "signed_flow_5s": sign * flow_5s if flow_5s is not None else None,
            "signed_flow_1s": sign * flow_1s if flow_1s is not None else None,
            "z_score": snap.get("z_score"), "active_momentum": snap.get("active_momentum_value"),
        })
    return out


def load_responses_and_controls(con, off_ids):
    """One-shot load of every raw signal_response + control_response row
    for the 24 OFF configs at the canonical latency/size, across ALL
    horizons -- filtering/grouping happens in Python from here on, so this
    is the only pass over these (potentially large) tables."""
    id_ph = ",".join("?" * len(off_ids))
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
    control_source_of = {r["control_id"]: r["source_signal_id"] for r in ctrl_rows}

    cresp_rows = con.execute(
        f"""SELECT * FROM control_response WHERE signal_config_id IN ({id_ph})
            AND latency_ms=? AND size_shares=?""",
        (*off_ids, CANON_LATENCY, CANON_SIZE),
    ).fetchall()
    control_responses = [control_response_from_row(r) for r in cresp_rows]
    return responses, control_responses, control_source_of


def _percentile(sorted_values, pct):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(sorted_values) - 1)
    frac = k - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def summarize_group(responses, control_responses, horizon, asset_filter=None):
    """compute_signal_response_summary over an arbitrary (already-filtered)
    subset of raw rows -- the same pure aggregation the pipeline itself
    uses, just called on ad-hoc groupings (buckets / retained-vs-rejected)
    instead of a signal_config."""
    rs = [r for r in responses if r.horizon_ms == horizon and (asset_filter is None or r.asset == asset_filter)]
    cs = [r for r in control_responses if r.horizon_ms == horizon and (asset_filter is None or r.asset == asset_filter)]
    mkt_of = {id(r): r.market_id for r in rs}
    c_mkt_of = {id(r): r.market_id for r in cs}
    return compute_signal_response_summary(
        "adhoc", asset_filter or ALL_ASSET, CANON_LATENCY, CANON_SIZE, horizon, rs, mkt_of, cs, c_mkt_of,
        bootstrap_iterations=0,
    )


def _bootstrap_delta_ci(flow_responses, flow_controls, off_responses, off_controls,
                         *, iterations, seed, confidence=0.95):
    """Paired bootstrap for Δuplift = uplift(flow) - uplift(off), uplift(x)
    = mean(signal raw_response) - mean(control raw_response). Resamples
    market_ids ONCE per iteration and applies that SAME draw to all four
    groups -- not four independent resamples -- because flow/off signals
    and their controls from the same market are correlated; independent
    resampling would overstate the delta's true variance (same rationale
    as aggregation.py's own _bootstrap_ci)."""
    def by_market(responses):
        g = defaultdict(list)
        for r in responses:
            if r.status == "AVAILABLE" and r.raw_response is not None:
                g[r.market_id].append(r.raw_response)
        return g

    fg, fcg, og, ocg = (by_market(x) for x in (flow_responses, flow_controls, off_responses, off_controls))
    market_ids = sorted(set(fg) | set(fcg) | set(og) | set(ocg))
    if not market_ids or iterations <= 0:
        return None, None, None

    all_f = [v for vs in fg.values() for v in vs]
    all_fc = [v for vs in fcg.values() for v in vs]
    all_o = [v for vs in og.values() for v in vs]
    all_oc = [v for vs in ocg.values() for v in vs]
    if not (all_f and all_fc and all_o and all_oc):
        return None, None, None
    point_delta = (sum(all_f) / len(all_f) - sum(all_fc) / len(all_fc)) - \
                  (sum(all_o) / len(all_o) - sum(all_oc) / len(all_oc))

    rng = random.Random(seed)
    deltas = []
    for _ in range(iterations):
        drawn = [rng.choice(market_ids) for _ in market_ids]
        fs, fc, os_, oc = [], [], [], []
        for m in drawn:
            fs.extend(fg.get(m, [])); fc.extend(fcg.get(m, []))
            os_.extend(og.get(m, [])); oc.extend(ocg.get(m, []))
        if not (fs and fc and os_ and oc):
            continue
        deltas.append((sum(fs) / len(fs) - sum(fc) / len(fc)) - (sum(os_) / len(os_) - sum(oc) / len(oc)))

    if not deltas:
        return point_delta, None, None
    deltas.sort()
    alpha = (1 - confidence) / 2
    return point_delta, _percentile(deltas, alpha), _percentile(deltas, 1 - alpha)


def main() -> None:
    global EXPERIMENT_ID
    con = connect()
    EXPERIMENT_ID = con.execute("SELECT experiment_id FROM experiments").fetchone()["experiment_id"]
    configs = load_configs(con)
    by_baseline = build_baseline_map(configs)
    off_ids = [v["off"] for v in by_baseline.values() if v["off"] is not None]

    out: list[str] = []

    def emit(line=""):
        out.append(line)

    emit(f"# Stage B winning-region validation — {EXPERIMENT_ID}")
    emit(f"Winning region under test: flow_window={WINNER_WINDOW_MS}ms, signed_flow >= {WINNER_THRESHOLD} "
         f"(field: {WINNER_FLOW_FIELD})")
    emit()

    off_signals = load_off_signals(con, EXPERIMENT_ID, off_ids)
    responses, control_responses, control_source_of = load_responses_and_controls(con, off_ids)

    # ---- Section 1: reconciliation ----
    emit("## 1. Reconciling the 72.5% vs 91% discrepancy")
    emit()
    n_total = len(off_signals)
    n_with_flow5s = sum(1 for s in off_signals if s["signed_flow_5s"] is not None)
    n_retained_5s_pooled = sum(1 for s in off_signals if s["signed_flow_5s"] is not None
                                and s["signed_flow_5s"] >= WINNER_THRESHOLD)
    n_with_flow1s = sum(1 for s in off_signals if s["signed_flow_1s"] is not None)
    n_retained_1s_pooled = sum(1 for s in off_signals if s["signed_flow_1s"] is not None
                                and s["signed_flow_1s"] >= WINNER_THRESHOLD)

    per_baseline_ratios_5s = []
    per_baseline_ratios_1s = []
    for key, entry in sorted(by_baseline.items()):
        off_id = entry["off"]
        sigs = [s for s in off_signals if s["signal_config_id"] == off_id]
        n = len(sigs)
        if n == 0:
            continue
        r5 = sum(1 for s in sigs if s["signed_flow_5s"] is not None and s["signed_flow_5s"] >= WINNER_THRESHOLD) / n
        r1 = sum(1 for s in sigs if s["signed_flow_1s"] is not None and s["signed_flow_1s"] >= WINNER_THRESHOLD) / n
        per_baseline_ratios_5s.append(r5)
        per_baseline_ratios_1s.append(r1)

    # actual ON-config (5000ms/0.6) signal counts, for cross-checking that
    # post-hoc classification of OFF's raw flow_5s reproduces what the real
    # detector kept (rules out hysteresis-gate divergence as an explanation).
    actual_on_total = 0
    actual_off_total = 0
    for key, entry in sorted(by_baseline.items()):
        off_id = entry["off"]
        on_id = entry["on"].get((WINNER_WINDOW_MS, WINNER_THRESHOLD))
        if off_id is None or on_id is None:
            continue
        off_cs = config_summary_row(con, EXPERIMENT_ID, off_id, ALL_ASSET, CANON_LATENCY)
        on_cs = config_summary_row(con, EXPERIMENT_ID, on_id, ALL_ASSET, CANON_LATENCY)
        if off_cs is None or on_cs is None:
            continue
        actual_off_total += off_cs["signal_count"]
        actual_on_total += on_cs["signal_count"]

    emit(f"- Universe: all raw baseline (flow=OFF) signals across the 24 baselines, pooled BTC/ETH/SOL. "
         f"Total raw signals = {n_total}.")
    emit(f"- **flow_1s** (what the original bucket table used): {n_retained_1s_pooled}/{n_with_flow1s} "
         f"pooled >= {WINNER_THRESHOLD} = **{fmt(n_retained_1s_pooled / n_with_flow1s, pct=True) if n_with_flow1s else '—'}** "
         f"(this is the ~91% figure).")
    emit(f"- **flow_5s** (the winning region's actual window): {n_retained_5s_pooled}/{n_with_flow5s} "
         f"pooled >= {WINNER_THRESHOLD} = **{fmt(n_retained_5s_pooled / n_with_flow5s, pct=True) if n_with_flow5s else '—'}**.")
    emit(f"- flow_5s retention, mean of 24 PER-BASELINE ratios (what the delta table's 72.5% actually is): "
         f"**{fmt(sum(per_baseline_ratios_5s) / len(per_baseline_ratios_5s), pct=True)}**.")
    emit(f"- flow_1s retention, mean of 24 per-baseline ratios (for comparison): "
         f"**{fmt(sum(per_baseline_ratios_1s) / len(per_baseline_ratios_1s), pct=True)}**.")
    emit()
    emit(f"- Cross-check against the ACTUAL detector (not post-hoc classification): summed flow_window=5000/"
         f"threshold=0.6 signal_count across 24 baselines = {actual_on_total}, summed OFF signal_count = "
         f"{actual_off_total}, ratio = **{fmt(actual_on_total / actual_off_total, pct=True) if actual_off_total else '—'}**. "
         "This should closely match the pooled flow_5s post-hoc number above; a large gap would mean the "
         "HysteresisGate's armed/cooldown state (which depends on which candidates got confirmed, not just "
         "flow value) is materially changing WHICH signals get detected, not just filtering a fixed set.")
    emit()
    emit("**Conclusion: the 72.5% and ~91% numbers were never the same measurement** -- different flow window "
         "(1s vs 5s) AND different aggregation (mean-of-ratios vs pooled-ratio). Both were computed correctly "
         "for what they measured; the mismatch was a labeling/scope problem, not a math error. flow_5s is the "
         "relevant number for this region and is used throughout the rest of this report.")
    emit()

    # ---- Section 2: fine-grained flow_5s buckets ----
    emit("## 2. Signed-flow_5s bucket table (fine-grained, with per-asset breakdown)")
    emit()
    bucket_of_signal = {}
    bucket_asset_of_signal = {}
    bucket_markets = defaultdict(set)
    bucket_signal_count = defaultdict(int)
    for s in off_signals:
        if s["signed_flow_5s"] is None:
            continue
        label = bucket_of(s["signed_flow_5s"])
        if label is None:
            continue
        bucket_of_signal[s["signal_id"]] = label
        bucket_asset_of_signal[s["signal_id"]] = s["asset"]
        bucket_markets[label].add(s["market_id"])
        bucket_signal_count[label] += 1

    resp_by_bucket_h = defaultdict(lambda: defaultdict(list))
    for r in responses:
        label = bucket_of_signal.get(r.signal_id)
        if label is not None:
            resp_by_bucket_h[label][r.horizon_ms].append(r)
    ctrl_by_bucket_h = defaultdict(lambda: defaultdict(list))
    for r in control_responses:
        src = control_source_of.get(r.control_id)
        label = bucket_of_signal.get(src) if src else None
        if label is not None:
            ctrl_by_bucket_h[label][r.horizon_ms].append(r)

    order = [label for _, _, label in FINE_BUCKETS]
    header = ["signed_flow_5s", "signals", "markets"] + [f"uplift@{h}ms" for h in HORIZONS] + \
             ["BTC n", "ETH n", "SOL n"]
    emit("| " + " | ".join(header) + " |")
    emit("|" + "---|" * len(header))
    bucket_uplift_2s = {}
    for label in order:
        n = bucket_signal_count.get(label, 0)
        markets = len(bucket_markets.get(label, ()))
        cells = [label, str(n), str(markets)]
        row_uplifts = {}
        for h in HORIZONS:
            rs_ = resp_by_bucket_h.get(label, {}).get(h, [])
            cs_ = ctrl_by_bucket_h.get(label, {}).get(h, [])
            if not rs_:
                row_uplifts[h] = None
                cells.append("—")
                continue
            mkt_of = {id(x): x.market_id for x in rs_}
            c_mkt_of = {id(x): x.market_id for x in cs_}
            summ = compute_signal_response_summary(
                "adhoc", ALL_ASSET, CANON_LATENCY, CANON_SIZE, h, rs_, mkt_of, cs_, c_mkt_of,
                bootstrap_iterations=0,
            )
            row_uplifts[h] = summ.uplift_mean_response
            cells.append(fmt(summ.uplift_mean_response))
        bucket_uplift_2s[label] = row_uplifts.get(CANON_HORIZON)
        for asset in ASSETS:
            cnt = sum(1 for sid, lbl in bucket_of_signal.items() if lbl == label and bucket_asset_of_signal[sid] == asset)
            cells.append(str(cnt))
        emit("| " + " | ".join(cells) + " |")
    emit()

    values = [bucket_uplift_2s[label] for label in order if bucket_signal_count.get(label, 0) >= BUCKET_MIN_N
              and bucket_uplift_2s.get(label) is not None]
    if len(values) >= 3:
        steps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        frac_nd = sum(1 for s in steps if s >= 0) / len(steps)
        monotonic = "YES" if frac_nd >= 0.8 else ("PARTIAL" if frac_nd >= 0.5 else "NO")
    else:
        monotonic = f"INCONCLUSIVE (only {len(values)} buckets with >= {BUCKET_MIN_N} signals)"
    emit(f"Bucket monotonicity @{CANON_HORIZON}ms on flow_5s: **{monotonic}**")
    emit()

    # ---- Section 3: paired retained vs rejected ----
    emit("## 3. Paired RETAINED vs REJECTED (flow_5s >= 0.6 vs < 0.6), pooled across 24 baselines")
    emit()
    retained_ids = {s["signal_id"] for s in off_signals
                     if s["signed_flow_5s"] is not None and s["signed_flow_5s"] >= WINNER_THRESHOLD}
    rejected_ids = {s["signal_id"] for s in off_signals
                     if s["signed_flow_5s"] is not None and s["signed_flow_5s"] < WINNER_THRESHOLD}

    def split(responses_or_controls, id_set, is_control):
        if is_control:
            return [r for r in responses_or_controls if control_source_of.get(r.control_id) in id_set]
        return [r for r in responses_or_controls if r.signal_id in id_set]

    retained_resp = split(responses, retained_ids, False)
    rejected_resp = split(responses, rejected_ids, False)
    retained_ctrl = split(control_responses, retained_ids, True)
    rejected_ctrl = split(control_responses, rejected_ids, True)

    emit(f"RETAINED: {len(retained_ids)} signals. REJECTED: {len(rejected_ids)} signals.")
    emit()
    emit("| group | horizon | n available | uplift mean | uplift p_positive |")
    emit("|---|---|---|---|---|")
    retained_uplift_2s = rejected_uplift_2s = None
    for group_name, resp, ctrl in (("RETAINED", retained_resp, retained_ctrl), ("REJECTED", rejected_resp, rejected_ctrl)):
        for h in HORIZONS:
            summ = summarize_group(resp, ctrl, h)
            if h == CANON_HORIZON and group_name == "RETAINED":
                retained_uplift_2s = summ.uplift_mean_response
            if h == CANON_HORIZON and group_name == "REJECTED":
                rejected_uplift_2s = summ.uplift_mean_response
            emit(f"| {group_name} | {h}ms | {summ.available_count} | {fmt(summ.uplift_mean_response)} | "
                 f"{fmt(summ.uplift_p_positive, pct=True)} |")
    emit()
    if retained_uplift_2s is not None and rejected_uplift_2s is not None:
        emit(f"**RETAINED uplift@{CANON_HORIZON}ms = {fmt(retained_uplift_2s)}, "
             f"REJECTED uplift@{CANON_HORIZON}ms = {fmt(rejected_uplift_2s)}, "
             f"gap = {fmt(retained_uplift_2s - rejected_uplift_2s)}**")
    emit()
    emit("Per-asset breakdown:")
    emit()
    emit("| asset | RETAINED uplift@2s | REJECTED uplift@2s | gap |")
    emit("|---|---|---|---|")
    for asset in ASSETS:
        r_summ = summarize_group(retained_resp, retained_ctrl, CANON_HORIZON, asset)
        j_summ = summarize_group(rejected_resp, rejected_ctrl, CANON_HORIZON, asset)
        gap = (r_summ.uplift_mean_response - j_summ.uplift_mean_response) \
            if (r_summ.uplift_mean_response is not None and j_summ.uplift_mean_response is not None) else None
        emit(f"| {asset} | {fmt(r_summ.uplift_mean_response)} | {fmt(j_summ.uplift_mean_response)} | {fmt(gap)} |")
    emit()

    # ---- Section 4: per-baseline delta table for THIS SPECIFIC region ----
    emit("## 4. Per-baseline Δuplift for flow_window=5000ms/threshold=0.6 specifically")
    emit()
    emit("(Not each baseline's own \"best\" variant -- the SAME region for all 24, since that's the region "
         "under test.)")
    emit()
    emit("| baseline (mom/z/vol) | OFF uplift@2s | flow uplift@2s | Δuplift | sig_retain | mkt_retain |")
    emit("|---|---|---|---|---|---|")
    deltas_24 = []
    flow_responses_winner = []
    flow_controls_winner = []
    off_responses_winner = []
    off_controls_winner = []
    for key, entry in sorted(by_baseline.items()):
        off_id = entry["off"]
        on_id = entry["on"].get((WINNER_WINDOW_MS, WINNER_THRESHOLD))
        if off_id is None or on_id is None:
            emit(f"| {key} | — | — | — | — | — |")
            continue
        off_resp = response_row(con, EXPERIMENT_ID, off_id, ALL_ASSET, CANON_LATENCY, CANON_HORIZON)
        on_resp = response_row(con, EXPERIMENT_ID, on_id, ALL_ASSET, CANON_LATENCY, CANON_HORIZON)
        off_cs = config_summary_row(con, EXPERIMENT_ID, off_id, ALL_ASSET, CANON_LATENCY)
        on_cs = config_summary_row(con, EXPERIMENT_ID, on_id, ALL_ASSET, CANON_LATENCY)
        if not (off_resp and on_resp and off_cs and on_cs) or off_resp["uplift_mean_response"] is None \
                or on_resp["uplift_mean_response"] is None:
            emit(f"| {key} | — | — | no data | — | — |")
            continue
        delta = on_resp["uplift_mean_response"] - off_resp["uplift_mean_response"]
        sig_retain = on_cs["signal_count"] / off_cs["signal_count"] if off_cs["signal_count"] else None
        mkt_retain = on_cs["market_count"] / off_cs["market_count"] if off_cs["market_count"] else None
        deltas_24.append(delta)
        emit(f"| {key} | {fmt(off_resp['uplift_mean_response'])} | {fmt(on_resp['uplift_mean_response'])} | "
             f"{fmt(delta)} | {fmt(sig_retain, pct=True)} | {fmt(mkt_retain, pct=True)} |")

        # gather raw rows for the pooled paired bootstrap in section 5
        for r in con.execute(
            """SELECT * FROM signal_response WHERE signal_config_id=? AND latency_ms=?
               AND size_shares=? AND horizon_ms=?""",
            (on_id, CANON_LATENCY, CANON_SIZE, CANON_HORIZON),
        ):
            flow_responses_winner.append(signal_response_from_row(r))
        for r in con.execute(
            """SELECT * FROM control_response WHERE signal_config_id=? AND latency_ms=?
               AND size_shares=? AND horizon_ms=?""",
            (on_id, CANON_LATENCY, CANON_SIZE, CANON_HORIZON),
        ):
            flow_controls_winner.append(control_response_from_row(r))
        for r in responses:
            if r.signal_config_id == off_id and r.horizon_ms == CANON_HORIZON:
                off_responses_winner.append(r)
        # control_response rows already carry signal_config_id directly --
        # no need to join through controls.source_signal_id at all (the
        # first version of this did, via a set-comprehension re-evaluated
        # per outer-loop iteration -- an accidental O(24 * |controls| *
        # |off_signals|) ~ 10^10-op bug that never finished).
        for r in control_responses:
            if r.horizon_ms == CANON_HORIZON and r.signal_config_id == off_id:
                off_controls_winner.append(r)
    emit()
    n_pos = sum(1 for d in deltas_24 if d > 0)
    sorted_deltas = sorted(deltas_24)
    emit(f"**Summary across {len(deltas_24)} baselines: {n_pos}/{len(deltas_24)} positive "
         f"({fmt(n_pos / len(deltas_24), pct=True) if deltas_24 else '—'}). "
         f"median Δ = {fmt(median(sorted_deltas)) if sorted_deltas else '—'}, "
         f"P25 = {fmt(_percentile(sorted_deltas, 0.25))}, P75 = {fmt(_percentile(sorted_deltas, 0.75))}**")
    emit()

    # ---- Section 5: proper paired bootstrap CI for the pooled incremental delta ----
    emit("## 5. Paired bootstrap CI95 for the incremental Δuplift (pooled across 24 baselines)")
    emit()
    point_delta, ci_lo, ci_hi = _bootstrap_delta_ci(
        flow_responses_winner, flow_controls_winner, off_responses_winner, off_controls_winner,
        iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED,
    )
    emit(f"Δuplift@{CANON_HORIZON}ms (point estimate, pooled) = {fmt(point_delta)}")
    if ci_lo is not None:
        emit(f"95% CI (paired market-resampled bootstrap, {BOOTSTRAP_ITERATIONS} iterations) = "
             f"[{fmt(ci_lo)}, {fmt(ci_hi)}]")
        excludes_zero = (ci_lo > 0) or (ci_hi < 0)
        emit(f"Excludes zero: **{'YES' if excludes_zero else 'NO'}**")
    else:
        emit("CI not available (insufficient market overlap across the four groups).")
    emit()

    # ---- Section 6: correlation ----
    emit("## 6. corr(z_score, signed flow_5s) and flow_5s distribution")
    emit()
    pairs = [(s["z_score"], s["signed_flow_5s"]) for s in off_signals
             if s["z_score"] is not None and s["signed_flow_5s"] is not None]
    if len(pairs) >= 2:
        zs = [p[0] for p in pairs]
        fs = [p[1] for p in pairs]
        mz, mf = sum(zs) / len(zs), sum(fs) / len(fs)
        cov = sum((z - mz) * (f - mf) for z, f in pairs) / len(pairs)
        sz = (sum((z - mz) ** 2 for z in zs) / len(pairs)) ** 0.5
        sf = (sum((f - mf) ** 2 for f in fs) / len(pairs)) ** 0.5
        corr = cov / (sz * sf) if sz > 0 and sf > 0 else None
    else:
        corr = None
    emit(f"corr(z_score, signed flow_5s) over {len(pairs)} baseline signals: **{fmt(corr, digits=3)}**")
    emit()
    flow_vals = sorted(s["signed_flow_5s"] for s in off_signals if s["signed_flow_5s"] is not None)
    emit("flow_5s distribution within baseline (momentum-triggered) signals:")
    emit()
    emit("| P10 | P25 | P50 | P75 | P90 |")
    emit("|---|---|---|---|---|")
    emit("| " + " | ".join(fmt(_percentile(flow_vals, p)) for p in (0.10, 0.25, 0.50, 0.75, 0.90)) + " |")
    emit()

    # ---- Section 7: verdict ----
    emit("## 7. Verdict")
    emit()
    pooled_retention_5s = n_retained_5s_pooled / n_with_flow5s if n_with_flow5s else None
    gap = (retained_uplift_2s - rejected_uplift_2s) if (retained_uplift_2s is not None and rejected_uplift_2s is not None) else None
    ci_excludes_zero = ci_lo is not None and ((ci_lo > 0) or (ci_hi < 0))
    frac_positive_24 = n_pos / len(deltas_24) if deltas_24 else 0

    if pooled_retention_5s is not None and pooled_retention_5s >= REDUNDANT_THRESHOLD:
        verdict = "REDUNDANT_CONFIRMATION"
        reason = (f"{fmt(pooled_retention_5s, pct=True)} of raw baseline signals already clear "
                  f"signed_flow_5s>=0.6 -- momentum-strong signals are already flow-strong almost by "
                  f"construction, so this \"confirmation\" barely confirms anything.")
    elif gap is not None and gap > 0 and ci_excludes_zero and frac_positive_24 >= 0.8:
        verdict = "YES"
        reason = "rejected signals are clearly worse, the incremental delta excludes zero, and it holds " \
                  "consistently across baselines."
    elif gap is not None and gap > 0 and frac_positive_24 >= 0.6:
        verdict = "WEAK"
        reason = "rejected signals are somewhat worse and the sign is mostly consistent across baselines, " \
                  "but the pooled CI does not clear zero (or consistency is not strong enough for a clean YES)."
    else:
        verdict = "NO"
        reason = "rejected signals are not meaningfully worse than retained ones, or the sign is not " \
                  "consistent across baselines."

    emit(f"FLOW_ADDS_INDEPENDENT_VALUE = {verdict}")
    emit()
    emit(reason)
    emit()
    emit(f"Inputs: retained/rejected uplift gap = {fmt(gap)}, pooled bootstrap CI excludes zero = "
         f"{ci_excludes_zero}, baselines with positive Δ = {n_pos}/{len(deltas_24)}, "
         f"pooled flow_5s retention = {fmt(pooled_retention_5s, pct=True)}, "
         f"corr(z_score, flow_5s) = {fmt(corr, digits=3)}, flow_5s bucket monotonicity = {monotonic}.")

    con.close()
    text = "\n".join(out)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(sys.stdout.encoding or "ascii", errors="replace").decode(sys.stdout.encoding or "ascii"))
    print(f"\n[report written to {OUT_PATH}, {len(out)} lines]", file=sys.stderr)


if __name__ == "__main__":
    main()
