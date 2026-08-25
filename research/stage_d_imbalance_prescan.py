"""Stage D pre-scan: does direction-normalized Binance orderbook imbalance
add predictive information on top of the confirmed Stage B momentum+flow
region, before committing to a full threshold-sweep grid (same
pre-scan-first discipline as Stage C's volume check).

Unlike volume_z (Stage C), imbalance needs NO fresh computation: every
signal's snapshot already carries imbalance_top1/top3/top5/top10
(book_imbalance() at signal time, captured unconditionally by
research/discovery/detect.py's build_snapshot regardless of which config
triggered it -- see SignalSnapshot). This script only reads and buckets
already-stored data, so there's no per-market replay, no parallelism, and
no checkpointing to design -- it runs in a couple seconds.

signed_imbalance = direction * raw_imbalance, where raw_imbalance =
(bid_depth - ask_depth) / (bid_depth + ask_depth) at the given depth (top1/
3/5/10). UP flips nothing; DOWN flips sign, so a positive signed_imbalance
always means "book confirms this direction" regardless of which way it is.

Region signals: reuses stage_c_volume_prescan's exact region definition
(flow_window in {3000,5000}ms, threshold=0.4, union across 24 baselines,
deduplicated by physical instant) -- the same "confirmed Stage-B region"
population, for an apples-to-apples pre-scan methodology across Stage C/D.

Run: python -m research.stage_d_imbalance_prescan
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict

from research.discovery.aggregation import compute_signal_response_summary
from research.discovery.reconstruct import control_response_from_row, signal_response_from_row
from research.discovery_types import ALL_ASSET
from research.stage_c_volume_prescan import (
    CANON_LATENCY,
    CANON_SIZE,
    RESULTS_DB,
    load_configs,
    load_region_signals,
    region_config_ids,
)

HORIZONS = (500, 1000, 2000, 3000, 5000)
ASSETS = ("BTC", "ETH", "SOL")
DEPTH_FIELDS = ("imbalance_top1", "imbalance_top3", "imbalance_top5", "imbalance_top10")

IMBALANCE_BUCKETS = [
    (None, -0.4, "<-0.4"), (-0.4, -0.2, "-0.4:-0.2"), (-0.2, 0.0, "-0.2:0"),
    (0.0, 0.1, "0:.1"), (0.1, 0.2, ".1:.2"), (0.2, 0.3, ".2:.3"),
    (0.3, 0.4, ".3:.4"), (0.4, 0.5, ".4:.5"), (0.5, None, ".5+"),
]
BUCKET_MIN_N = 20

OUT_PATH = "research/stage_d_imbalance_prescan_report.md"


def fmt(v, pct=False, digits=4):
    if v is None:
        return "—"
    if pct:
        return f"{v * 100:.2f}%"
    return f"{v:.{digits}f}"


def bucket_of(v: float) -> str | None:
    for lo, hi, label in IMBALANCE_BUCKETS:
        if lo is not None and v < lo:
            continue
        if hi is not None and v >= hi:
            continue
        return label
    return None


def load_responses_and_controls(con, cfg_ids):
    id_ph = ",".join("?" * len(cfg_ids))
    resp_rows = con.execute(
        f"""SELECT * FROM signal_response WHERE signal_config_id IN ({id_ph})
            AND latency_ms=? AND size_shares=?""",
        (*cfg_ids, CANON_LATENCY, CANON_SIZE),
    ).fetchall()
    responses = [signal_response_from_row(r) for r in resp_rows]
    cresp_rows = con.execute(
        f"""SELECT * FROM control_response WHERE signal_config_id IN ({id_ph})
            AND latency_ms=? AND size_shares=?""",
        (*cfg_ids, CANON_LATENCY, CANON_SIZE),
    ).fetchall()
    control_responses = [control_response_from_row(r) for r in cresp_rows]
    ctrl_rows = con.execute(
        f"""SELECT control_id, source_signal_id FROM controls WHERE signal_config_id IN ({id_ph})""",
        cfg_ids,
    ).fetchall()
    control_source_of = {r["control_id"]: r["source_signal_id"] for r in ctrl_rows}
    return responses, control_responses, control_source_of


def main() -> None:
    con = sqlite3.connect(RESULTS_DB)
    con.row_factory = sqlite3.Row
    experiment_id = con.execute("SELECT experiment_id FROM experiments").fetchone()["experiment_id"]
    configs = load_configs(con)
    cfg_ids = region_config_ids(configs)
    region_signals = load_region_signals(con, experiment_id, cfg_ids)

    # pull direction + snapshot for each region signal's representative signal_id
    rep_ids = [s["signal_id"] for s in region_signals]
    id_ph = ",".join("?" * len(rep_ids))
    snap_rows = con.execute(
        f"SELECT signal_id, direction, snapshot_json FROM signals WHERE signal_id IN ({id_ph})", rep_ids,
    ).fetchall()
    snapshots = {}
    for r in snap_rows:
        snap = json.loads(r["snapshot_json"])
        sign = 1.0 if r["direction"] == "UP" else -1.0
        snapshots[r["signal_id"]] = {f: (sign * snap[f] if snap.get(f) is not None else None) for f in DEPTH_FIELDS}

    asset_of_signal = {s["signal_id"]: s["asset"] for s in region_signals}

    responses, control_responses, control_source_of = load_responses_and_controls(con, cfg_ids)
    rep_set = set(rep_ids)
    responses = [r for r in responses if r.signal_id in rep_set]
    control_responses = [r for r in control_responses if control_source_of.get(r.control_id) in rep_set]

    out: list[str] = []

    def emit(line=""):
        out.append(line)

    emit(f"# Stage D imbalance pre-scan — {experiment_id}")
    emit(f"Region: same as Stage C (flow_window in {{3000,5000}}ms, threshold=0.4, 24 baselines, "
         f"deduplicated by physical instant). {len(region_signals)} signals.")
    emit()

    order = [label for _, _, label in IMBALANCE_BUCKETS]
    monotonic_by_depth = {}

    for depth in DEPTH_FIELDS:
        emit(f"## {depth} (direction-normalized)")
        emit()

        bucket_of_signal = {}
        bucket_markets = defaultdict(set)
        bucket_count = defaultdict(int)
        for s in region_signals:
            v = snapshots.get(s["signal_id"], {}).get(depth)
            if v is None:
                continue
            label = bucket_of(v)
            if label is None:
                continue
            bucket_of_signal[s["signal_id"]] = label
            bucket_markets[label].add(s["market_id"])
            bucket_count[label] += 1

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

        header = ["signed_imbalance", "signals", "markets"] + [f"uplift@{h}ms" for h in HORIZONS] + \
                 ["BTC n", "ETH n", "SOL n"]
        emit("| " + " | ".join(header) + " |")
        emit("|" + "---|" * len(header))
        uplift_2s = {}
        for label in order:
            n = bucket_count.get(label, 0)
            markets = len(bucket_markets.get(label, ()))
            cells = [label, str(n), str(markets)]
            for h in HORIZONS:
                rs_ = resp_by_bucket_h.get(label, {}).get(h, [])
                cs_ = ctrl_by_bucket_h.get(label, {}).get(h, [])
                if not rs_:
                    cells.append("—")
                    continue
                mkt_of = {id(x): x.market_id for x in rs_}
                c_mkt_of = {id(x): x.market_id for x in cs_}
                summ = compute_signal_response_summary(
                    "adhoc", ALL_ASSET, CANON_LATENCY, CANON_SIZE, h, rs_, mkt_of, cs_, c_mkt_of,
                    bootstrap_iterations=0,
                )
                if h == 2000:
                    uplift_2s[label] = summ.uplift_mean_response
                cells.append(fmt(summ.uplift_mean_response))
            for asset in ASSETS:
                cnt = sum(1 for sid, lbl in bucket_of_signal.items()
                          if lbl == label and asset_of_signal.get(sid) == asset)
                cells.append(str(cnt))
            emit("| " + " | ".join(cells) + " |")
        emit()

        values = [uplift_2s[label] for label in order
                  if bucket_count.get(label, 0) >= BUCKET_MIN_N and uplift_2s.get(label) is not None]
        if len(values) >= 3:
            steps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
            frac_nd = sum(1 for s in steps if s >= 0) / len(steps)
            monotonic = "YES" if frac_nd >= 0.8 else ("PARTIAL" if frac_nd >= 0.5 else "NO")
        else:
            monotonic = f"INCONCLUSIVE ({len(values)} buckets with >= {BUCKET_MIN_N} signals)"
        monotonic_by_depth[depth] = monotonic
        emit(f"Monotonicity @2s: **{monotonic}** (n buckets with data: {len(values)})")
        emit()

    # ---- verdict ----
    emit("## Verdict")
    emit()
    clean = [d for d, m in monotonic_by_depth.items() if m == "YES"]
    partial = [d for d, m in monotonic_by_depth.items() if m == "PARTIAL"]
    if clean:
        verdict = "YES"
        reason = f"clean monotonic relation found for depth(s): {clean}"
    elif len(partial) >= len(monotonic_by_depth) / 2:
        verdict = "WEAK"
        reason = f"partial/inconsistent monotonicity across most depths ({len(partial)}/{len(monotonic_by_depth)} PARTIAL)"
    else:
        verdict = "NO"
        reason = "no depth showed a clean monotonic imbalance-response relation"
    emit(f"IMBALANCE_ADDS_VALUE = {verdict}")
    emit()
    emit(reason)
    emit()
    if verdict == "NO":
        emit("Recommendation: do not build a full Stage D threshold-sweep grid. Move to Stage E "
             "(Polymarket repricing lag) directly.")
    else:
        emit("Recommendation: proceed to a full Stage D threshold sweep on the depth(s) that showed structure.")

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
