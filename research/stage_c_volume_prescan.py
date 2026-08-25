"""Stage C pre-scan: does aggressive Binance trade VOLUME add predictive
information on top of the confirmed Stage B momentum+flow region, before
committing to a full threshold-sweep grid (task: "сначала дешёвая
conditional analysis -> потом expensive search только если есть
структура")?

volume_z is NOT part of any SignalSnapshot captured during Stage B (no
Stage B config ever swept volume_window_ms), so it's computed fresh here
directly from each market's raw binance_trades tape -- same definition as
research/discovery/detect.py's _active_volume_z (current windowed volume
vs the mean/stdev of resampled tumbling-window volumes over a lookback),
just with an EXPLICIT lookback instead of the pipeline's derived
max(60s, 10*window) formula, per the task's request.

Region signals: the union of the two loosest members of the declared
Stage B robust neighborhood -- flow_window in {3000, 5000}, threshold=0.4
-- across all 24 baselines. Deduplicated by physical identity (market_id,
direction, signal_ts), not signal_id, since the SAME instant gets a
DIFFERENT signal_id per config (id hash includes signal_config_id) and
would otherwise be double-counted where the two window variants overlap.

Parallelism: one worker process per (asset, market_id) -- each asset has
only 5 raw markets (15 total), so this is a small, fast, embarrassingly-
parallel job; still checkpointed to a local JSON cache file per (asset,
market) so a crash mid-run doesn't lose already-computed markets and
progress is visible as it happens (not one opaque blocking pass).

Run: python -m research.stage_c_volume_prescan
"""
from __future__ import annotations

import bisect
import json
import logging
import os
import sqlite3
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict as dc_asdict

from research.data.validator import open_readonly
from research.discovery.aggregation import compute_signal_response_summary
from research.discovery.reconstruct import control_response_from_row, signal_response_from_row
from research.discovery_types import ALL_ASSET

log = logging.getLogger(__name__)

RESULTS_DB = "data/stage_b_flow_results.db"
ASSET_DB = {"BTC": "data/analyzer_btc.db", "ETH": "data/analyzer_eth.db", "SOL": "data/analyzer_sol.db"}

CANON_LATENCY = 250
CANON_SIZE = 100.0
HORIZONS = (500, 1000, 2000, 3000, 5000)
ASSETS = ("BTC", "ETH", "SOL")

REGION_FLOW_VARIANTS = ((3000, 0.4), (5000, 0.4))  # loosest members of the declared Stage B plateau

VOLUME_WINDOWS_MS = (500, 1_000, 2_000, 5_000)
LOOKBACKS_MS = (60_000, 120_000)
MIN_BUCKET_SAMPLES = 5  # minimum resampled tumbling-window count for a volume_z estimate to be trusted

VOLUME_Z_BUCKETS = [
    (None, 0.0, "<0"), (0.0, 0.5, "0-.5"), (0.5, 1.0, ".5-1"), (1.0, 1.5, "1-1.5"),
    (1.5, 2.0, "1.5-2"), (2.0, 2.5, "2-2.5"), (2.5, 3.0, "2.5-3"), (3.0, None, "3+"),
]
BUCKET_MIN_N = 20

CACHE_DIR = "research/.stage_c_volume_cache"
OUT_PATH = "research/stage_c_volume_prescan_report.md"

MAX_WORKERS = 24


def fmt(v, pct=False, digits=4):
    if v is None:
        return "—"
    if pct:
        return f"{v * 100:.2f}%"
    return f"{v:.{digits}f}"


def bucket_of(z: float) -> str | None:
    for lo, hi, label in VOLUME_Z_BUCKETS:
        if lo is not None and z < lo:
            continue
        if hi is not None and z >= hi:
            continue
        return label
    return None


# ---------------------------------------------------------------------------
# Region signal identification (dedup by physical identity)
# ---------------------------------------------------------------------------

def load_configs(con):
    return {r["signal_config_id"]: json.loads(r["config_json"])
            for r in con.execute("SELECT signal_config_id, config_json FROM signal_configs")}


def region_config_ids(configs) -> list[str]:
    ids = []
    for cfg_id, cfg in configs.items():
        if (cfg["flow_window_ms"], cfg["flow_threshold"]) in REGION_FLOW_VARIANTS:
            ids.append(cfg_id)
    return ids


def load_region_signals(con, experiment_id, cfg_ids):
    """One row per PHYSICAL (market_id, asset, direction, signal_ts) --
    first-seen signal_id kept as the representative (response VALUES are
    proven config-independent, see research/discovery/shared_cache.py's
    docstring, so any one of the duplicate signal_ids' response rows is
    exactly as valid as any other)."""
    id_ph = ",".join("?" * len(cfg_ids))
    rows = con.execute(
        f"""SELECT signal_id, signal_config_id, market_id, asset, direction, signal_ts FROM signals
            WHERE experiment_id=? AND signal_config_id IN ({id_ph})
            ORDER BY market_id, direction, signal_ts""",
        (experiment_id, *cfg_ids),
    ).fetchall()
    seen: dict[tuple, dict] = {}
    for r in rows:
        key = (r["market_id"], r["asset"], r["direction"], r["signal_ts"])
        if key not in seen:
            seen[key] = {"signal_id": r["signal_id"], "market_id": r["market_id"], "asset": r["asset"],
                          "direction": r["direction"], "signal_ts": r["signal_ts"]}
    return list(seen.values())


# ---------------------------------------------------------------------------
# Volume_z computation (worker-side, one (asset, market_id) job)
# ---------------------------------------------------------------------------

def _range_sum(ts_list, prefix, t0, t1) -> float:
    i0 = bisect.bisect_left(ts_list, t0)
    i1 = bisect.bisect_left(ts_list, t1)
    return prefix[i1] - prefix[i0]


def _volume_z(ts_list, prefix, now_ms: int, window_ms: int, lookback_ms: int) -> float | None:
    n_buckets = lookback_ms // window_ms
    if n_buckets < MIN_BUCKET_SAMPLES:
        return None
    sums = []
    t = now_ms - lookback_ms
    for _ in range(n_buckets):
        t1 = t + window_ms
        sums.append(_range_sum(ts_list, prefix, t, t1))
        t = t1
    mean = sum(sums) / len(sums)
    if len(sums) < 2:
        return None
    var = sum((s - mean) ** 2 for s in sums) / (len(sums) - 1)
    sigma = var ** 0.5
    if sigma <= 0:
        return None
    current = _range_sum(ts_list, prefix, now_ms - window_ms, now_ms)
    return (current - mean) / sigma


def _worker(asset: str, market_id: str, db_path: str, signal_ts_list: list[int]) -> tuple[str, str, dict]:
    """Returns (asset, market_id, {signal_ts: {"window_ms,lookback_ms": z_or_None}})."""
    conn = open_readonly(db_path)
    try:
        rows = conn.execute(
            "SELECT exchange_ts, qty FROM binance_trades WHERE market_id=? ORDER BY exchange_ts",
            (market_id,),
        ).fetchall()
    finally:
        conn.close()
    ts_list = [r[0] for r in rows]
    prefix = [0.0]
    for r in rows:
        prefix.append(prefix[-1] + r[1])

    out: dict[str, dict[str, float | None]] = {}
    for now_ms in signal_ts_list:
        combos = {}
        for window_ms in VOLUME_WINDOWS_MS:
            for lookback_ms in LOOKBACKS_MS:
                combos[f"{window_ms},{lookback_ms}"] = _volume_z(ts_list, prefix, now_ms, window_ms, lookback_ms)
        out[str(now_ms)] = combos
    return asset, market_id, out


def compute_volume_z_for_region(region_signals: list[dict]) -> dict[str, dict[str, dict[str, float | None]]]:
    """Returns signal_id -> {"window_ms,lookback_ms": z}. Checkpointed to
    CACHE_DIR per (asset, market_id) -- a rerun skips markets already
    cached on disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    by_market: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in region_signals:
        by_market[(s["asset"], s["market_id"])].append(s)

    jobs = []
    cached: dict[tuple[str, str], dict] = {}
    for (asset, market_id), sigs in by_market.items():
        cache_path = os.path.join(CACHE_DIR, f"{asset}_{market_id}.json")
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                cached[(asset, market_id)] = json.load(f)
            log.info("cache hit for %s/%s (%d signals)", asset, market_id, len(sigs))
        else:
            jobs.append((asset, market_id, sigs))

    if jobs:
        log.info("computing volume_z for %d/%d (asset, market) jobs across %d workers",
                  len(jobs), len(by_market), min(len(jobs), MAX_WORKERS))
        with ProcessPoolExecutor(max_workers=min(len(jobs), MAX_WORKERS)) as pool:
            futures = {
                pool.submit(_worker, asset, market_id, ASSET_DB[asset],
                            [s["signal_ts"] for s in sigs]): (asset, market_id)
                for asset, market_id, sigs in jobs
            }
            for future in as_completed(futures):
                asset, market_id = futures[future]
                _asset, _market_id, result = future.result()
                cache_path = os.path.join(CACHE_DIR, f"{asset}_{market_id}.json")
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(result, f)
                cached[(asset, market_id)] = result
                log.info("done %s/%s (%d signal timestamps)", asset, market_id, len(result))

    out: dict[str, dict[str, float | None]] = {}
    for s in region_signals:
        market_result = cached.get((s["asset"], s["market_id"]), {})
        out[s["signal_id"]] = market_result.get(str(s["signal_ts"]), {})
    return out


# ---------------------------------------------------------------------------
# Response bucketing (reuses the pipeline's own aggregation function)
# ---------------------------------------------------------------------------

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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s INFO %(message)s")
    con = sqlite3.connect(RESULTS_DB)
    con.row_factory = sqlite3.Row
    experiment_id = con.execute("SELECT experiment_id FROM experiments").fetchone()["experiment_id"]
    configs = load_configs(con)
    cfg_ids = region_config_ids(configs)
    log.info("region config ids: %d (flow_window in {3000,5000}, threshold=0.4, x 24 baselines)", len(cfg_ids))

    region_signals = load_region_signals(con, experiment_id, cfg_ids)
    log.info("region signals (deduplicated by physical identity): %d", len(region_signals))

    volume_z_by_signal = compute_volume_z_for_region(region_signals)

    responses, control_responses, control_source_of = load_responses_and_controls(con, cfg_ids)
    # Only keep response rows belonging to our representative (deduped)
    # signal_ids -- the region config set's raw responses include the
    # OTHER (non-representative) duplicate signal_ids too, which must be
    # excluded here to avoid double-counting the same physical instant.
    rep_signal_ids = {s["signal_id"] for s in region_signals}
    responses = [r for r in responses if r.signal_id in rep_signal_ids]
    control_responses = [r for r in control_responses
                          if control_source_of.get(r.control_id) in rep_signal_ids]

    asset_of_signal = {s["signal_id"]: s["asset"] for s in region_signals}
    market_of_signal = {s["signal_id"]: s["market_id"] for s in region_signals}

    out: list[str] = []

    def emit(line=""):
        out.append(line)

    emit(f"# Stage C volume pre-scan — {experiment_id}")
    emit(f"Region: flow_window in {{3000,5000}}ms, threshold=0.4, across 24 baselines "
         f"(union, deduplicated by physical instant). {len(region_signals)} signals.")
    emit()

    verdict_evidence = []

    for window_ms in VOLUME_WINDOWS_MS:
        for lookback_ms in LOOKBACKS_MS:
            combo_key = f"{window_ms},{lookback_ms}"
            emit(f"## volume_window={window_ms}ms, lookback={lookback_ms}ms")
            emit()

            bucket_of_signal = {}
            bucket_markets = defaultdict(set)
            bucket_count = defaultdict(int)
            for s in region_signals:
                z = volume_z_by_signal.get(s["signal_id"], {}).get(combo_key)
                if z is None:
                    continue
                label = bucket_of(z)
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

            order = [label for _, _, label in VOLUME_Z_BUCKETS]
            header = ["volume_z", "signals", "markets"] + [f"uplift@{h}ms" for h in HORIZONS] + \
                     ["BTC n", "ETH n", "SOL n"]
            emit("| " + " | ".join(header) + " |")
            emit("|" + "---|" * len(header))
            uplift_2s_by_bucket = {}
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
                        uplift_2s_by_bucket[label] = summ.uplift_mean_response
                    cells.append(fmt(summ.uplift_mean_response))
                for asset in ASSETS:
                    cnt = sum(1 for sid, lbl in bucket_of_signal.items()
                              if lbl == label and asset_of_signal.get(sid) == asset)
                    cells.append(str(cnt))
                emit("| " + " | ".join(cells) + " |")
            emit()

            values = [uplift_2s_by_bucket[label] for label in order
                      if bucket_count.get(label, 0) >= BUCKET_MIN_N and uplift_2s_by_bucket.get(label) is not None]
            if len(values) >= 3:
                steps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
                frac_nd = sum(1 for s in steps if s >= 0) / len(steps)
                monotonic = "YES" if frac_nd >= 0.8 else ("PARTIAL" if frac_nd >= 0.5 else "NO")
            else:
                monotonic = f"INCONCLUSIVE ({len(values)} buckets with >= {BUCKET_MIN_N} signals)"
            emit(f"Monotonicity @2s: **{monotonic}** (n buckets with data: {len(values)})")
            emit()
            verdict_evidence.append((window_ms, lookback_ms, monotonic, values))

    # ---- verdict ----
    emit("## Verdict")
    emit()
    strong = [(w, l) for w, l, m, v in verdict_evidence if m == "YES"]
    partial = [(w, l) for w, l, m, v in verdict_evidence if m == "PARTIAL"]
    if strong:
        verdict = "YES"
        reason = f"clean monotonic relation found for volume_window/lookback = {strong}"
    elif len(partial) >= len(verdict_evidence) / 2:
        verdict = "WEAK"
        reason = f"partial/inconsistent monotonicity across most window/lookback combos ({len(partial)}/{len(verdict_evidence)} PARTIAL)"
    else:
        verdict = "NO"
        reason = "no window/lookback combo showed a clean monotonic volume-response relation"
    emit(f"VOLUME_ADDS_VALUE = {verdict}")
    emit()
    emit(reason)
    emit()
    if verdict == "NO":
        emit("Recommendation: do not build a full Stage C threshold-sweep grid. Move to Stage D "
             "(Binance imbalance) with this same pre-scan-first discipline.")
    else:
        emit("Recommendation: proceed to a full Stage C threshold sweep on the window/lookback combo(s) "
             "that showed structure.")

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
