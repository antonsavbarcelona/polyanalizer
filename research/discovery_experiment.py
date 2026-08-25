"""Signal-response discovery orchestrator (IMPLEMENTATION CONTRACT #1,
#39-41; task items 1-4). TP/SL/timeout/exit-policy/portfolio code must
never be imported here -- the only question this module answers is what
happened to the executable Polymarket price after a potential Binance
signal.

Parallelism: one worker process per (baseline, asset) job -- a "baseline"
is every SignalDiscoveryConfig field EXCEPT flow_window_ms/flow_threshold,
so all of a baseline's flow variants (1 OFF + however many window x
threshold combos the grid sweeps) land in the SAME job. Checkpointing
still happens at the finer (signal_config, asset) grain (task item 3: a
BTC/ETH COMPLETE + SOL FAILED run only ever recomputes SOL on --resume,
and a job that's only partially checkpoint-complete skips its already-done
configs instead of redoing them) -- grouping by baseline is purely a perf
optimization on top of that, not a change to what gets persisted or how
resume works. Single writer in the parent throughout (contract #39 / task
item 4 forbid concurrent SQLite writers). Once every asset for a given
signal_config is COMPLETE, the parent pools that config's per-asset raw
rows (read back from the DB, not from in-memory worker state, so pooling
is resume-safe across separate runs) into the asset=NULL "ALL" summary
rows.

Why group by baseline: profiling one real Stage B unit showed ~91% of its
time in compute_entry_mark/compute_signal_path (entry.py/path_walk.py),
called once per signal and once per every matched control -- both PROVEN
pure functions of (market_id, direction, ts, latency_ms, size_shares,
market_row, fee_model), never of signal_config_id/flow settings (see
research/discovery/shared_cache.py's docstring for the full argument).
Grouping a baseline's flow variants into one worker job lets them share one
EntryResponsePathCache, so a (market, direction, ts) instant computed once
for the first variant that hits it is free for every later variant that
hits the same instant -- while detect_signals (per-config trigger/gate
timing) and control selection/ranking (controls.py -- genuinely
config-specific: eligibility depends on whether THIS config would itself
fire there, exclusion depends on THIS config's own signal timing) stay
byte-identical to running each variant fully independently. Regression-
tested in tests/test_signal_discovery_shared_cache.py.

MAX_WORKERS is capped, not os.cpu_count() -- the earlier "16 workers never
finish spawning" symptom turned out to be HDD contention during import (the
whole project + data lived on a spinning disk), not a hard process-count
ceiling: 8 raw processes AND 8 workers importing this module's full chain
both spawned in ~1s once isolated from the hang. Data now lives on the
dev machine's SSD (see data/ junction).

Capped AT the dev machine's 12-core/24-thread count (was briefly set to 32,
which oversubscribed the CPU by 8 processes -- pure context-switch/cache
overhead with no added parallelism past the thread count, plus starving
every other process on the machine). Raise only after confirming on
hardware with more threads; do not just bump the constant.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any

from poly_analyzer.discovery import DiscoverySignalConfig as ReplayBufferSizing
from poly_analyzer.discovery import iter_replay, list_market_ids, load_market_meta
from poly_analyzer.features import realized_vol

from research.discovery.aggregation import (
    compute_first_passage_summary,
    compute_hit_summary,
    compute_signal_config_summary,
    compute_signal_response_summary,
)
from research.discovery.controls import scan_candidates, select_matched_controls_from_candidates
from research.discovery.detect import detect_signals
from research.discovery.plateau import compute_plateau_metrics, find_neighbors
from research.discovery.reconstruct import (
    control_response_from_row,
    entry_mark_from_row,
    path_stats_from_row,
    signal_response_from_row,
)
from research.discovery.regimes import compute_volatility_boundaries
from research.discovery.response import HORIZONS_MS
from research.discovery.shared_cache import EntryResponsePathCache
from research.discovery_config import SignalDiscoveryExperimentConfig
from research.discovery_grid import generate_signal_discovery_configs
from research.discovery_types import (
    ALL_ASSET,
    Control,
    ControlEntryMark,
    ControlResponse,
    DiscoveryEntryMark,
    SignalConfigSummary,
    SignalDiscoveryConfig,
    SignalFirstPassageSummary,
    SignalHitSummary,
    SignalPathStats,
    SignalResponse,
    SignalResponseSummary,
    SignalSnapshot,
    VolatilityRegimeBoundaries,
    canonical_json,
)
from research.data.validator import fingerprint_db, open_readonly, validate_collector_db
from research.fees import FeeModel
from research.storage.discovery_repository import DiscoveryRepository

log = logging.getLogger(__name__)

MAX_WORKERS = 24

# First-passage level pairs (contract #17): X,Y in {0.01..0.05}.
FIRST_PASSAGE_LEVELS: tuple[float, ...] = (0.01, 0.02, 0.03, 0.04, 0.05)

# path_stats/hit/first-passage horizons reuse the full fixed response-horizon
# set (contract #3) for simplicity/correctness on this dataset's scale --
# now cheap regardless, since compute_signal_path answers every one of
# these in a single pass over the future path (see path_walk.py).
STATS_HORIZONS_MS: tuple[int, ...] = HORIZONS_MS


@dataclass
class SignalConfigAssetResult:
    """Everything one (signal_config, asset) worker computed -- including
    that asset's OWN (non-pooled) summary rows, which don't need to wait
    for other assets. The parent pools the asset=NULL "ALL" rows separately
    once every asset for this config is COMPLETE (see _pool_config_summaries_compute /
    _pool_config_worker -- parallelized + checkpointed the same way as the
    main dispatch loop, not one giant single-threaded transaction)."""

    signal_config_id: str
    asset: str
    market_ids: set[str] = field(default_factory=set)
    signals: list[SignalSnapshot] = field(default_factory=list)
    entry_marks: list[DiscoveryEntryMark] = field(default_factory=list)
    responses: list[SignalResponse] = field(default_factory=list)
    path_stats: list[SignalPathStats] = field(default_factory=list)
    controls: list[Control] = field(default_factory=list)
    control_entry_marks: list[ControlEntryMark] = field(default_factory=list)
    control_responses: list[ControlResponse] = field(default_factory=list)
    config_summaries: list[SignalConfigSummary] = field(default_factory=list)
    response_summaries: list[SignalResponseSummary] = field(default_factory=list)
    hit_summaries: list[SignalHitSummary] = field(default_factory=list)
    first_passage_summaries: list[SignalFirstPassageSummary] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)


def _market_row(conn, market_id: str, asset: str) -> dict[str, Any]:
    row = load_market_meta(conn, market_id) or {"market_id": market_id}
    row.setdefault("asset", asset)
    if row.get("fee_rate") is None:
        row["fee_rate"] = 0.07
    if row.get("fee_exponent") is None:
        row["fee_exponent"] = 1.0
    return row


def _baseline_key(cfg: SignalDiscoveryConfig) -> tuple:
    """Every SignalDiscoveryConfig field except id/flow_window_ms/
    flow_threshold -- two configs share a baseline iff they differ ONLY in
    their flow confirmation. Stage A (no flow sweep at all) puts every
    config in its own singleton group, so this is a no-op there; Stage B's
    49 flow variants per momentum/z/vol combo land in one group."""
    d = asdict(cfg)
    d.pop("id")
    d.pop("flow_window_ms")
    d.pop("flow_threshold")
    return tuple(sorted(d.items()))


def _run_baseline_asset(
    baseline_configs: list[SignalDiscoveryConfig],
    already_complete_ids: set[str],
    asset: str,
    db_path: str,
    max_markets_per_asset: int | None,
    latency_grid_ms: tuple[int, ...],
    size_grid_shares: tuple[float, ...],
    controls_per_signal: int,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    vol_boundary: VolatilityRegimeBoundaries | None,
) -> tuple[dict[str, SignalConfigAssetResult], dict[str, float]]:
    """One worker job = every not-yet-COMPLETE flow variant of one baseline,
    for one asset. Detection (detect_signals) and control selection/ranking
    (scan_candidates/select_matched_controls_from_candidates) are run fully
    independently per config -- byte-identical to the old one-job-per-config
    version. Only entry-mark + future-path computation (for both signals
    and controls) routes through a cache SHARED across every config in this
    job, since that part is provably config-independent (see
    research/discovery/shared_cache.py)."""
    t0 = time.perf_counter()
    pending_configs = [c for c in baseline_configs if c.id not in already_complete_ids]
    results: dict[str, SignalConfigAssetResult] = {
        c.id: SignalConfigAssetResult(signal_config_id=c.id, asset=asset) for c in pending_configs
    }
    fee_model = FeeModel()
    entry_cache = EntryResponsePathCache()

    entry_marks_by_key: dict[str, dict[tuple[int, float], list[DiscoveryEntryMark]]] = {
        c.id: defaultdict(list) for c in pending_configs
    }
    responses_by_key: dict[str, dict[tuple[int, float, int], list[SignalResponse]]] = {
        c.id: defaultdict(list) for c in pending_configs
    }
    control_responses_by_key: dict[str, dict[tuple[int, float, int], list[ControlResponse]]] = {
        c.id: defaultdict(list) for c in pending_configs
    }
    path_stats_by_key: dict[str, dict[tuple[int, float, int], list[SignalPathStats]]] = {
        c.id: defaultdict(list) for c in pending_configs
    }
    controls_ct: dict[str, dict[tuple[int, float], int]] = {c.id: defaultdict(int) for c in pending_configs}

    conn = open_readonly(db_path)
    try:
        market_ids = list_market_ids(conn)
        if max_markets_per_asset is not None:
            market_ids = market_ids[:max_markets_per_asset]

        for market_id in market_ids:
            for cfg in pending_configs:
                result = results[cfg.id]
                signals = detect_signals(conn, market_id, cfg, asset)
                if not signals:
                    continue
                result.market_ids.add(market_id)
                market_row = _market_row(conn, market_id, asset)
                all_signal_ts = [s.signal_ts for s in signals]
                # Per (cfg, market) -- see _baseline_key docstring: unlike
                # entry/response, control eligibility/exclusion IS
                # config-specific (controls.py), so this cache (matching the
                # old per-config candidates_cache exactly, just re-scoped)
                # must never be shared across configs.
                candidates_cache: dict[str, list] = {}

                for signal in signals:
                    result.signals.append(signal)

                    candidates = candidates_cache.get(signal.direction)
                    if candidates is None:
                        candidates = scan_candidates(conn, market_id, cfg, signal.direction, vol_boundary)
                        candidates_cache[signal.direction] = candidates
                    controls = select_matched_controls_from_candidates(
                        candidates, market_id, asset, cfg, signal.signal_id, signal.direction,
                        source_tte_s=(signal.time_remaining_ms or 0) / 1000.0,
                        source_price=signal.target_ask or 0.0,
                        source_spread=signal.target_spread or 0.0,
                        source_vol_30s=signal.vol_30s,
                        source_tte_regime=signal.tte_regime, source_price_regime=signal.price_regime,
                        source_spread_regime=signal.spread_regime,
                        source_volatility_regime=vol_boundary.classify(signal.vol_30s)
                        if (vol_boundary and signal.vol_30s is not None) else None,
                        all_signal_ts_this_config=all_signal_ts,
                        k=controls_per_signal,
                    )
                    result.controls.extend(controls)

                    for latency_ms in latency_grid_ms:
                        for size_shares in size_grid_shares:
                            entry, responses, stats_rows = entry_cache.for_signal(
                                conn, signal, latency_ms, size_shares, market_row, fee_model,
                                HORIZONS_MS, STATS_HORIZONS_MS,
                            )
                            result.entry_marks.append(entry)
                            entry_marks_by_key[cfg.id][(latency_ms, size_shares)].append(entry)

                            if entry.status != "EXECUTED":
                                continue

                            for response in responses:
                                result.responses.append(response)
                                key = (latency_ms, size_shares, response.horizon_ms)
                                responses_by_key[cfg.id][key].append(response)
                            for stats in stats_rows:
                                result.path_stats.append(stats)
                                key = (latency_ms, size_shares, stats.stats_horizon_ms)
                                path_stats_by_key[cfg.id][key].append(stats)

                            controls_ct[cfg.id][(latency_ms, size_shares)] += len(controls)

                            for control in controls:
                                # A control is evaluated through the exact
                                # same entry/response engine as a real
                                # signal (contract #26).
                                c_entry, c_responses = entry_cache.for_control(
                                    conn, control, latency_ms, size_shares, market_row, fee_model,
                                    HORIZONS_MS, STATS_HORIZONS_MS,
                                )
                                result.control_entry_marks.append(c_entry)
                                if c_entry.status != "EXECUTED":
                                    continue
                                for c_response in c_responses:
                                    result.control_responses.append(c_response)
                                    key = (latency_ms, size_shares, c_response.horizon_ms)
                                    control_responses_by_key[cfg.id][key].append(c_response)

            # Every cache key is scoped to this market_id -- once every
            # config in this job has finished its pass over it, nothing
            # later can ever hit these entries again (see
            # EntryResponsePathCache.clear_market's docstring for why an
            # unbounded cache here caused a real overnight MemoryError /
            # BrokenProcessPool crash). Bounds memory to "one market's
            # signals+controls x this chunk's configs", not "every market
            # this job has touched so far".
            entry_cache.clear_market()
    finally:
        conn.close()

    # ---- each config's OWN summaries (contract #29-32) ----
    for cfg in pending_configs:
        result = results[cfg.id]
        for latency_ms in latency_grid_ms:
            for size_shares in size_grid_shares:
                key = (latency_ms, size_shares)
                result.config_summaries.append(compute_signal_config_summary(
                    cfg.id, asset, latency_ms, size_shares, entry_marks_by_key[cfg.id][key], result.market_ids,
                    controls_ct[cfg.id][key],
                ))

                for horizon_ms in HORIZONS_MS:
                    rkey = (latency_ms, size_shares, horizon_ms)
                    responses = responses_by_key[cfg.id][rkey]
                    mkt_of = {id(r): r.market_id for r in responses}
                    c_responses = control_responses_by_key[cfg.id][rkey]
                    c_mkt_of = {id(r): r.market_id for r in c_responses}
                    result.response_summaries.append(compute_signal_response_summary(
                        cfg.id, asset, latency_ms, size_shares, horizon_ms, responses, mkt_of,
                        c_responses, c_mkt_of,
                        bootstrap_iterations=bootstrap_iterations, bootstrap_seed=bootstrap_seed,
                    ))

                for stats_horizon_ms in STATS_HORIZONS_MS:
                    skey = (latency_ms, size_shares, stats_horizon_ms)
                    stats_rows = path_stats_by_key[cfg.id][skey]
                    for level in FIRST_PASSAGE_LEVELS:
                        result.hit_summaries.append(
                            compute_hit_summary(cfg.id, asset, latency_ms, size_shares, stats_horizon_ms,
                                                 level, "FAVORABLE", stats_rows))
                        result.hit_summaries.append(
                            compute_hit_summary(cfg.id, asset, latency_ms, size_shares, stats_horizon_ms,
                                                 level, "ADVERSE", stats_rows))
                    for plus_lvl in FIRST_PASSAGE_LEVELS:
                        for minus_lvl in FIRST_PASSAGE_LEVELS:
                            result.first_passage_summaries.append(
                                compute_first_passage_summary(cfg.id, asset, latency_ms, size_shares,
                                                                stats_horizon_ms, plus_lvl, minus_lvl, stats_rows))

    job_timings = {
        "total": time.perf_counter() - t0,
        "cache_hits": float(entry_cache.hits),
        "cache_misses": float(entry_cache.misses),
    }
    return results, job_timings


# ---------------------------------------------------------------------------
# Volatility regime boundaries (contract #28): computed ONCE per asset over
# the whole discovery dataset, frozen into experiment metadata -- never
# recomputed per signal_config.
# ---------------------------------------------------------------------------

def _compute_asset_volatility_boundaries(db_path: str, asset: str,
                                          max_markets_per_asset: int | None) -> VolatilityRegimeBoundaries | None:
    conn = open_readonly(db_path)
    try:
        market_ids = list_market_ids(conn)
        if max_markets_per_asset is not None:
            market_ids = market_ids[:max_markets_per_asset]
        buffer_cfg = ReplayBufferSizing(momentum_window_ms=1_000, volatility_lookback_ms=30_000,
                                         flow_window_ms=1_000, poly_lag_window_ms=500)
        samples: list[float] = []
        for market_id in market_ids:
            last_sampled_bucket = None
            for now_ms, state, kind in iter_replay(conn, market_id, buffer_cfg):
                if kind != "book":
                    continue
                bucket = now_ms // 5_000  # sample every 5s of simulated time -- a coarse, cheap pre-pass
                if bucket == last_sampled_bucket:
                    continue
                last_sampled_bucket = bucket
                v = realized_vol(state, now_ms, 30_000)
                if v is not None:
                    samples.append(v)
        return compute_volatility_boundaries(asset, samples)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class SignalDiscoveryRunResult:
    experiment_id: str
    signal_configs: int
    signals: int
    executable_entries: int
    controls: int
    failed_units: int = 0
    timings: dict[str, float] = field(default_factory=dict)


def run_signal_discovery_experiment(config: SignalDiscoveryExperimentConfig) -> SignalDiscoveryRunResult:
    t_start = time.perf_counter()
    validation_reports = [validate_collector_db(asset, path) for asset, path in config.data.items()]
    failed = [r for r in validation_reports if not r.ok]
    if failed:
        details = "; ".join(f"{r.asset}: {', '.join(r.errors)}" for r in failed)
        raise RuntimeError(f"data validation failed: {details}")
    before_fingerprint = {r.asset: r.fingerprint for r in validation_reports}

    signal_configs = generate_signal_discovery_configs(config.signal_grid)

    repo = DiscoveryRepository(config.results_db)
    repo.connect()
    fee_model = FeeModel()
    experiment_id = repo.create_experiment(
        config.name, config, data_sources=config.data,
        data_fingerprint=canonical_json_fingerprint(before_fingerprint),
        code_version="unknown", fee_model_version=fee_model.version,
        bootstrap_seed=config.bootstrap_seed, bootstrap_iterations=config.bootstrap_iterations,
    )

    t_vol = time.perf_counter()
    vol_boundaries: dict[str, VolatilityRegimeBoundaries | None] = {
        asset: _compute_asset_volatility_boundaries(path, asset, config.max_markets_per_asset)
        for asset, path in config.data.items()
    }
    repo.set_volatility_boundaries(experiment_id, canonical_json_fingerprint(
        {a: (b.__dict__ if b else None) for a, b in vol_boundaries.items()}
    ))
    timing_vol_boundaries = time.perf_counter() - t_vol

    try:
        for cfg in signal_configs:
            repo.save_signal_config(cfg)
        repo.commit()

        # ---- build the (baseline, asset) work queue. Checkpoints still
        # live at the (signal_config, asset) grain (task item 3): COMPLETE
        # units are skipped outright; anything else (PENDING/RUNNING/
        # FAILED, or no checkpoint row at all -- e.g. a prior crash
        # mid-unit) is discarded and redone. A baseline job is dispatched
        # if ANY of its configs still needs work; the worker internally
        # skips configs already COMPLETE for that asset (see
        # _run_baseline_asset's already_complete_ids param), so a
        # partially-done baseline never redoes already-checkpointed
        # configs. ----
        baseline_groups: dict[tuple, list[SignalDiscoveryConfig]] = defaultdict(list)
        for cfg in signal_configs:
            baseline_groups[_baseline_key(cfg)].append(cfg)

        # Split each baseline's not-yet-complete configs into bounded chunks
        # rather than dispatching all 49 flow variants as one job: a single
        # 49-wide job produces its first checkpoint only once EVERY config
        # in it is done, which (measured on real Stage B data) pushed the
        # first observable progress out past 45+ minutes with zero
        # intermediate checkpoints -- both a resume-safety regression (one
        # crash loses the whole chunk, not one config) and an observability
        # regression versus the pre-cache one-job-per-config baseline.
        # CHUNK_SIZE trades that off against cache-sharing benefit (which
        # comes from configs in the SAME job hitting the same (market,
        # direction, ts) instants -- neighboring thresholds/windows of one
        # baseline overlap the most, so even a modest chunk captures most
        # of it) -- keeps a crash's blast radius small while still sharing
        # the cache across several variants per job.
        CHUNK_SIZE = 8
        pending: list[tuple[list[SignalDiscoveryConfig], str]] = []
        already_complete: dict[str, set[str]] = defaultdict(set)
        for group in baseline_groups.values():
            for asset in config.data:
                todo: list[SignalDiscoveryConfig] = []
                for cfg in group:
                    status = repo.checkpoint_status(experiment_id, cfg.id, asset)
                    if status == "COMPLETE":
                        already_complete[cfg.id].add(asset)
                    else:
                        repo.discard_partial_asset(experiment_id, cfg.id, asset)
                        todo.append(cfg)
                for i in range(0, len(todo), CHUNK_SIZE):
                    pending.append((todo[i:i + CHUNK_SIZE], asset))
        repo.commit()

        total_units = len(signal_configs) * len(config.data)
        already_complete_units = sum(len(v) for v in already_complete.values())
        signals_total = 0
        executable_entries_total = 0
        controls_total = 0
        failed_units = 0
        agg_timings: dict[str, float] = defaultdict(float)
        completed = 0
        assets_done_this_config: dict[str, set[str]] = defaultdict(set, {k: set(v) for k, v in already_complete.items()})

        if pending:
            max_workers = min(len(pending), MAX_WORKERS)
            log.info(
                "running %d (baseline, asset) jobs covering %d/%d (signal_config, asset) units "
                "(%d already complete) across %d worker processes",
                len(pending), total_units - already_complete_units, total_units, already_complete_units,
                max_workers,
            )
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(
                        _run_baseline_asset, chunk, set(), asset, config.data[asset],
                        config.max_markets_per_asset, config.latency_grid_ms, config.size_grid_shares,
                        config.controls_per_signal, config.bootstrap_iterations, config.bootstrap_seed,
                        vol_boundaries.get(asset),
                    ): (chunk, asset)
                    for chunk, asset in pending
                }
                for future in as_completed(futures):
                    chunk, asset = futures[future]
                    completed += 1
                    pending_ids = [c.id for c in chunk]
                    try:
                        results_by_cfg, job_timings = future.result()
                    except Exception as exc:  # noqa: BLE001 -- one job's failure must not abort the sweep
                        log.exception("baseline job (%d configs) asset %s FAILED", len(pending_ids), asset)
                        for cfg_id in pending_ids:
                            repo.mark_checkpoint_failed(experiment_id, cfg_id, asset, str(exc))
                        failed_units += len(pending_ids)
                        continue

                    job_signals = 0
                    job_executable = 0
                    job_controls = 0
                    for cfg_id in pending_ids:
                        result = results_by_cfg[cfg_id]
                        for s in result.signals:
                            repo.save_signal(experiment_id, s)
                        for m in result.entry_marks:
                            repo.save_entry_mark(m)
                        for r in result.responses:
                            repo.save_signal_response(r)
                        for p in result.path_stats:
                            repo.save_path_stats(p)
                        for c in result.controls:
                            repo.save_control(c)
                        for cem in result.control_entry_marks:
                            repo.save_control_entry_mark(cem)
                        for cr in result.control_responses:
                            repo.save_control_response(cr)
                        for cs in result.config_summaries:
                            repo.save_config_summary(experiment_id, cs)
                        for rs in result.response_summaries:
                            repo.save_response_summary(experiment_id, rs)
                        for hs in result.hit_summaries:
                            repo.save_hit_summary(experiment_id, hs)
                        for fps in result.first_passage_summaries:
                            repo.save_first_passage_summary(experiment_id, fps)

                        repo.mark_checkpoint_complete(experiment_id, cfg_id, asset)
                        assets_done_this_config[cfg_id].add(asset)

                        job_signals += len(result.signals)
                        job_executable += sum(1 for m in result.entry_marks if m.status == "EXECUTED")
                        job_controls += len(result.controls)

                    signals_total += job_signals
                    executable_entries_total += job_executable
                    controls_total += job_controls
                    for k, v in job_timings.items():
                        agg_timings[k] += v

                    log.info(
                        "checkpointed baseline job asset=%s [%d/%d jobs, %d configs]: %d signals, "
                        "%d executable entries, %d controls, cache hits=%d misses=%d",
                        asset, completed, len(pending), len(pending_ids), job_signals, job_executable,
                        job_controls, int(job_timings.get("cache_hits", 0)), int(job_timings.get("cache_misses", 0)),
                    )

        # Pool every fully-complete config's ALL summary -- covers both
        # configs whose assets were ALL already COMPLETE before this run
        # started, and ones this run just finished. Idempotent (INSERT OR
        # REPLACE), so re-pooling an already-pooled config is harmless --
        # skipped below via a cheap existence check, which is what makes
        # this loop itself resumable (kill the process mid-pool, rerun,
        # only the not-yet-pooled configs get redone). Parallelized across
        # MAX_WORKERS processes (same single-writer discipline as the main
        # dispatch loop: workers only compute, the parent commits, one
        # config at a time -- so progress is visible in the DB as it
        # happens instead of sitting invisible in one giant transaction
        # until every one of 1176 configs is done).
        t_pool_all = time.perf_counter()
        to_pool = [cfg for cfg in signal_configs if assets_done_this_config[cfg.id] >= set(config.data)]
        already_pooled = {
            r[0] for r in repo._require_conn().execute(
                "SELECT DISTINCT signal_config_id FROM signal_config_summary WHERE experiment_id=? AND asset=?",
                (experiment_id, ALL_ASSET),
            )
        }
        pool_pending = [cfg for cfg in to_pool if cfg.id not in already_pooled]
        log.info(
            "pooling ALL summaries for %d/%d configs (%d already pooled) across %d worker processes",
            len(pool_pending), len(to_pool), len(to_pool) - len(pool_pending), MAX_WORKERS,
        )
        if pool_pending:
            max_pool_workers = min(len(pool_pending), MAX_WORKERS)
            pooled_count = 0
            with ProcessPoolExecutor(max_workers=max_pool_workers) as pool:
                futures = {
                    pool.submit(_pool_config_worker, cfg.id, experiment_id, config, config.results_db): cfg.id
                    for cfg in pool_pending
                }
                for future in as_completed(futures):
                    cfg_id, cs, rs, hs, fps = future.result()
                    for x in cs:
                        repo.save_config_summary(experiment_id, x)
                    for x in rs:
                        repo.save_response_summary(experiment_id, x)
                    for x in hs:
                        repo.save_hit_summary(experiment_id, x)
                    for x in fps:
                        repo.save_first_passage_summary(experiment_id, x)
                    repo.commit()
                    pooled_count += 1
                    if pooled_count % 20 == 0 or pooled_count == len(pool_pending):
                        log.info("pooled %d/%d configs", pooled_count, len(pool_pending))
        agg_timings["pooling"] += time.perf_counter() - t_pool_all

        # ---- plateau/neighbor pass (contract #36-37): needs every config's
        # own ALL summary to already exist. ----
        t_plateau = time.perf_counter()
        _run_plateau_pass(repo, experiment_id, signal_configs, config)
        agg_timings["plateau"] += time.perf_counter() - t_plateau

        after_fingerprint = {asset: fingerprint_db(path) for asset, path in config.data.items()}
        if before_fingerprint != after_fingerprint:
            raise RuntimeError("collector DB fingerprint changed during discovery run")

        repo.finish_experiment(experiment_id, "FAILED" if failed_units else "SUCCESS",
                                {"failed_units": failed_units} if failed_units else None)
    except Exception as exc:
        repo.finish_experiment(experiment_id, "FAILED", {"error": str(exc)})
        repo.close()
        raise

    repo.close()
    agg_timings["volatility_boundaries"] = timing_vol_boundaries
    agg_timings["total_wall"] = time.perf_counter() - t_start
    return SignalDiscoveryRunResult(
        experiment_id=experiment_id, signal_configs=len(signal_configs), signals=signals_total,
        executable_entries=executable_entries_total, controls=controls_total, failed_units=failed_units,
        timings=dict(agg_timings),
    )


def _pool_config_summaries_compute(
    conn, experiment_id: str, cfg_id: str, config: SignalDiscoveryExperimentConfig,
) -> tuple[list[SignalConfigSummary], list[SignalResponseSummary], list[SignalHitSummary],
           list[SignalFirstPassageSummary]]:
    """Pure-compute half of pooling one config's asset=NULL "ALL" summary
    rows (pooled across every asset, per contract: "ALL пересчитывать из
    underlying observations", never by averaging per-asset metrics) --
    reads via `conn` (a read-only connection is fine, including one opened
    fresh by a worker process) and RETURNS the computed rows instead of
    writing them, so this can run in parallel across configs (contract #39
    still forbids concurrent writers -- only the parent writes, via
    _pool_config_worker's caller)."""
    market_rows = conn.execute(
        "SELECT DISTINCT market_id FROM signals WHERE experiment_id=? AND signal_config_id=?",
        (experiment_id, cfg_id),
    ).fetchall()
    all_market_ids = {r["market_id"] for r in market_rows}
    control_count = conn.execute(
        "SELECT COUNT(*) FROM controls WHERE signal_config_id=?", (cfg_id,)
    ).fetchone()[0]

    config_summaries: list[SignalConfigSummary] = []
    response_summaries: list[SignalResponseSummary] = []
    hit_summaries: list[SignalHitSummary] = []
    first_passage_summaries: list[SignalFirstPassageSummary] = []

    for latency_ms in config.latency_grid_ms:
        for size_shares in config.size_grid_shares:
            entry_marks = [entry_mark_from_row(r) for r in conn.execute(
                """SELECT em.* FROM entry_marks em JOIN signals s ON s.signal_id=em.signal_id
                   WHERE s.experiment_id=? AND s.signal_config_id=? AND em.latency_ms=? AND em.size_shares=?""",
                (experiment_id, cfg_id, latency_ms, size_shares),
            )]
            config_summaries.append(compute_signal_config_summary(
                cfg_id, ALL_ASSET, latency_ms, size_shares, entry_marks, all_market_ids, control_count,
            ))

            for horizon_ms in HORIZONS_MS:
                responses = [signal_response_from_row(r) for r in conn.execute(
                    """SELECT * FROM signal_response WHERE signal_config_id=? AND latency_ms=?
                       AND size_shares=? AND horizon_ms=?""",
                    (cfg_id, latency_ms, size_shares, horizon_ms),
                )]
                control_responses = [control_response_from_row(r) for r in conn.execute(
                    """SELECT * FROM control_response WHERE signal_config_id=? AND latency_ms=?
                       AND size_shares=? AND horizon_ms=?""",
                    (cfg_id, latency_ms, size_shares, horizon_ms),
                )]
                mkt_of = {id(r): r.market_id for r in responses}
                c_mkt_of = {id(r): r.market_id for r in control_responses}
                response_summaries.append(compute_signal_response_summary(
                    cfg_id, ALL_ASSET, latency_ms, size_shares, horizon_ms, responses, mkt_of,
                    control_responses, c_mkt_of,
                    bootstrap_iterations=config.bootstrap_iterations, bootstrap_seed=config.bootstrap_seed,
                ))

            for stats_horizon_ms in STATS_HORIZONS_MS:
                stats_rows = [path_stats_from_row(r) for r in conn.execute(
                    """SELECT sps.* FROM signal_path_stats sps
                       JOIN entry_marks em ON em.entry_mark_id=sps.entry_mark_id
                       JOIN signals s ON s.signal_id=em.signal_id
                       WHERE s.experiment_id=? AND s.signal_config_id=? AND em.latency_ms=?
                         AND em.size_shares=? AND sps.stats_horizon_ms=?""",
                    (experiment_id, cfg_id, latency_ms, size_shares, stats_horizon_ms),
                )]
                for level in FIRST_PASSAGE_LEVELS:
                    hit_summaries.append(compute_hit_summary(
                        cfg_id, ALL_ASSET, latency_ms, size_shares, stats_horizon_ms, level, "FAVORABLE", stats_rows))
                    hit_summaries.append(compute_hit_summary(
                        cfg_id, ALL_ASSET, latency_ms, size_shares, stats_horizon_ms, level, "ADVERSE", stats_rows))
                for plus_lvl in FIRST_PASSAGE_LEVELS:
                    for minus_lvl in FIRST_PASSAGE_LEVELS:
                        first_passage_summaries.append(compute_first_passage_summary(
                            cfg_id, ALL_ASSET, latency_ms, size_shares, stats_horizon_ms, plus_lvl, minus_lvl,
                            stats_rows,
                        ))

    return config_summaries, response_summaries, hit_summaries, first_passage_summaries


def _pool_config_worker(
    cfg_id: str, experiment_id: str, config: SignalDiscoveryExperimentConfig, results_db: str,
) -> tuple[str, list[SignalConfigSummary], list[SignalResponseSummary], list[SignalHitSummary],
           list[SignalFirstPassageSummary]]:
    """ProcessPoolExecutor entry point: opens its OWN read-only connection
    (safe to run concurrently with the parent's writer connection -- the
    results DB is in WAL mode, and pooling only READS the raw signal/
    response/control tables, which are already fully committed by the time
    pooling starts; it never touches them). Returns cfg_id alongside the
    computed rows so the parent can attribute results without relying on
    submission order."""
    conn = open_readonly(results_db)
    try:
        cs, rs, hs, fps = _pool_config_summaries_compute(conn, experiment_id, cfg_id, config)
    finally:
        conn.close()
    return cfg_id, cs, rs, hs, fps


def _run_plateau_pass(repo: DiscoveryRepository, experiment_id: str, signal_configs: list[SignalDiscoveryConfig],
                       config: SignalDiscoveryExperimentConfig) -> None:
    conn = repo._require_conn()
    rows = conn.execute(
        """SELECT signal_config_id, asset, latency_ms, size_shares, uplift_mean_response
           FROM signal_response_summary WHERE experiment_id=? AND horizon_ms=?""",
        (experiment_id, config.ranking_horizon_ms),
    ).fetchall()
    uplift_by_key: dict[tuple[str, Any, int, float], float | None] = {
        (r["signal_config_id"], r["asset"], r["latency_ms"], r["size_shares"]): r["uplift_mean_response"]
        for r in rows
    }
    signal_count_rows = conn.execute(
        """SELECT signal_config_id, asset, latency_ms, size_shares, signal_count
           FROM signal_config_summary WHERE experiment_id=?""",
        (experiment_id,),
    ).fetchall()
    signal_count_by_key = {
        (r["signal_config_id"], r["asset"], r["latency_ms"], r["size_shares"]): r["signal_count"]
        for r in signal_count_rows
    }

    for latency_ms in config.latency_grid_ms:
        for size_shares in config.size_grid_shares:
            for target_cfg in signal_configs:
                key = (target_cfg.id, ALL_ASSET, latency_ms, size_shares)
                central_uplift = uplift_by_key.get(key)
                signal_count = signal_count_by_key.get(key, 0)
                neighbors = find_neighbors(target_cfg, signal_configs, config.signal_grid)
                neighbor_uplifts = [
                    uplift_by_key.get((n.id, ALL_ASSET, latency_ms, size_shares)) for n in neighbors
                ]
                metrics = compute_plateau_metrics(central_uplift, signal_count, neighbor_uplifts)
                conn.execute(
                    """UPDATE signal_config_summary
                       SET plateau_score=?, neighbor_count=?, neighbor_positive_ratio=?,
                           neighbor_mean_uplift=?, neighbor_std_uplift=?
                       WHERE experiment_id=? AND signal_config_id=? AND asset = ?
                         AND latency_ms=? AND size_shares=?""",
                    (metrics.plateau_score, metrics.neighbor_count, metrics.neighbor_positive_ratio,
                     metrics.neighbor_mean_uplift, metrics.neighbor_std_uplift,
                     experiment_id, target_cfg.id, ALL_ASSET, latency_ms, size_shares),
                )
    conn.commit()


def canonical_json_fingerprint(value: Any) -> str:
    return canonical_json(value)
