"""Collector entry point: connects to Binance/Polymarket/Chainlink, records
raw data, tracks market rollover + settlement. Deliberately does NOT run
any live signal detection, fair-value computation, or console UI -- this
process's only job is to save facts for later offline research (see
db.py's 2026-08-22 raw-data redesign docstring). Every question about
signals, entries, exits, or thresholds is answered later by replaying the
recorded raw tables (research/discovery.py), never decided live.

2026-08-25: stripped down from the earlier version, which also ran a live
(non-authoritative) SignalEngine + fair-value calc + a Rich console UI --
none of that is read by anything downstream; only the raw event writes
(binance_trades/binance_book/polymarket_book/chainlink_observations,
written directly by the feed classes themselves, independent of this
App/on_update) and market/settlement metadata matter. A periodic log line
replaces the console UI for unattended multi-day runs (a Rich Live display
is pointless once stdout is redirected to a log file -- see the project's
own lesson about redirected-output buffering on Windows).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time

from .binance_feed import BinanceFeed
from .chainlink_feed import ChainlinkFeed
from .config import ASSETS, Config, build_config
from .db import Recorder
from .features import build_binance_features_row
from .market_discovery import find_current_market
from .polymarket_feed import PolymarketFeed
from .settlement import track_settlement
from .state import MarketState

log = logging.getLogger(__name__)

STATUS_LOG_INTERVAL_S = 30.0


def now_ms() -> int:
    return int(time.time() * 1000)


def _log_task_exception(task: asyncio.Task) -> None:
    """One-shot background tasks (settlement tracking, one per market
    rollover) are deliberately fire-and-forget via create_task() -- they
    can't be folded into App.run()'s single gather() call, which is fixed
    upfront and runs forever. Without this callback, a failure inside one
    is a silently orphaned Task: nothing awaits it, so the exception
    doesn't surface until the Task object is garbage collected (which for
    a short-lived process can mean never) -- the same failure mode fixed
    for the Recorder's writer task, just structurally unfixable the same
    way here since these tasks are created continuously, not once."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.error("background task failed", exc_info=exc)


class App:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.state = MarketState()
        self.recorder = Recorder(cfg.recorder)
        is_debug = cfg.debug_mode
        self.binance = BinanceFeed(cfg.market, self.state, recorder=self.recorder,
                                    on_update=self.on_update, is_debug=is_debug)
        self.poly = PolymarketFeed(cfg.market, self.state, recording=cfg.recording, recorder=self.recorder,
                                    on_update=self.on_update, is_debug=is_debug)
        self.chainlink = ChainlinkFeed(cfg.market, self.state, recorder=self.recorder,
                                        on_update=self.on_update, is_debug=is_debug)

        self.last_binance_features_write_ms = 0
        self.current_market_id: str | None = None
        self._current_market_row: dict | None = None
        self._reference_price_stamped = False

    def on_update(self, source: str) -> None:
        # Raw event rows (binance_trades/binance_book/polymarket_book/
        # chainlink_observations) are already written by the feed classes
        # themselves, independent of this callback. All that's left here is
        # the optional convenience feature snapshot and stamping the
        # Chainlink reference price onto the current market row once known.
        now = now_ms()
        if self.state.market_id is None:
            return
        self._maybe_write_binance_features(now, source)
        self._maybe_update_reference_price()

    def _maybe_write_binance_features(self, now: int, source: str) -> None:
        if now - self.last_binance_features_write_ms < self.cfg.recording.binance_features_interval_ms:
            return
        self.last_binance_features_write_ms = now
        row = build_binance_features_row(self.state, now, self.state.market_id, self.cfg.debug_mode)
        self.recorder.enqueue("binance_features", row)

    def _maybe_update_reference_price(self) -> None:
        if self._reference_price_stamped or self._current_market_row is None:
            return
        if self.state.chainlink_reference_price is None:
            return
        self._current_market_row["reference_price"] = self.state.chainlink_reference_price
        self.recorder.enqueue("markets", dict(self._current_market_row))
        self._reference_price_stamped = True

    async def market_rollover_loop(self) -> None:
        while True:
            await self._maybe_rollover()
            await asyncio.sleep(5)

    async def _maybe_rollover(self, discover=find_current_market) -> None:
        # Only re-search once we actually need a new market. Re-deriving
        # "current" from the Gamma API on every tick is not just wasted
        # work -- a single transient/empty response for the exact-current
        # slug makes find_current_market() fall through to the NEXT window
        # (which always exists and is always "not yet ended"), causing a
        # spurious rollover-and-back-again flip-flop that was observed live.
        # Trusting our own known end_ts is authoritative and avoids
        # re-asking a question we already know the answer to.
        need_market = self.current_market_id is None or now_ms() >= (self.state.market_end_ts or 0)
        if not need_market:
            return
        try:
            info = await asyncio.to_thread(discover, self.cfg.market)
        except Exception:
            log.exception("market discovery failed")
            return
        if info is None or info.market_id == self.current_market_id:
            return

        ending_market_id = self.current_market_id
        ending_market_end_ts = self.state.market_end_ts
        ending_reference_price = self.state.chainlink_reference_price
        if ending_market_id is not None:
            settlement_task = asyncio.create_task(track_settlement(
                self.cfg, ending_market_id, ending_market_end_ts, ending_reference_price, self.recorder,
            ))
            settlement_task.add_done_callback(_log_task_exception)

        log.info("[%s] market rollover -> %s (ends in %.0fs)", self.cfg.recorder.asset, info.slug,
                  (info.end_ts_ms - now_ms()) / 1000.0)
        self.current_market_id = info.market_id
        self.state.reset_for_new_market(
            info.market_id, info.condition_id, info.slug,
            info.up_token_id, info.down_token_id,
            info.start_ts_ms, info.end_ts_ms, info.tick_size,
        )
        self.poly.set_tokens(info.up_token_id, info.down_token_id)
        self._reference_price_stamped = False
        self._current_market_row = {
            "market_id": info.market_id, "slug": info.slug,
            "condition_id": info.condition_id,
            "up_token_id": info.up_token_id, "down_token_id": info.down_token_id,
            "start_ts": info.start_ts_ms, "end_ts": info.end_ts_ms,
            "tick_size": info.tick_size, "reference_price": None,
            "resolution_source": info.resolution_source,
            "twap_window": self.cfg.market.chainlink_twap_window_s,
            "maker_base_fee": info.maker_base_fee, "taker_base_fee": info.taker_base_fee,
            "fee_rate": info.fee_rate, "fee_exponent": info.fee_exponent,
            "created_at": now_ms(),
        }
        self.recorder.enqueue("markets", dict(self._current_market_row))

    async def status_log_loop(self) -> None:
        """Lightweight periodic heartbeat for an unattended multi-day run --
        replaces the old Rich console UI, which is pointless once stdout is
        redirected to a log file."""
        while True:
            await asyncio.sleep(STATUS_LOG_INTERVAL_S)
            remaining_s = None
            if self.state.market_end_ts is not None:
                remaining_s = (self.state.market_end_ts - now_ms()) / 1000.0
            log.info(
                "[%s] status: binance=%s polymarket=%s chainlink=%s market=%s remaining=%s",
                self.cfg.recorder.asset, self.binance.connected, self.poly.connected, self.chainlink.connected,
                self.current_market_id, f"{remaining_s:.0f}s" if remaining_s is not None else "?",
            )

    async def run(self) -> None:
        self.recorder.connect()
        self.recorder.start()
        try:
            await asyncio.gather(
                self.binance.run(),
                self.poly.run(),
                self.chainlink.run(),
                self.market_rollover_loop(),
                self.status_log_loop(),
                self.recorder.wait(),
            )
        finally:
            await self.recorder.flush()
            self.recorder.close()


RESTART_BACKOFF_S = 5.0


async def _run_resilient(asset: str, debug_mode: bool) -> None:
    """Runs one asset's collector forever, restarting it on any crash
    instead of propagating -- used by run_all() so one asset's WebSocket/DB
    hiccup can never take the other two down with it (plain asyncio.gather
    cancels every sibling task the moment ANY one of them raises)."""
    while True:
        try:
            app = App(build_config(asset, debug_mode=debug_mode))
            await app.run()
        except Exception:
            log.exception("[%s] collector crashed, restarting in %.0fs", asset.upper(), RESTART_BACKOFF_S)
            await asyncio.sleep(RESTART_BACKOFF_S)


async def run_all(debug_mode: bool) -> None:
    """All three collectors in ONE process -- fine now that Postgres (not
    per-asset SQLite files) is the shared write target, and each asset is
    fault-isolated via _run_resilient. Trades cross-asset isolation (a
    Railway/Docker restart of this one service restarts all three) for
    much simpler deployment (one service instead of three, one place to
    look at logs) -- worth it at this project's scale."""
    await asyncio.gather(*(_run_resilient(asset, debug_mode) for asset in sorted(ASSETS)))


def main() -> None:
    # ASSET env var (POLY_ANALYZER_ASSET) picks the asset when --asset isn't
    # passed explicitly -- lets three Railway services share the exact same
    # Start Command (python -m poly_analyzer.main) and one railway.toml,
    # differentiated purely by each service's env vars (POLY_ANALYZER_ASSET
    # = btc/eth/sol) instead of a per-service Start Command override.
    env_asset = os.environ.get("POLY_ANALYZER_ASSET", "all").lower()

    parser = argparse.ArgumentParser(description="15m Polymarket Up/Down lead-lag discovery collector")
    parser.add_argument("--asset", choices=sorted(ASSETS) + ["all"], default=env_asset,
                         help="which underlying to track (overrides POLY_ANALYZER_ASSET env var if given). "
                              "'all' runs btc+eth+sol together in this one process/worker (each still gets "
                              "its own WS subscriptions + MarketState); needs POLY_ANALYZER_DSN set (see "
                              ".env.example) so all three can safely share one database.")
    parser.add_argument("--debug", action="store_true", help="mark all rows is_debug=1 (first 24h burn-in)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        if args.asset == "all":
            asyncio.run(run_all(debug_mode=args.debug))
        else:
            cfg = build_config(args.asset, debug_mode=args.debug)
            app = App(cfg)
            asyncio.run(app.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
