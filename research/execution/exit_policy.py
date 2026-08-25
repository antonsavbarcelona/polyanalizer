"""Pure exit-policy evaluator over an already-fetched SignalEpisode path."""
from __future__ import annotations

from poly_analyzer.discovery import extract_levels

from research.execution.episode import SignalEpisode, first_row_at_or_after
from research.execution.vwap import full_vwap_fill
from research.fees import FeeModel
from research.types import ExitConfig, TradeResult, deterministic_id

PRICE_EPSILON = 1e-12


def _fill_at(row: dict, prefix: str, size_shares: float) -> tuple[float, float] | None:
    bids = sorted(extract_levels(row, prefix, "bid"), key=lambda level: level[0], reverse=True)
    return full_vwap_fill(bids, size_shares)


def _best_bid(row: dict, prefix: str) -> float | None:
    bids = sorted(extract_levels(row, prefix, "bid"), key=lambda level: level[0], reverse=True)
    if bids:
        return bids[0][0]
    return row.get(f"{prefix}_best_bid")


def _resolve_exit_at_or_after(
    path_rows: list[dict],
    path_ts: list[int],
    target_ts: int,
    prefix: str,
    size_shares: float,
) -> tuple[int, float, float | None] | None:
    idx = first_row_at_or_after(path_ts, target_ts)
    if idx is None:
        return None
    for i in range(idx, len(path_rows)):
        fill = _fill_at(path_rows[i], prefix, size_shares)
        if fill is not None:
            return path_rows[i]["ts"], fill[0], _best_bid(path_rows[i], prefix)
    return None


def _last_valid_fill(path_rows: list[dict], prefix: str, size_shares: float) -> tuple[int, float, float | None] | None:
    for row in reversed(path_rows):
        fill = _fill_at(row, prefix, size_shares)
        if fill is not None:
            return row["ts"], fill[0], _best_bid(row, prefix)
    return None


def _resolve_hold_to_resolution(episode: SignalEpisode, direction: str) -> tuple[int, float] | None:
    outcome = (
        episode.market_row.get("official_outcome")
        or episode.market_row.get("derived_outcome")
        or episode.market_row.get("resolution")
    )
    if outcome not in {"UP", "DOWN"}:
        return None
    exit_ts = episode.market_row.get("resolution_ts") or episode.market_row.get("end_ts")
    if exit_ts is None and episode.path_ts:
        exit_ts = episode.path_ts[-1]
    if exit_ts is None:
        return None
    return exit_ts, 1.0 if outcome == direction else 0.0


def _resolve_stop(
    episode: SignalEpisode,
    trigger_ts: int,
    trigger_sell_vwap: float,
    exit_cfg: ExitConfig,
    prefix: str,
    size_shares: float,
) -> tuple[int, float, str, bool, float | None]:
    if exit_cfg.stop_exit_latency_ms <= 0:
        return trigger_ts, trigger_sell_vwap, "SL", True, None

    resolved = _resolve_exit_at_or_after(
        episode.path_rows,
        episode.path_ts,
        trigger_ts + exit_cfg.stop_exit_latency_ms,
        prefix,
        size_shares,
    )
    if resolved is not None:
        return resolved[0], resolved[1], "SL", True, resolved[2]

    fallback = _last_valid_fill(episode.path_rows, prefix, size_shares)
    if fallback is None:
        raise ValueError("no executable exit after stop trigger")
    return fallback[0], fallback[1], "MARKET_END", False, fallback[2]


def evaluate_exit(
    episode: SignalEpisode,
    exit_cfg: ExitConfig,
    fee_model: FeeModel,
    size_shares: float,
    asset: str | None = None,
) -> TradeResult | None:
    """Evaluate one ExitConfig without querying the DB.

    Taker exits require enough bid-side depth for the full requested size.
    Timeout has priority at and after its boundary; a TP/SL observed only on
    the first row after the timeout is not allowed to beat the forced exit.
    """
    entry_vwap = episode.entry_mark.entry_vwap
    entry_ts = episode.entry_mark.actual_ts
    if entry_vwap is None or entry_ts is None:
        return None

    direction = episode.signal.direction
    prefix = "up" if direction == "UP" else "down"
    tp_target = exit_cfg.resolve_take_profit(entry_vwap)
    sl_target = exit_cfg.resolve_stop_loss(entry_vwap)

    mfe: float | None = None
    mae: float | None = None
    exit_ts: int | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    exit_execution: str | None = None
    exit_best_bid: float | None = None
    tp_hit = False
    sl_hit = False
    ambiguous_exit = False

    if exit_cfg.hold_to_resolution:
        resolution = _resolve_hold_to_resolution(episode, direction)
        if resolution is None:
            return None
        exit_ts, exit_price = resolution
        exit_reason, exit_execution = "RESOLUTION", "RESOLUTION"

    for row in [] if exit_reason is not None else episode.path_rows:
        ts = row["ts"]
        elapsed = ts - entry_ts
        fill = _fill_at(row, prefix, size_shares)
        sell_vwap = fill[0] if fill is not None else None
        best_bid = _best_bid(row, prefix)

        if sell_vwap is not None:
            ret = sell_vwap - entry_vwap
            mfe = ret if mfe is None else max(mfe, ret)
            mae = ret if mae is None else min(mae, ret)

        if exit_cfg.max_holding_ms is not None and elapsed >= exit_cfg.max_holding_ms:
            resolved = _resolve_exit_at_or_after(
                episode.path_rows,
                episode.path_ts,
                entry_ts + exit_cfg.max_holding_ms,
                prefix,
                size_shares,
            )
            if resolved is None:
                fallback = _last_valid_fill(episode.path_rows, prefix, size_shares)
                if fallback is None:
                    return None
                exit_ts, exit_price, exit_best_bid = fallback
                exit_reason = "MARKET_END"
            else:
                exit_ts, exit_price, exit_best_bid = resolved
                exit_reason = "TIMEOUT"
            exit_execution = exit_cfg.timeout_execution
            break

        tp_trigger = tp_target is not None and best_bid is not None and best_bid + PRICE_EPSILON >= tp_target
        sl_trigger = sl_target is not None and sell_vwap is not None and sell_vwap <= sl_target + PRICE_EPSILON

        if tp_trigger and sl_trigger:
            ambiguous_exit = True
            sl_hit = True
            try:
                exit_ts, exit_price, exit_reason, sl_hit, exit_best_bid = _resolve_stop(
                    episode, ts, sell_vwap, exit_cfg, prefix, size_shares
                )
            except ValueError:
                return None
            exit_execution = exit_cfg.stop_execution
            break

        if tp_trigger:
            exit_reason, exit_price, exit_execution = "TP", tp_target, exit_cfg.tp_execution
            tp_hit = True
            exit_ts = ts
            exit_best_bid = best_bid
            break

        if sl_trigger:
            try:
                exit_ts, exit_price, exit_reason, sl_hit, exit_best_bid = _resolve_stop(
                    episode, ts, sell_vwap, exit_cfg, prefix, size_shares
                )
            except ValueError:
                return None
            exit_execution = exit_cfg.stop_execution
            break
    else:
        fallback = _last_valid_fill(episode.path_rows, prefix, size_shares)
        if fallback is None:
            return None
        exit_ts, exit_price, exit_best_bid = fallback
        exit_reason, exit_execution = "MARKET_END", "TAKER"

    gross_pnl_per_share = exit_price - entry_vwap
    entry_fee_per_share = fee_model.taker_fee_per_share(entry_vwap, episode.market_row)
    exit_fee_per_share = fee_model.exit_fee(exit_price, 1.0, exit_execution, episode.market_row)
    fees_per_share = entry_fee_per_share + exit_fee_per_share
    net_pnl_per_share = gross_pnl_per_share - fees_per_share
    pnl_total = net_pnl_per_share * size_shares
    holding_ms = exit_ts - entry_ts

    signal = episode.signal
    signal_id = deterministic_id(
        "signal",
        {
            "config_id": signal.config_id,
            "market_id": signal.market_id,
            "direction": signal.direction,
            "signal_ts": signal.signal_ts,
        },
    )
    entry_slippage = None
    if episode.entry_mark.best_ask is not None:
        entry_slippage = entry_vwap - episode.entry_mark.best_ask
    exit_slippage = None
    if exit_execution != "MAKER" and exit_best_bid is not None:
        exit_slippage = exit_best_bid - exit_price

    result = TradeResult(
        strategy_id="",
        signal_id=signal_id,
        asset=asset if asset is not None else episode.market_row.get("asset"),
        market_id=signal.market_id,
        direction=signal.direction,
        signal_ts=signal.signal_ts,
        entry_requested_ts=episode.entry_mark.target_ts,
        entry_actual_ts=entry_ts,
        entry_vwap=entry_vwap,
        exit_ts=exit_ts,
        exit_price=exit_price,
        exit_reason=exit_reason,
        holding_ms=holding_ms,
        gross_pnl_per_share=gross_pnl_per_share,
        fees_per_share=fees_per_share,
        net_pnl_per_share=net_pnl_per_share,
        pnl_total=pnl_total,
        mfe=mfe,
        mae=mae,
        tp_hit=tp_hit,
        sl_hit=sl_hit,
        ambiguous_exit=ambiguous_exit,
        entry_fee_per_share=entry_fee_per_share,
        exit_fee_per_share=exit_fee_per_share,
        entry_slippage=entry_slippage,
        exit_slippage=exit_slippage,
    )
    result.trade_result_id = deterministic_id(
        "trade",
        {
            "strategy_id": result.strategy_id,
            "signal_id": result.signal_id,
            "entry_actual_ts": result.entry_actual_ts,
            "exit_ts": result.exit_ts,
            "exit_reason": result.exit_reason,
        },
    )
    return result
