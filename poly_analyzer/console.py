from __future__ import annotations

import time

from rich.console import Console
from rich.table import Table

from .features import Features
from .state import MarketState


def render(state: MarketState, features: Features | None, fair_up: float | None,
           signals_total: int, connected: dict[str, bool],
           last_signal_marks: dict[int, dict] | None = None) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="left")
    table.add_column(justify="right")

    now_ms = int(time.time() * 1000)
    remaining_s = (state.market_end_ts - now_ms) / 1000.0 if state.market_end_ts else None

    table.add_row("Market", state.slug or "(discovering...)")
    table.add_row("Feeds", " ".join(f"{k}={'up' if v else 'DOWN'}" for k, v in connected.items()))
    table.add_row("BTC Binance", f"{state.btc_mid:,.2f}" if state.btc_mid else "-")
    table.add_row("Chainlink TWAP", f"{state.chainlink_twap:,.2f}" if state.chainlink_twap else "-")
    table.add_row("Reference (R0)", f"{state.chainlink_reference_price:,.2f}" if state.chainlink_reference_price else "-")
    table.add_row("Time left", f"{remaining_s:,.1f}s" if remaining_s is not None else "-")

    if features:
        table.add_row("z_1s", f"{features.z_1s:+.2f}" if features.z_1s is not None else "-")
        table.add_row("momentum 250ms", f"{features.momentum_250ms:+.4%}" if features.momentum_250ms is not None else "-")
        table.add_row("flow 1s", f"{features.flow_1s:+.2f}" if features.flow_1s is not None else "-")
        table.add_row("book imbalance", f"{features.book_imbalance:+.2f}" if features.book_imbalance is not None else "-")

    table.add_row("fair UP", f"{fair_up:.1%}" if fair_up is not None else "-")
    table.add_row("UP bid/ask", f"{state.up.bid}/{state.up.ask}")
    table.add_row("DOWN bid/ask", f"{state.down.bid}/{state.down.ask}")
    table.add_row("Signals total", str(signals_total))

    if last_signal_marks:
        marks = " | ".join(
            f"{lat}ms: ask={m.get('ask')} vwap={m.get('vwap')}"
            for lat, m in sorted(last_signal_marks.items())
        )
        table.add_row("Last signal marks", marks)

    return table


def make_console() -> Console:
    return Console()
