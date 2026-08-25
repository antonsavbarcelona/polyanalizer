from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from .timeseries import TimeSeries

PRICE_HISTORY_MAX_AGE_MS = 130_000  # 130s: covers 120s sigma lookback + slack
TRADE_BUFFER_MAX_AGE_MS = 6_000
ASK_HISTORY_MAX_AGE_MS = 2_000
SECOND_RETURNS_MAX_LEN = 130


@dataclass
class TokenBook:
    bid: float | None = None
    ask: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    last_update_ts: int | None = None
    last_trade_price: float | None = None
    last_trade_size: float | None = None
    ask_history: TimeSeries = field(default_factory=lambda: TimeSeries(ASK_HISTORY_MAX_AGE_MS))

    @property
    def spread(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid

    @property
    def valid(self) -> bool:
        if self.bid is None or self.ask is None:
            return False
        return self.bid <= self.ask


class MarketState:
    """Single shared mutable object updated by connectors, read by feature/signal engines.

    Buffer sizes default to the live collector's fixed windows but are
    overridable so discovery.py can replay the exact same update methods
    (guaranteeing byte-identical offline recomputation, see
    tests/test_offline_recompute.py) with buffers wide enough for whatever
    momentum/volatility/flow windows a given signal config asks for.
    """

    def __init__(self, price_history_max_age_ms: int = PRICE_HISTORY_MAX_AGE_MS,
                 trade_buffer_max_age_ms: int = TRADE_BUFFER_MAX_AGE_MS,
                 ask_history_max_age_ms: int = ASK_HISTORY_MAX_AGE_MS) -> None:
        # market identity/timing
        self.market_id: str | None = None
        self.condition_id: str | None = None
        self.slug: str | None = None
        self.up_token_id: str | None = None
        self.down_token_id: str | None = None
        self.market_start_ts: int | None = None  # ms, epoch
        self.market_end_ts: int | None = None  # ms, epoch
        self.tick_size: float = 0.01

        self._trade_buffer_max_age_ms = trade_buffer_max_age_ms
        self._ask_history_max_age_ms = ask_history_max_age_ms

        # Binance
        self.btc_bid: float | None = None
        self.btc_ask: float | None = None
        self.btc_last: float | None = None
        self.binance_bids: list[tuple[float, float]] = []
        self.binance_asks: list[tuple[float, float]] = []
        self.binance_last_update_ts: int | None = None

        self.price_history = TimeSeries(price_history_max_age_ms)
        self.second_returns: deque[float] = deque(maxlen=SECOND_RETURNS_MAX_LEN)
        self._last_return_bucket: int | None = None

        self.trade_buffer: deque[tuple[int, float, bool]] = deque()  # (ts_ms, qty, is_buy)

        # Chainlink
        self.chainlink_twap: float | None = None
        self.chainlink_twap_observation_ts: int | None = None
        self.chainlink_reference_price: float | None = None  # R0, latched at market start

        # Polymarket
        self.up = TokenBook(ask_history=TimeSeries(ask_history_max_age_ms))
        self.down = TokenBook(ask_history=TimeSeries(ask_history_max_age_ms))

        self.now_ms: int = 0

    @property
    def btc_mid(self) -> float | None:
        if self.btc_bid is None or self.btc_ask is None:
            return None
        return (self.btc_bid + self.btc_ask) / 2.0

    @property
    def binance_valid(self) -> bool:
        if self.btc_bid is None or self.btc_ask is None:
            return False
        return self.btc_bid <= self.btc_ask

    def token_book(self, direction: str) -> TokenBook:
        return self.up if direction == "UP" else self.down

    # ---- Binance updates ----

    def on_binance_trade(self, ts_ms: int, price: float, qty: float, is_buyer_maker: bool) -> None:
        self.now_ms = max(self.now_ms, ts_ms)
        self.btc_last = price
        # isBuyerMaker=True means the taker was a seller -> aggressive sell volume.
        is_buy = not is_buyer_maker
        self.trade_buffer.append((ts_ms, qty, is_buy))
        cutoff = ts_ms - self._trade_buffer_max_age_ms
        while self.trade_buffer and self.trade_buffer[0][0] < cutoff:
            self.trade_buffer.popleft()

    def on_binance_book_ticker(self, ts_ms: int, bid: float, ask: float) -> None:
        """Updates the live best-bid/ask display fields only. Deliberately
        does NOT feed price_history/momentum: bookTicker fires faster than
        anything we persist raw, so if it drove momentum, live features
        would be unreproducible offline (see OFFLINE-01..03). depth10, which
        IS persisted raw every ~100ms, is the single source of truth for
        price used by both live and discovery.py -- see on_binance_depth.
        """
        self.now_ms = max(self.now_ms, ts_ms)
        if bid > ask:
            return  # crossed book: reject, keep last-known-good state
        self.btc_bid = bid
        self.btc_ask = ask
        self.binance_last_update_ts = ts_ms

    def on_binance_depth(self, ts_ms: int, bids: list[tuple[float, float]], asks: list[tuple[float, float]]) -> None:
        self.now_ms = max(self.now_ms, ts_ms)
        self.binance_bids = sorted(bids, key=lambda x: -x[0])
        self.binance_asks = sorted(asks, key=lambda x: x[0])
        if self.binance_bids and self.binance_asks:
            best_bid, best_ask = self.binance_bids[0][0], self.binance_asks[0][0]
            if best_bid <= best_ask:
                mid = (best_bid + best_ask) / 2.0
                self.price_history.append(ts_ms, mid)
                self._maybe_record_second_return(ts_ms, mid)

    def _maybe_record_second_return(self, ts_ms: int, mid: float) -> None:
        bucket = ts_ms // 1000
        if self._last_return_bucket is not None and bucket == self._last_return_bucket:
            return
        past = self.price_history.value_before(ts_ms - 1000)
        if past is not None:
            _, past_val = past
            if past_val > 0:
                self.second_returns.append(math.log(mid / past_val))
        self._last_return_bucket = bucket

    # ---- Polymarket updates ----

    def on_poly_book(self, direction: str, ts_ms: int, bid: float | None, ask: float | None,
                      bid_size: float | None, ask_size: float | None) -> None:
        self.now_ms = max(self.now_ms, ts_ms)
        book = self.token_book(direction)
        if bid is not None and ask is not None and bid > ask:
            return
        book.bid, book.ask = bid, ask
        book.bid_size, book.ask_size = bid_size, ask_size
        book.last_update_ts = ts_ms
        if ask is not None:
            book.ask_history.append(ts_ms, ask)

    def on_poly_trade(self, direction: str, price: float, size: float | None) -> None:
        book = self.token_book(direction)
        book.last_trade_price = price
        book.last_trade_size = size

    # ---- Chainlink updates ----

    def on_chainlink_twap(self, observation_ts_ms: int, value: float) -> None:
        if self.chainlink_twap_observation_ts is not None and observation_ts_ms < self.chainlink_twap_observation_ts:
            return  # stale/out-of-order observation
        self.chainlink_twap = value
        self.chainlink_twap_observation_ts = observation_ts_ms
        if self.chainlink_reference_price is None:
            self.chainlink_reference_price = value

    def reset_for_new_market(self, market_id: str, condition_id: str, slug: str,
                              up_token_id: str, down_token_id: str,
                              start_ts_ms: int, end_ts_ms: int, tick_size: float) -> None:
        self.market_id = market_id
        self.condition_id = condition_id
        self.slug = slug
        self.up_token_id = up_token_id
        self.down_token_id = down_token_id
        self.market_start_ts = start_ts_ms
        self.market_end_ts = end_ts_ms
        self.tick_size = tick_size
        self.up = TokenBook(ask_history=TimeSeries(self._ask_history_max_age_ms))
        self.down = TokenBook(ask_history=TimeSeries(self._ask_history_max_age_ms))
        self.chainlink_reference_price = None
