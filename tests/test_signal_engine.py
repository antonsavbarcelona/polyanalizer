from poly_analyzer.config import SignalConfig
from poly_analyzer.features import Features
from poly_analyzer.signal_engine import HysteresisGate, SignalEngine
from poly_analyzer.state import MarketState

NOW = 1_000_000


def make_state(direction: str) -> MarketState:
    state = MarketState()
    state.up_token_id = "UP_TOKEN"
    state.down_token_id = "DOWN_TOKEN"
    state.market_end_ts = NOW + 400_000  # 400s remaining
    book = state.up if direction == "UP" else state.down
    if direction == "UP":
        book.bid, book.ask = 0.49, 0.50
    else:
        book.bid, book.ask = 0.49, 0.50
    book.ask_history.append(NOW - 600, book.ask)  # unrepriced: flat over 500ms
    return state


def make_features(direction: str, **overrides) -> Features:
    sign = 1.0 if direction == "UP" else -1.0
    base = dict(
        momentum_250ms=sign * 0.001, momentum_1s=sign * 0.002, momentum_5s=sign * 0.002,
        sigma_1s=0.0004, z_1s=sign * 3.0,
        flow_1s=sign * 0.30, flow_5s=sign * 0.30,
        book_imbalance=sign * 0.15,
        vol_5s=0.0004, vol_30s=0.0004,
    )
    base.update(overrides)
    return Features(**base)


def test_up_signal_all_conditions_pass():
    cfg = SignalConfig()
    engine = SignalEngine(cfg)
    state = make_state("UP")
    features = make_features("UP")
    signal = engine.evaluate(state, features, NOW)
    assert signal is not None
    assert signal.direction == "UP"
    assert signal.token_id == "UP_TOKEN"


def test_down_signal_all_conditions_pass():
    cfg = SignalConfig()
    engine = SignalEngine(cfg)
    state = make_state("DOWN")
    features = make_features("DOWN")
    signal = engine.evaluate(state, features, NOW)
    assert signal is not None
    assert signal.direction == "DOWN"
    assert signal.token_id == "DOWN_TOKEN"


def _check(direction: str, **feature_overrides):
    cfg = SignalConfig()
    engine = SignalEngine(cfg)
    state = make_state(direction)
    features = make_features(direction, **feature_overrides)
    z = features.z_1s
    return engine._check_confirmations(state, features, NOW, direction, z)


def test_z_below_trigger_never_reaches_confirmations_via_gate():
    cfg = SignalConfig()
    engine = SignalEngine(cfg)
    state = make_state("UP")
    features = make_features("UP", z_1s=2.49)
    assert engine.evaluate(state, features, NOW) is None


def test_momentum_wrong_sign_blocks_signal():
    assert _check("UP", momentum_250ms=-0.001) is None


def test_flow_below_threshold_blocks_signal():
    assert _check("UP", flow_1s=0.24) is None


def test_imbalance_below_threshold_blocks_signal():
    assert _check("UP", book_imbalance=0.09) is None


def test_flow_exactly_at_threshold_passes():
    assert _check("UP", flow_1s=0.25) is not None


def test_imbalance_exactly_at_threshold_passes():
    assert _check("UP", book_imbalance=0.10) is not None


def test_poly_already_repriced_blocks_signal():
    """I-04: Binance impulse is real but Polymarket already moved -> no edge."""
    cfg = SignalConfig()
    engine = SignalEngine(cfg)
    state = make_state("UP")
    state.up.ask_history.append(NOW - 600, 0.46)  # ask was .46 500ms ago, now .50 -> +4c already
    features = make_features("UP")
    assert engine._check_confirmations(state, features, NOW, "UP", features.z_1s) is None


def test_spread_too_wide_blocks_signal():
    cfg = SignalConfig()
    engine = SignalEngine(cfg)
    state = make_state("UP")
    state.up.bid, state.up.ask = 0.46, 0.50  # 4c spread
    state.up.ask_history.append(NOW - 600, 0.50)
    features = make_features("UP")
    assert engine._check_confirmations(state, features, NOW, "UP", features.z_1s) is None


def test_ask_out_of_range_blocks_signal():
    cfg = SignalConfig()
    engine = SignalEngine(cfg)
    state = make_state("UP")
    state.up.bid, state.up.ask = 0.18, 0.19
    state.up.ask_history.append(NOW - 600, 0.19)
    features = make_features("UP")
    assert engine._check_confirmations(state, features, NOW, "UP", features.z_1s) is None


def test_time_remaining_too_low_blocks_signal():
    cfg = SignalConfig()
    engine = SignalEngine(cfg)
    state = make_state("UP")
    state.market_end_ts = NOW + 89_000  # 89s < 90s minimum
    features = make_features("UP")
    assert engine._check_confirmations(state, features, NOW, "UP", features.z_1s) is None


def test_time_remaining_too_high_blocks_signal():
    cfg = SignalConfig()
    engine = SignalEngine(cfg)
    state = make_state("UP")
    state.market_end_ts = NOW + 781_000
    features = make_features("UP")
    assert engine._check_confirmations(state, features, NOW, "UP", features.z_1s) is None


# ---- Hysteresis / anti-spam ----

def test_hysteresis_single_excursion_yields_one_trigger():
    cfg = SignalConfig()
    gate = HysteresisGate(cfg)
    zs = [2.4, 2.6, 3.1, 3.4, 2.8]
    triggers = [gate.observe(z, NOW + i * 100) for i, z in enumerate(zs)]
    assert triggers.count(True) == 1
    assert triggers[1] is True  # fires exactly on the 2.4 -> 2.6 crossing


def test_hysteresis_reset_then_new_excursion_yields_two_triggers():
    cfg = SignalConfig()
    gate = HysteresisGate(cfg)
    t = NOW
    assert gate.observe(2.6, t) is True
    gate.consume(t)
    t += int((cfg.cooldown_s + 1) * 1000)
    assert gate.observe(0.8, t) is False  # reset (armed again)
    t += 100
    assert gate.observe(2.7, t) is True  # second, independent excursion


def test_hysteresis_reset_without_cooldown_blocks_second_signal():
    cfg = SignalConfig()
    gate = HysteresisGate(cfg)
    t = NOW
    gate.observe(2.6, t)
    gate.consume(t)
    t += 500  # well under 3s cooldown
    gate.observe(0.8, t)  # z<1.0 but cooldown not elapsed -> stays disarmed
    t += 100
    assert gate.observe(2.7, t) is False


def test_hysteresis_no_new_excursion_without_dip_below_trigger():
    cfg = SignalConfig()
    gate = HysteresisGate(cfg)
    t = NOW
    gate.observe(2.6, t)
    gate.consume(t)
    t += int((cfg.cooldown_s + 1) * 1000)
    # cooldown elapsed but z stayed >= trigger the whole time (no crossing) -> armed but in_excursion still True
    assert gate.observe(2.8, t) is False
