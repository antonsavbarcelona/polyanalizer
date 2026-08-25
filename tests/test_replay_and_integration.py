from poly_analyzer.config import CONFIG
from poly_analyzer.signal_engine import SignalEngine
from tests.test_signal_engine import make_features, make_state

NOW = 1_000_000


def test_signal_engine_replay_is_deterministic():
    """U-LOOK-05: same input sequence, same arrival order -> same signals,
    whether "live" or replayed. SignalEngine reads only its explicit
    arguments (no wall-clock, no randomness), so two independent runs over
    the same scripted (state, features, ts) sequence must match exactly.
    """
    cfg = CONFIG.signal
    state = make_state("UP")
    zs = [1.0, 2.0, 2.6, 3.1, 0.5, 2.7, 3.0]

    def run():
        engine = SignalEngine(cfg)
        out = []
        for i, z in enumerate(zs):
            t = NOW + i * 1000
            features = make_features("UP", z_1s=z)
            sig = engine.evaluate(state, features, t)
            out.append((sig.direction, sig.ts_ms, round(sig.z_1s, 6)) if sig else None)
        return out

    first, second = run(), run()
    assert first == second
    # Only the 2.4->2.6 style crossing at index 2 fires: cooldown (3s) hasn't
    # elapsed by the time z dips to 0.5 at index 4 (only 2s after the
    # signal), so the gate is still disarmed when 2.7 arrives at index 5 --
    # same anti-spam rule as test_hysteresis_reset_without_cooldown_blocks_second_signal.
    assert first.count(None) == len(zs) - 1
    assert first[2] is not None
