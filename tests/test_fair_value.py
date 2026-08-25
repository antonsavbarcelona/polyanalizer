from poly_analyzer.fair_value import fair_value_v0


def test_symmetric_price_gives_half():
    fv = fair_value_v0(reference_price=100.0, current_price=100.0, sigma_1s=0.001,
                        remaining_s=60.0, chainlink_observation_ts_ms=1000, now_ms=1000)
    assert abs(fv.fair_up - 0.5) < 1e-9


def test_price_above_reference_is_bullish():
    fv = fair_value_v0(100.0, 101.0, sigma_1s=0.001, remaining_s=60.0,
                        chainlink_observation_ts_ms=1000, now_ms=1000)
    assert fv.fair_up > 0.5


def test_price_below_reference_is_bearish():
    fv = fair_value_v0(100.0, 99.0, sigma_1s=0.001, remaining_s=60.0,
                        chainlink_observation_ts_ms=1000, now_ms=1000)
    assert fv.fair_up < 0.5


def test_shorter_horizon_is_more_extreme():
    short = fair_value_v0(100.0, 101.0, sigma_1s=0.001, remaining_s=5.0,
                           chainlink_observation_ts_ms=1000, now_ms=1000)
    long = fair_value_v0(100.0, 101.0, sigma_1s=0.001, remaining_s=500.0,
                          chainlink_observation_ts_ms=1000, now_ms=1000)
    assert short.fair_up > long.fair_up > 0.5


def test_higher_volatility_pulls_toward_half():
    low_vol = fair_value_v0(100.0, 101.0, sigma_1s=0.0005, remaining_s=60.0,
                             chainlink_observation_ts_ms=1000, now_ms=1000)
    high_vol = fair_value_v0(100.0, 101.0, sigma_1s=0.005, remaining_s=60.0,
                              chainlink_observation_ts_ms=1000, now_ms=1000)
    assert 0.5 < high_vol.fair_up < low_vol.fair_up


def test_always_bounded_0_1_and_complementary():
    fv = fair_value_v0(100.0, 500.0, sigma_1s=0.0001, remaining_s=1.0,
                        chainlink_observation_ts_ms=1000, now_ms=1000)
    assert 0.0 <= fv.fair_up <= 1.0
    assert abs(fv.fair_up + fv.fair_down - 1.0) < 1e-9


def test_zero_volatility_does_not_crash():
    fv = fair_value_v0(100.0, 101.0, sigma_1s=0.0, remaining_s=60.0,
                        chainlink_observation_ts_ms=1000, now_ms=1000)
    assert fv.fair_up is None


def test_stale_chainlink_blocks_fair_value():
    fv = fair_value_v0(100.0, 101.0, sigma_1s=0.001, remaining_s=60.0,
                        chainlink_observation_ts_ms=1000, now_ms=1000 + 10_000)
    assert fv.fair_up is None
