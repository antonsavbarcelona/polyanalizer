import pytest

from poly_analyzer.config import ASSETS, build_config


def test_each_asset_gets_its_own_db_path():
    paths = {asset: build_config(asset).recorder.db_path for asset in ASSETS}
    assert len(set(paths.values())) == len(paths)  # all distinct, nothing shared
    assert paths["btc"] == "data/analyzer_btc.db"
    assert paths["eth"] == "data/analyzer_eth.db"
    assert paths["sol"] == "data/analyzer_sol.db"


def test_each_asset_gets_its_own_symbols_and_slug():
    btc = build_config("btc").market
    eth = build_config("eth").market
    sol = build_config("sol").market

    assert btc.binance_symbol == "btcusdt" and btc.chainlink_symbol == "btc/usd"
    assert eth.binance_symbol == "ethusdt" and eth.chainlink_symbol == "eth/usd"
    assert sol.binance_symbol == "solusdt" and sol.chainlink_symbol == "sol/usd"

    slugs = {btc.asset_slug_prefix, eth.asset_slug_prefix, sol.asset_slug_prefix}
    assert slugs == {"btc-updown-15m", "eth-updown-15m", "sol-updown-15m"}


def test_asset_lookup_is_case_insensitive():
    assert build_config("ETH").market.binance_symbol == "ethusdt"


def test_unknown_asset_raises():
    with pytest.raises(ValueError):
        build_config("doge")


def test_signal_thresholds_shared_across_assets():
    """The H1 signal logic itself doesn't change per-asset, only the feeds do."""
    assert build_config("btc").signal == build_config("eth").signal == build_config("sol").signal


def test_debug_flag_propagates():
    cfg = build_config("eth", debug_mode=True)
    assert cfg.debug_mode is True
