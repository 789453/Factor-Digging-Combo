from __future__ import annotations

import numpy as np
import pandas as pd

from src.alpha_mvp import fastops
from src.alpha_mvp.data import load_crypto_parquet, load_futures_parquet, make_simulated_data
from src.alpha_mvp.evaluator import BatchEvaluator, make_panels
from src.alpha_mvp.fields import (
    DEFAULT_FEATURES,
    FIELD_SPECS,
    add_basic_features,
    add_selected_features,
    FULL_MARKET_OPTIMIZED_FIELDS,
)
from src.alpha_mvp.crypto_fields import (
    CRYPTO_DEFAULT_FEATURES,
    CRYPTO_FIELD_SPECS,
    add_crypto_features,
)
from src.alpha_mvp.ops import ALL_OPERATORS, OPERATOR_SPECS, gate_neg, gate_pos
from src.alpha_mvp.parser import canonical, parse_expr
from src.alpha_mvp.validator import Validator


def test_canonicalization_respects_commutativity():
    left = canonical(parse_expr("Add($b,$a)"))
    right = canonical(parse_expr("Add($a,$b)"))
    assert left == right == "Add($a,$b)"
    assert canonical(parse_expr("Sub($b,$a)")) != canonical(
        parse_expr("Sub($a,$b)")
    )


def test_validator_rejects_unsafe_log_and_bad_window():
    validator = Validator({"x"}, {10})
    assert not validator.validate(parse_expr("Log($x)")).ok
    assert not validator.validate(parse_expr("TsMean($x,20)")).ok
    assert validator.validate(parse_expr("Rank(TsMean($x,10))")).ok


def test_cross_section_rank_matches_pandas_average_ties():
    values = np.array([
        [1.0, 1.0, 3.0, np.nan],
        [4.0, 2.0, 2.0, 1.0],
    ])
    actual = fastops.rank_cs(values)
    expected = pd.DataFrame(values).rank(axis=1, pct=True).to_numpy()
    assert np.allclose(actual, expected, equal_nan=True)


def test_time_series_zscore_and_regime_gates_match_reference():
    values = np.array([
        [1.0, 4.0], [2.0, np.nan], [3.0, 4.0], [4.0, 4.0],
        [5.0, 8.0], [np.nan, 10.0],
    ])
    actual = fastops.fast_rolling_zscore(values, 4)
    expected = np.full_like(values, np.nan)
    for column in range(values.shape[1]):
        series = pd.Series(values[:, column])
        mean = series.rolling(4, min_periods=3).mean()
        std = series.rolling(4, min_periods=3).std(ddof=0)
        expected[:, column] = ((series - mean) / std).to_numpy()
    expected[:3] = np.nan
    assert np.allclose(actual, expected, equal_nan=True)
    state = np.array([[1.0, -1.0, 0.0, np.nan]])
    signal = np.array([[2.0, 3.0, 4.0, 5.0]])
    assert np.allclose(gate_pos(state, signal), [[2.0, 0.0, 0.0, 0.0]])
    assert np.allclose(gate_neg(state, signal), [[0.0, 3.0, 0.0, 0.0]])


def test_sliding_corr_and_std_match_pandas_with_missing_values():
    rng = np.random.default_rng(42)
    left = rng.normal(size=(80, 3))
    right = 0.3 * left + rng.normal(size=(80, 3))
    left[7:12, 1] = np.nan
    right[20:24, 2] = np.nan
    window = 12
    actual_corr = fastops.fast_rolling_corr(left, right, window)
    actual_std = fastops.fast_rolling_std(left, window)
    expected_corr = np.column_stack([
        pd.Series(left[:, index]).rolling(window, min_periods=window // 2).corr(
            pd.Series(right[:, index])
        )
        for index in range(left.shape[1])
    ])
    expected_std = np.column_stack([
        pd.Series(left[:, index]).rolling(
            window, min_periods=window // 2 + 1
        ).std(ddof=0)
        for index in range(left.shape[1])
    ])
    expected_corr[:window - 1] = np.nan
    expected_std[:window - 1] = np.nan
    assert np.allclose(actual_corr, expected_corr, equal_nan=True, atol=1e-10)
    assert np.allclose(actual_std, expected_std, equal_nan=True, atol=1e-10)


def test_rank_ic_matches_pandas_with_ties():
    x = np.tile(np.array([1, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10], dtype=float), (3, 1))
    y = np.tile(np.array([9, 8, 7, 6, 5, 4, 4, 3, 2, 1, 0, -1], dtype=float), (3, 1))
    actual = fastops.daily_corr(x, y, rank=True)
    expected = pd.Series(x[0]).rank().corr(pd.Series(y[0]).rank())
    assert np.allclose(actual, expected)


def test_field_to_expression_chain():
    raw = make_simulated_data(n_days=50, n_stocks=40, seed=3)
    featured = add_basic_features(raw)
    assert set(DEFAULT_FEATURES).issubset(featured.columns)
    panels, dates, codes = make_panels(featured, DEFAULT_FEATURES, "close")
    evaluator = BatchEvaluator(
        {name: panel for name, panel in panels.items() if name != "close"},
        dates,
        codes,
        windows=(10, 20),
    )
    values, status = evaluator.eval_expr("Rank(TsMean($ret_1d,10))")
    assert status == "OK"
    assert values.shape == (50, 40)
    assert np.isfinite(values).any()


def test_feature_prefix_is_not_changed_by_future_rows():
    raw = make_simulated_data(n_days=45, n_stocks=20, seed=9)
    cutoff = sorted(raw["trade_date"].unique())[29]
    short = add_basic_features(raw[raw["trade_date"] <= cutoff].copy())
    full = add_basic_features(raw.copy())
    keys = ["trade_date", "ts_code"]
    merged = short[keys + list(DEFAULT_FEATURES)].merge(
        full[keys + list(DEFAULT_FEATURES)],
        on=keys,
        suffixes=("_short", "_full"),
    )
    for field in DEFAULT_FEATURES:
        assert np.allclose(
            merged[f"{field}_short"],
            merged[f"{field}_full"],
            equal_nan=True,
        ), field


def test_selected_feature_path_matches_full_builder():
    raw = make_simulated_data(n_days=45, n_stocks=20, seed=12)
    fields = sorted(FULL_MARKET_OPTIMIZED_FIELDS)
    full = add_basic_features(raw.copy())
    selected = add_selected_features(raw.copy(), fields)
    merged = selected.merge(
        full[["trade_date", "ts_code", *fields]],
        on=["trade_date", "ts_code"],
        suffixes=("_selected", "_full"),
    )
    for field in fields:
        assert np.allclose(
            merged[f"{field}_selected"],
            merged[f"{field}_full"],
            equal_nan=True,
        ), field


def test_field_and_operator_catalogs_cover_executable_vocabulary():
    assert set(DEFAULT_FEATURES).issubset(FIELD_SPECS)
    assert all(FIELD_SPECS[name].role == "search" for name in DEFAULT_FEATURES)
    assert set(OPERATOR_SPECS) == ALL_OPERATORS


def test_futures_oi_fields_handle_missing_and_extreme_inputs():
    raw = make_simulated_data(n_days=30, n_stocks=3, seed=19)
    missing = add_basic_features(raw.copy())
    oi_fields = [name for name, spec in FIELD_SPECS.items() if spec.group == "futures_oi"]
    assert missing[oi_fields].isna().all().all()

    extreme = raw.copy()
    extreme["open_interest"] = np.resize(
        np.array([0.0, 1.0, 1e-300, 1e300, -1.0, np.nan]), len(extreme)
    )
    featured = add_basic_features(extreme)
    assert not np.isinf(featured[oi_fields].to_numpy(dtype=float)).any()
    assert featured["oi_change_1d"].notna().any()


def test_futures_loader_rejects_unknown_universe(tmp_path):
    path = tmp_path / "empty.parquet"
    pd.DataFrame().to_parquet(path)
    with np.testing.assert_raises_regex(ValueError, "Unsupported futures universe"):
        load_futures_parquet(str(path), "unsupported", "20240101", "20240131")


def test_crypto_fields_handle_missing_and_extreme_inputs():
    dates = pd.date_range("2025-01-01", periods=500, freq="h", tz="UTC")
    index = pd.MultiIndex.from_product(
        [["BTCUSDT", "ETHUSDT"], dates], names=["ts_code", "date"]
    )
    raw = index.to_frame(index=False)
    raw["trade_date"] = raw["date"].dt.strftime("%Y%m%d%H%M")
    step = np.tile(np.arange(500, dtype=float), 2)
    raw["open"] = 100.0 + step * 0.01
    raw["close"] = raw["open"] * (1.0 + np.sin(step) * 0.001)
    raw["high"] = np.maximum(raw["open"], raw["close"]) * 1.002
    raw["low"] = np.minimum(raw["open"], raw["close"]) * 0.998
    raw["pre_close"] = raw.groupby("ts_code")["close"].shift(1)
    raw["quote_volume"] = 1e6 + step * 100
    raw["vol"] = 1000.0 + step
    raw["amount"] = raw["quote_volume"]
    raw["trade_count"] = 1000 + step
    raw["taker_buy_volume"] = raw["vol"] * 0.52
    raw["taker_sell_volume"] = raw["vol"] * 0.48
    raw["taker_buy_ratio"] = 0.50 + 0.02 * np.sin(step / 5.0)
    raw["vwap"] = (raw["open"] + raw["close"]) / 2
    raw["micro_bar_count"] = 4
    for column, value in {
        "micro_first_return": 0.001,
        "micro_last_return": -0.0005,
        "micro_realized_vol": 0.003,
        "micro_volume_concentration": 0.3,
        "micro_taker_imbalance": 0.04,
        "micro_trade_concentration": 0.28,
        "micro_volume_trend": 0.1,
        "micro_imbalance_trend": -0.1,
        "micro_range_expansion": 1.2,
        "micro_vwap_dispersion": 0.0003,
        "micro_path_efficiency": 0.6,
        "micro_return_reversal": -0.0015,
        "micro_upside_share": 0.55,
        "micro_jump_share": 0.4,
        "micro_volume_hhi": 0.27,
        "micro_trade_hhi": 0.26,
        "micro5_realized_vol": 0.0032,
        "micro5_upside_share": 0.52,
        "micro5_jump_share": 0.25,
        "micro5_path_efficiency": 0.5,
        "micro5_taker_imbalance": 0.03,
        "micro5_volume_trend": 0.05,
        "micro5_imbalance_trend": -0.03,
        "micro5_volume_hhi": 0.09,
        "micro5_trade_hhi": 0.085,
    }.items():
        raw[column] = value
    raw.loc[0, "quote_volume"] = 0.0
    raw.loc[1, "trade_count"] = 0.0
    raw.loc[2, "high"] = np.inf
    featured = add_crypto_features(raw)
    assert set(CRYPTO_DEFAULT_FEATURES) == set(CRYPTO_FIELD_SPECS)
    assert set(CRYPTO_DEFAULT_FEATURES).issubset(featured.columns)
    assert not np.isinf(featured[CRYPTO_DEFAULT_FEATURES].to_numpy()).any()
    assert featured[CRYPTO_DEFAULT_FEATURES].notna().any().all()


def test_crypto_multiscale_loader_aggregates_5m_and_15m_without_infinity(tmp_path):
    symbol_dir = tmp_path / "BTCUSDT"
    symbol_dir.mkdir()
    hours = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")

    def bars(index):
        count = len(index)
        open_ = 100.0 + np.arange(count) * 0.01
        close = open_ * (1.0 + 0.0002 * np.sin(np.arange(count)))
        return pd.DataFrame({
            "date": index, "open": open_, "high": np.maximum(open_, close) * 1.001,
            "low": np.minimum(open_, close) * 0.999, "close": close,
            "volume": np.arange(count, dtype=float) + 10.0,
            "quote_volume": (np.arange(count, dtype=float) + 10.0) * close,
            "trade_count": np.arange(count, dtype=float) + 20.0,
            "taker_buy_volume": (np.arange(count, dtype=float) + 10.0) * 0.52,
            "taker_sell_volume": (np.arange(count, dtype=float) + 10.0) * 0.48,
            "taker_buy_ratio": 0.52,
            "log_return": np.log(close / open_),
            "range_pct": 0.002,
            "body_pct": (close - open_) / open_,
            "vwap": (open_ + close) / 2.0,
        })

    bars(hours).to_parquet(symbol_dir / "1h.parquet", index=False)
    bars(pd.date_range(hours[0], periods=12, freq="15min", tz="UTC")).to_parquet(
        symbol_dir / "15m.parquet", index=False
    )
    bars(pd.date_range(hours[0], periods=36, freq="5min", tz="UTC")).to_parquet(
        symbol_dir / "5m.parquet", index=False
    )
    loaded = load_crypto_parquet(
        str(tmp_path), "2025-01-01", "2025-01-01 02:59:59", "1h", "15m", "5m"
    )
    assert loaded["micro_bar_count"].tolist() == [4, 4, 4]
    assert loaded["micro5_bar_count"].tolist() == [12, 12, 12]
    assert not np.isinf(loaded.select_dtypes(include=[np.number]).to_numpy()).any()
