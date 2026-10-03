from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_futures_data_profile(raw: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    """Write descriptive source-quality evidence, separate from factor scoring."""
    required = {"trade_date", "ts_code", "open_interest", "vol", "close"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Futures profile cannot be built; missing columns: {missing}")
    profile = raw.groupby("trade_date", sort=True).agg(
        instruments=("ts_code", "nunique"),
        oi_coverage=("open_interest", lambda value: value.notna().mean()),
        volume_coverage=("vol", lambda value: value.notna().mean()),
        price_coverage=("close", lambda value: value.notna().mean()),
        median_oi=("open_interest", "median"),
        median_volume=("vol", "median"),
    ).reset_index()
    profile.to_csv(out_dir / "futures_data_profile.csv", index=False, encoding="utf-8-sig")

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "每日截面合约数",
            "OI、成交量与价格覆盖率",
            "截面 OI 中位数",
            "截面成交量中位数",
        ),
    )
    x = profile["trade_date"]
    figure.add_trace(go.Scatter(x=x, y=profile["instruments"], name="合约数"), 1, 1)
    for column, label in (
        ("oi_coverage", "OI"),
        ("volume_coverage", "成交量"),
        ("price_coverage", "结算价"),
    ):
        figure.add_trace(go.Scatter(x=x, y=profile[column], name=label), 1, 2)
    figure.add_trace(go.Scatter(x=x, y=profile["median_oi"], name="中位 OI"), 2, 1)
    figure.add_trace(go.Scatter(x=x, y=profile["median_volume"], name="中位成交量"), 2, 2)
    figure.update_layout(
        title="期货研究输入质量：连续/主力合约截面（描述性，不参与因子评分）",
        template="plotly_white",
        height=780,
    )
    figure.update_yaxes(tickformat=".0%", row=1, col=2)
    path = out_dir / "futures_data_profile.html"
    figure.write_html(path, include_plotlyjs="cdn")
    return {
        "futures_data_profile": "futures_data_profile.html",
        "futures_data_profile_csv": "futures_data_profile.csv",
    }
