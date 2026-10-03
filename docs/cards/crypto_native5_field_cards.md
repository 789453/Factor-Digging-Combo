# 原生 5m 字段知识卡（逐字段声明，v1）

统一公式版本 `2026-09-28-native5-fields-v1`；可执行依赖表为 `src/alpha_mvp/native5_fields.py::NATIVE5_FIELD_SPECS`。每条 5m parquet 的 `date` 是开盘时刻，信号面板 `trade_date=date+5min` 是该 bar 收盘后可用时刻。`ret_15m_completed`、`flow_15m_completed` 通过 15m bar 的 `date+15min` 反向 as-of 合并，不使用未完成 15m bar。所有滚动量逐币种、含当前已完成 bar、窗口不足返回 NaN；输入非有限值传播为 NaN，分母为零使用 `safe_div` 返回 NaN，最终无穷值改为 NaN。极端但有限的正价格/成交输入需保持有限结果；非正价格没有对数收益。没有任何字段使用未来 bar 或 holdout 拟合参数。

| 字段 | 依赖与公式 | 数值/用途 |
|---|---|---|
| `ret_5m`, `ret_15m`, `ret_1h` | 同币 `close` 当前与前 1/3/12 个已完成 5m bar 的对数比 | 无量纲价格变化；名称里的 15m/1h 是**回看窗口**，不是单独调仓时钟 |
| `range_5m` | `(high-low)/open` | 无量纲 bar 波幅 |
| `vwap_bias` | `(close-vwap)/vwap` | 无量纲收盘位置 |
| `taker_imbalance` | `2*taker_buy_ratio-1` | [-1,1] 主动成交方向代理；比率无效则 NaN |
| `volume_shock_12` | `log1p(quote_volume)` 的 12 bar 滚动 z-score | 当前成交量相对过去 1h 的冲击；滚动标准差为零则 NaN |
| `trade_shock_12` | `log1p(trade_count)` 的 12 bar 滚动 z-score | 当前成交笔数冲击 |
| `amihud_shock_12` | `log1p(1e9*abs(ret_5m)/quote_volume)` 的 12 bar 滚动 z-score | 相对冲击代理；未把交易所冲击成本等同于该量 |
| `rv_12` | `sqrt(sum_{k=0}^{11} ret_5m[t-k]^2)` | 过去 1h 实现波动，非未来标签 |
| `upside_share_12` | 过去 12 bar 正收益平方和 / 全部平方和 | [0,1] 上行贡献；零方差返回 NaN |
| `jump_share_12` | 过去 12 bar 最大绝对收益 / 绝对收益和 | [0,1] 路径集中度，非严格跳跃检验 |
| `path_efficiency_12` | `abs(ret_1h)` / 过去 12 bar 绝对收益和 | [0,1] 路径效率 |
| `flow_trend_12` | 当前 `taker_imbalance` 减其过去 12 bar 均值 | [-2,2] 流量变化 |
| `flow_price_agreement` | `taker_imbalance * ret_5m` | 同号成交与价格确认，无量纲 |
| `ret_15m_completed` | 最近已完成 15m bar 的 `log(close/open)` | 无量纲跨源确认，as-of 时间必须不晚于 5m 信号时间 |
| `flow_15m_completed` | 最近已完成 15m bar 的 `2*taker_buy_ratio-1` | [-1,1] 跨源流量确认 |
| `us_day_flag`, `us_evening_flag`, `us_overnight_flag` | 5m 收盘时刻转纽约本地时区，分 08–16、16–24、00–08 | 互斥 0/1，DST 自动转换，非“机构交易事实” |

字段属于原生 5m 决策通道，不得在报告中误写为 5m 聚合至 1h。参考测试覆盖缺失、极端、5m/15m 可用时间和纽约 DST。
