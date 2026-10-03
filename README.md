# Alpha Research Framework

这是一个支持加密货币、期货和股票的多资产因子研究框架。近期主航向是加密货币合约的逐资产时间序列因子挖掘；股票和期货的截面研究接口继续保留。

框架的目标不是保存一批“最终因子”，而是可复现地完成：

1. 版本化构建基础字段；
2. 通过有界 YAML 模板确定性生成、校验和去重表达式；
3. 用 discovery / validation / holdout 防火墙评估；
4. 通过初筛、中筛、细筛高效处理大候选池；
5. 选择结构多样、回测稳健的因子并形成低相似度配对；
6. 累积字段、算子、窗口、频率和模板证据，为结构化/半挖掘因子提供基础。

## 最新一体化研究（2026-09-30）

最新完成的加密货币研究使用 2023-01 至 2025-06 发现期、2025-07 至 2026-01
验证期和 2026-02 至 2026-09-24 测试期，主赛道粗筛 10,000 条表达式，
另有 1,000 条原生 5m 表达式。挖掘、幅度标定、组合和执行使用同一未来 5m
收益路径与 4 bps 单位换仓费用。测试期主组合费后 +0.73%，仍未达到强基线标准；
归因和证据边界见[研究报告](docs/CRYPTO_ALIGNED_FACTOR_RESEARCH_20260930.md)。

克隆仓库后可离线打开 [48 页交互可视化](docs/visualizations/aligned_crypto_v9/index.html)，
包含搜索漏斗、因子公式、分通道收益、分期执行和 12 币逐段页面。
图表是已完成实验的只读快照。原始行情、Parquet 账本、缓存及大规模生成结果
不随 Git 仓库分发；复算真实数据实验需自行配置本地数据根目录，并使用**新的**输出目录。
生产入口仍只有：

```powershell
python -m src.alpha_mvp.research_cli --config configs/research/crypto_aligned_11000_native5_20260930_v9.yaml
```

## 历史冻结实验

较早认可并冻结的加密货币实验位于：

```text
outputs/crypto_multiscale_sessions_30000_robust20_20260926
```

它对 12 个合约使用 1h 决策面板，并加入向量化的 15m/5m 小时内特征；生成 30,000 个纯时间序列表达式，最终保留 20 个 signed long-short 因子和 10 组配对。详细结论见 [阶段状态与路线图](docs/PROJECT_STATUS_AND_ROADMAP_20260928.md)。

## 唯一生产入口

所有研究必须使用：

```powershell
python -m src.alpha_mvp.research_cli --config <config.yaml>
```

较早加密货币 30,000 表达式配置：

```powershell
python -m src.alpha_mvp.research_cli --config configs/research/crypto_multiscale_sessions_30000.yaml
```

加密货币模拟 smoke：

```powershell
python -m src.alpha_mvp.research_cli --config configs/research/crypto_multiscale_sessions_smoke.yaml
```

5m/15m 小时内特征的有界执行尺度检查：

```powershell
python -m src.alpha_mvp.research_cli --config configs/research/crypto_execution_probe_purged_20260928.yaml
```

该检查仍使用 1h 决策面板；已完成实验不可原目录重跑，复跑时需在配置中指定新的输出目录。研究解释见 [执行尺度与聚类可行性](docs/CRYPTO_EXECUTION_AND_CLUSTER_RESEARCH_20260928.md)。

当前机制研究示例使用同一入口和角色约束模板，将预测目标显式设为未来 5m 上/下行半方差。四份小规模配置位于 `configs/research/crypto_mechanism_*_baselines_20260928.yaml`；已完成输出目录不可覆盖。研究问题、基线与半结构化工作流见[机制研究设计](docs/CRYPTO_MECHANISM_RESEARCH_DESIGN_20260928.md)。

股票真实数据配置示例：

```powershell
python -m src.alpha_mvp.research_cli --config configs/research/real_2500.yaml
```

商品期货 OI 快速研究示例：

```powershell
python -m src.alpha_mvp.research_cli --config configs/research/futures_commodity_oi_rapid.yaml
```

配置错误、字段缺失或数据源不匹配必须明确失败，不允许静默替换模式或数据集。

## 当前主流程

```text
YAML 配置
→ 数据提取与版本化字段
→ 对齐 T×N 面板和严格签名缓存
→ 确定性模板生成 / Parser / Canonical / Validator
→ 初筛：向量化主指标与基础有效性
→ 中筛：分段/时段生存与结构多样性
→ 细筛：净 long-short、Sharpe、IC/ICIR、换手、成本与稳健性
→ 多样性选择与低相似度配对
→ 冻结研究决定
→ Holdout 隔离审计
→ HTML / CSV / Manifest 不可覆盖产物
```

加密货币当前默认逐资产时间序列模式，不使用根部截面 `Rank`。策略允许每个合约独立多空；方向只从 discovery 推断。validation 用于生存与排序，holdout 只在选择冻结后审计。

## 目录

```text
configs/research/                         实验与模板配置
src/alpha_mvp/fields.py                   股票/通用字段
src/alpha_mvp/crypto_fields.py            加密货币多尺度及时段字段
src/alpha_mvp/ops.py                      算子语义
src/alpha_mvp/research/                   生成、评估、筛选、缓存、报告与编排
docs/                                     项目状态、协议、知识卡和实施说明
tests/                                    数值语义与端到端回归
outputs/                                  不可覆盖的实验证据
```

## 主要产物

通用产物：

```text
manifest.json
search_results.csv
selected_factors.csv
factor_pairs.csv
holdout_audit.csv
attribution_field.csv
attribution_operator.csv
attribution_template.csv
attribution_window.csv
overview.html
summary.json
```

当前加密货币报告还包括：

```text
crypto_time_series_backtest_metrics.csv
crypto_time_series_factor_returns.html
crypto_time_series_pair_metrics.csv
crypto_time_series_factor_pairs.html
```

累计收益报告按 validation 净累计收益展示 1–5、6–15、16–20 三组，并包含前 5 信号分组与纽约时段热力图。图表排序不改变多目标选择规则，更不能读取 holdout 反向调参。

## 当前限制与下一阶段

- 5m/15m 当前被聚合为 1h 面板特征，尚不是原生中频调仓策略；下一阶段需分离特征、信号、调仓和持有频率。
- 最终因子可包含市场、趋势、波动、流动性或时段 beta；下一阶段将增加 alpha/beta/style 暴露分层和中性化后的剩余表现。
- 下一阶段将引入角色约束模板、快变量转慢状态、经济意义门控和专家结构模块，同时保持 YAML、有界复杂度、归因和测试要求。

完整要求见 [长期研究需求](docs/RESEARCH_REQUIREMENTS.md)。

## 开发与交接

```powershell
python -m pytest -q
```

新增或修改字段、算子、模板时，必须同步更新元数据、知识卡、公式版本和数值测试。新窗口按 [文档导航](docs/README.md) 阅读；完整约束见 [研究协议](docs/RESEARCH_PROTOCOL.md)、[架构说明](docs/ARCHITECTURE.md) 和根目录 `AGENTS.md`。
