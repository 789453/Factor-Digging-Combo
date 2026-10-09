# Alpha Research Framework

**当前有效成果（2026-10-09）**：[完整研究报告](docs/EFFECTIVE_FACTOR_AND_COMBO_RESEARCH_RESULTS_20261009.md)、[策略与因子产品卡](docs/EFFECTIVE_FACTOR_AND_COMBO_PRODUCT_CARDS_20261009.md)、[41页交互总览](visualizations/effective_combo_20261009_full_review_v2/index.html)。组合／15叶两套历史基线验证费后+5.5922%／+7.1711%，三发现折、两个完整验证季度及1月均正，8bps仍正；164测试及模拟CLI通过。原自动第一失败保留，不按验证改参数。产品为共同方向／条件择时，不标纯残差Alpha或独立未来保证。下文全亏损是较早阶段结果。

这是一个支持加密货币、期货和股票的多资产因子研究框架。近期主航向是加密货币合约的逐资产时间序列因子挖掘；股票和期货的截面研究接口继续保留。

框架的目标不是保存一批“最终因子”，而是可复现地完成：

1. 版本化构建基础字段；
2. 通过有界 YAML 模板确定性生成、校验和去重表达式；
3. 用 discovery / validation / holdout 防火墙评估；
4. 通过初筛、中筛、细筛高效处理大候选池；
5. 保留有信息线索及条件适用性的预测特征，控制冗余并建立组合预测；
6. 累积字段、算子、窗口、频率和模板证据，为结构化/半挖掘因子提供基础。

## 当前优先级 预测特征与直接组合（2026-10-09）

最新[深度研究报告](docs/PREDICTIVE_FACTOR_DEEP_RESEARCH_RESULTS_20261009.md)进一步拆开弱信号的共同方向、均值偏差和动态幅度；同一冻结池扩展／十二个月训练共十二模型的验证损失仍均不如零预测，固定交易映射均亏损。数值特征、对齐目标、全部预测和失败解释已经交付；157测试、模拟与有限十二币真实核验通过，不新搜索、不读holdout。[完整结果页](visualizations/predictive_factor_deep_review_20261009_v1/index.html)。滚动跨度没有被升级为默认赢家，后续先研究一个成熟标签校准／更新假设。

最新[整改报告](docs/PREDICTIVE_FACTOR_RESEARCH_RECTIFICATION_20261009.md)与[项目级规范](docs/PREDICTIVE_FACTOR_RESEARCH_POLICY_20261009.md)优先约束后续研究：预测池不要求单因子独立支付交易费用，组合预测与交易准入分别交付。重用R5候选完成48条OLD、48条混合函数和六个直接Ridge基线；结果可观察但历史验证仍弱、固定交易映射均亏损，不称已得到稳定Alpha。151项测试通过。

当前配置为 `configs/research/complex_alpha_prediction_reuse_20261009.yaml`，通过同一 `research_cli --config`、同一 `aligned_crypto/complex_alpha` 路由的候选复用阶段执行。完成输出不可覆盖，复跑须改版本和输出目录。默认重用旧候选，重型诊断关闭，不新开大搜索。[结果页](visualizations/complex_alpha_prediction_reuse_20261009_v1/index.html)。

## 复杂表示与联合 Alpha 历史研究（2026-10-08—10-09）

在原 `aligned_crypto` 生产入口内，已加入分钟分布、单侧尺度、独立三阶对数签名、状态转移、SPD几何、有限记忆和核分布表示；主30k＋旧表示30k同预算搜索，保留全部试验和折内模型。最新读出1024维包含结构、旧控制及筛选函数，不能称1024条已证Alpha。

详见[研究报告](docs/COMPLEX_ALPHA_RESEARCH_RESULTS_20261008.md)、[字段与算子规格](docs/COMPLEX_ALPHA_FIELDS_AND_OPERATORS_20261008.md)及[迭代日志](docs/COMPLEX_ALPHA_ITERATION_LOG_20261008.md)。原R3联合候选在两段已见历史迁移中有稀疏费后正结果，固定家族移除支持旧非线性和转移结构共同参与；证据及边界见[产品卡](docs/COMPLEX_ALPHA_PRODUCT_CARD_20261008.md)和[固定移除可视化](visualizations/complex_alpha_20261009_fixed_readout_ablation_v3/index.html)。

自查修复Pandas日期整数单位对相位抽样和非重叠诊断的影响，[时钟勘误](docs/COMPLEX_ALPHA_CLOCK_CORRIGENDUM_20261008.md)必须联读。五轮真实研究均已完成；R5相同参数修复复跑冻结0入选，主pool12h验证−6.8190%，费用高于毛边际。147项测试、原生产模拟v17和有界真实检查通过；原历史结果不覆盖。

结果入口：[五轮综合交互总览（22页）](visualizations/complex_alpha_20261008_final_review_v2/index.html)及[R5核心结果（40页）](visualizations/complex_alpha_20261008_round5/index.html)。最终复核状态PASS，覆盖原账本、输入/源码指纹、基线复原和静态链接；静态PNG已检查，未声称浏览器渲染验收。

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

## 人工半结构化 Alpha 研究（2026-10-03）

同一 `aligned_crypto` 入口已加入 20 条逐项声明的加密货币交易机制、Optuna 有界搜参、衍生品 funding 结算和原始 5m 四分之一小时相位字段。六轮真实数据研究最终只有 1 条机制通过发现/验证筛选，且在已见 2026 测试段转负；不能把这 20 条候选称为已证实的高 Alpha。详见[完整研究报告](docs/MANUAL_ALPHA_SEMISTRUCTURED_RESULTS_20261003.md)与[25 页离线总览](visualizations/manual_alpha_semistructured_20261003_round6/index.html)。

本机运行需安装 Optuna；当前可用环境是 `D:\Total_Tools\miniforge3\envs\universal`。已完成输出不可覆盖，复跑时先复制[配置](configs/research/manual_alpha_semistructured_20261003_round6.yaml)并更改版本和输出目录。

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
