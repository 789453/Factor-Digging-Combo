# 有效因子与组合产品卡

本卡用于重现、观察和继续研究当前有效基线；完整论证见[研究报告](EFFECTIVE_FACTOR_AND_COMBO_RESEARCH_RESULTS_20261009.md)，价格／预测／仓位与逐币路径见[41页交互总览](../visualizations/effective_combo_20261009_full_review_v2/index.html)。历史研究产品，不是已连接交易账户的实盘程序。

## 产品A：预声明等权条件组合 edge16

名称：`ensemble__edge16`。角色：保留透明线性读出与两个有界非线性读出的原始收益预测合作，作为当前弱信号combo研究基线。模型方向及强度来自成熟过去标签，不以未来正负翻方向。

输入：冻结106条OLD／既有控制坐标、48条OLD挖掘函数、48条混合挖掘函数，共202个市场数值列；另外12个静态币种身份one-hot。数值列顺序必须使用源[feature_contract.json](../outputs/effective_combo_20261009_round1/feature_contract.json)，不能按字典、因子分数或CSV显示顺序重排。规范化、NaN填充、截尾与函数计算版本沿用[数值池合同](../outputs/complex_alpha_prediction_estimation_20261009_v1/numeric_pool_contract.json)和冻结定义。

资产顺序：ADAUSDT、AVAXUSDT、BCHUSDT、BNBUSDT、BTCUSDT、DOGEUSDT、ETHUSDT、LINKUSDT、LTCUSDT、SOLUSDT、TRXUSDT、XRPUSDT。单位为每币未来4h固定份额简单价格收益减真实资金费率的完成价格计价代理，不是OLD误差，也不是风险残差预测。

三个成员：

| 成员 | 固定训练配置 |
|---|---|
| ridge_asset_online | λ10，214输入，Ridge训练内尺度；过去12月，月初更新 |
| tree15_online | LightGBM 4.6.0，160树、15叶、叶内≥256、学习率0.05、L2=10；过去12月，月初更新 |
| tree31_online | 同版本，240树、31叶、叶内≥128、学习率0.05、L2=10；过去12月，月初更新 |

树固定种子20261009、deterministic、force_col_wise，不按验证早停。训练标签用bps，固定±2000bps截尾；预测还原简单收益单位。训练截止规则为标签起点＋4h严格早于UTC月初，随后一个月系数／树固定。当前保存的是截至2026-01历史重放时的模型状态，不把它当成2026-10可直接使用的最新实盘模型。

组合预测：p=(p_Ridge+p_15+p_31)/3。权重不训练、不按验证换成员。目标暴露：

$$
q_j=0.5\operatorname{sign}(p_j)\tanh\!\left(\frac{\max(|p_j|-0.0016,0)}{0.0050}\right).
$$

q是单币在12币平均意义下的暴露单位，实际美元目标为q_j×当时组合资本/12。组合总绝对目标≤0.5，完成4h决策价格建立份额，随后完整5m价格变化入账，决策间固定份额。结算资金费使用结算前份额，按实际净变仓单向4bps收费，终点平仓也收费。没有加EMA、独立风险投影、动态触发校准或额外事后止损。

历史证据：2025-07—2026-01费后净财富+5.59225%，日收益Sharpe2.04460，最大回撤−3.93228%；2024H1／H2／2025H1三反馈折净财富+6.96615%、+0.62930%、+3.16446%；2025Q3／Q4分别+1.90949%、+1.85661%，2026年1月+1.72511%。3,011段自然完成暴露，12段终点删失，持有中位4h、90分位8h；平均绝对暴露3.83595%，有币持仓的时钟覆盖84.9079%。

信息与风险：验证IC0.0464256，11/12币时间序列IC正；R²相对零预测−0.84466%，绝对预测幅度未校准。10/12币精确净贡献正，最大正贡献币占正贡献总和18.3195%。主要是共同方向／择时，不能标纯Alpha。4／6／8bps费后净财富依次+5.59225%、+3.95932%、+2.35155%，12bps为负。

直接证据：

- [按时间组合预测](../outputs/effective_combo_20261009_round1/ensemble_prediction.npy)、[决策日期](../outputs/effective_combo_20261009_round1/dates.npy)、[真实目标](../outputs/effective_combo_20261009_round1/target.npy)。
- [完整5m账本](../outputs/effective_combo_20261009_round1/ensemble__edge16_validation_ledger.parquet)、[实际逐币仓位与价格路径](../outputs/effective_combo_20261009_round1/ensemble__edge16_validation_asset_path.npz)、[全部持仓段](../outputs/effective_combo_20261009_round1/ensemble__edge16_validation_trades.csv)。
- [月度更新记录](../outputs/effective_combo_20261009_round1/update_records.csv)、[源数据摘要](../outputs/effective_combo_20261009_round1/source_provenance.json)。
- 冻结模型分别在`outputs/effective_combo_20261009_round1/models/ridge_asset_online/`、`tree15_online/`、`tree31_online/`，按UTC更新时间命名；读取时选择当时已有状态，不选择未来日期文件。

## 产品B：15叶条件预测 edge16

名称：`tree15_online__edge16`。输入、标签、成熟规则、执行、预算、费用与A相同，预测只使用15叶月更新读出。它是拟合前声明的独立模型产品，不是事后从组合中删成员再归一化形成的新配置。

历史证据：验证净财富+7.17113%，Sharpe2.14911，最大回撤−4.07231%；三个发现折+8.15420%、+0.40861%、+2.58381%；2025Q3+2.27337%、Q4+2.99905%、2026年1月+1.73773%。七个月均正，3,831段自然完成、12段终点删失，中位4h、90分位8h。平均绝对暴露5.72063%，有效时钟覆盖94.1069%。

验证IC0.0462290，七个月均正，11/12币IC正，10/12币净贡献正；R²为−1.90421%，幅度校准仍不足。最大正贡献币占17.0381%。6／8bps净财富+4.78078%、+2.44360%，12bps−2.07598%。主要价值同样来自共同方向／条件择时。

直接证据：[预测](../outputs/effective_combo_20261009_round1/tree15_online_prediction.npy)、[5m账本](../outputs/effective_combo_20261009_round1/tree15_online__edge16_validation_ledger.parquet)、[实际逐币路径](../outputs/effective_combo_20261009_round1/tree15_online__edge16_validation_asset_path.npz)、[持仓段](../outputs/effective_combo_20261009_round1/tree15_online__edge16_validation_trades.csv)。所有模型在[完整模型对照](../visualizations/effective_combo_20261009_full_review_v2/models.html)可查。

## 产品C：冻结信息池与弱条件函数

交付物为106个已知坐标、OLD与混合各48个函数输入，以及逐条档案、数值矩阵和公式。它们是联合预测输入，不是96个已经证明独立费后盈利的Alpha，也不是所有202个输入独立有效。

函数定义：[feature_pool_freeze.json](../outputs/effective_combo_20261009_round1/feature_pool_freeze.json)。每条有源家族、模式、左右投影索引与权重、支持分钟、非线性阶数、复杂度节点、发现方向与core／conditional档案。原坐标索引必须对照R5的[field_registry.json](../outputs/complex_alpha_20261008_round5/field_registry.json)，不能拿214维新读出序号直接替换源公式索引。

观察产品：[202输入信息表](../outputs/effective_combo_20261009_signal_review_v1/individual_features.csv)、[逐币逐月预测表](../outputs/effective_combo_20261009_signal_review_v1/prediction_breadth.csv)、[家族与逐条定义页面](../visualizations/effective_combo_20261009_full_review_v2/factors.html)。信息表按现有方向显示，不根据验证重新翻方向或淘汰输入。

有效组合依赖：固定模型里条件／不稳定48个输入置零，组合净财富由5.59225%降至3.93397%；混合48输入置零为3.87694%；OLD106坐标置零为0.64553%。置零不是独立因子的盈利认证，也不是重训最优消融；它支持这些弱函数正在被当前组合有效使用。核心48函数置零略改善本期曲线，记录该负贡献，不据验证重构新池。

## 产品D：真实标签强拟合诊断

名称：`overfit_diagnostic`。255叶500树在全部成熟发现标签上训练并回放同一时期，IC0.974931、零预测R²89.0008%，固定edge8仓位、完整价格／资金费／费用账本。预测、模型及训练回放都在主生产输出中。

性质：高容量表达能力和同样本策略诊断。回放使用该时期标签训练模型，不能真实提前得到模型；夸张复利财富明确不作为可实现资金增长。其前向静态同配置验证仍亏损。它不会替代A和B的按时间历史认证。

直接证据：[指标](../outputs/effective_combo_20261009_round1/overfit_diagnostic_metrics.json)、[预测](../outputs/effective_combo_20261009_round1/overfit_diagnostic_prediction.npy)、[回放账本](../outputs/effective_combo_20261009_round1/overfit_diagnostic_in_sample_ledger.parquet)、[诊断页面](../visualizations/effective_combo_20261009_full_review_v2/overfit.html)。

## 统一边界与继续研究起点

源候选全发现选择，反馈折不独立；验证历史以前已见；本轮holdout不开。A、B是通过预声明历史规则的候选，不是独立未来稳定性证明。资金费有真实事件费率但缺结算mark，使用完成价格代理；没有真实盘口冲击、滑点或实盘订单观测。

自动发现第一Ridge edge8验证亏损，其冻结排序完整保留：[selection_freeze.json](../outputs/effective_combo_20261009_round1/selection_freeze.json)。A、B作为全部预声明产品中的历史存活基线展示，不回写成原来自动第一。不得根据当前验证再挑币种、删Ridge、重估成员权重或调阈值后伪称本轮盲选成功。

下一项研究应围绕一个具体问题：成熟过去预测幅度校准、自动选优能否避免薄信号、或有效方向在显式中性约束下能保留多少。当前先冻结A、B及其全部证据，作为真实可比较的起点；不再回到“没有产品所以只能解释失败”。
