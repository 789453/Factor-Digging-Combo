"""Read-only narrative and interactive delivery for completed R2--R4 research."""
from pathlib import Path
import json,html,shutil,os,re,hashlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.alpha_mvp.research.factor_combo_reporting import _CSS,_plot,_table


def md(frame):
    def cell(v):
        if isinstance(v,(float,np.floating)):return f'{v:.4f}'
        return str(v).replace('|','/').replace('\n',' ')
    lines=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    lines+=['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(lines)


def build():
    history=Path('outputs/effective_combo_mature_state_20261009_r4_history');oos=Path('outputs/effective_combo_mature_state_20261009_r4_oos')
    probes=Path('outputs/effective_combo_mature_state_20261009_r4_calibration_probes');qa=Path('outputs/effective_combo_mature_state_20261009_r4_frozen_review')
    delivery=Path('outputs/effective_combo_20261009_extended_delivery_review_v1')
    final=Path('visualizations/effective_combo_20261009_extended_final_review_v1')
    if final.exists():raise ValueError('new final visualization required')
    h=pd.read_csv(history/'portfolio_metrics.csv');v=h.loc[h.period=='validation'];o=pd.read_csv(oos/'portfolio_metrics.csv')
    merged=v.merge(o,on='id',suffixes=('_validation','_oos'))
    ids=['base_ensemble__edge16','base15__edge16','wide15__edge16','wide15_static__edge16','wide_cov__edge16','asset_shrunk__edge16',
         'pair_prediction__edge16','stack_asset__edge16','state_cal1__edge8','state_cal1__edge16','state_cal3__edge8','state_cal6__edge16','state_score1__edge16','state_score3_smooth__edge16']
    compare=merged.set_index('id').loc[ids].reset_index()[['id','net_return_pct_validation','sharpe_validation','max_drawdown_pct_validation','net_return_pct_oos','sharpe_oos','max_drawdown_pct_oos']]
    core=['state_cal1__edge8','state_cal1__edge16'];coins=[]
    for root,period in [(history,'validation'),(oos,'oos')]:
        a=pd.read_csv(root/'asset_contribution.csv');e=pd.read_csv(root/'holding_episodes.csv')
        for k in core:
            x=a.loc[a.id==k];positive=x.wealth_contribution_pct.clip(lower=0)
            coins.append(dict(id=k,period=period,positive_wealth_assets=int(x.wealth_contribution_pct.gt(0).sum()),
                maximum_positive_share=float(positive.max()/positive.sum()),**e.set_index('id').loc[k].to_dict()))
    breadth=pd.DataFrame(coins)
    feature=pd.read_csv(history/'expansion_archive.csv');retained=feature.loc[feature.reason=='retained']
    family=retained.groupby(['stream','research_tier','family']).size().reset_index(name='added_count')
    states=pd.read_csv(oos/'state_coefficients.csv');states=states.loc[states.window_months==1,['update','mature_labels','gamma','last_eligible_origin']]
    pm=pd.concat([pd.read_csv(history/'prediction_metrics.csv'),pd.read_csv(oos/'prediction_metrics.csv')]);pm=pm.loc[(pm.model.isin(['base15','pair_prediction','state_cal1','state_cal3']))&pm.period.isin(['validation','oos'])].copy();pm['r2_vs_zero_pct']=pm.r2_vs_zero*100;pm['rms_bps']=pm.prediction_rms*1e4
    costs=pd.concat([pd.read_csv(history/'cost_sensitivity.csv'),pd.read_csv(oos/'cost_sensitivity.csv')]);costs=costs.loc[costs.id.isin(core)]
    probe=pd.read_csv(probes/'frozen_calibration_probes.csv');probe=probe[['id','period','probe','net_return_pct','sharpe','max_drawdown_pct','mean_abs_position']]
    ci=pd.read_csv(delivery/'core_temporal_uncertainty.csv')
    months=pd.concat([pd.read_csv(history/'period_breadth.csv'),pd.read_csv(oos/'period_breadth.csv')]);months=months.loc[months.id.isin(core)]
    exposure=pd.concat([pd.read_csv(history/'exposure_decomposition.csv'),pd.read_csv(oos/'exposure_decomposition.csv')]);exposure=exposure.loc[exposure.id.isin(core)]
    ls=pd.read_csv(delivery/'core_long_short_decomposition.csv')
    report=r'''# 弱信号组合、扩池与成熟幅度校准：完整跨期研究报告

本报告承接R1有效基线，完整记录R2扩池、R3共享/单币组合和R4成熟状态校准。它交付可复现的信号、模型与策略，也保留原基线的2026失效、主状态假设失败以及收益集中问题。2026年是按日期向前预测的历史OOS复核，不称全新独立盲测。

## 1. 本轮真正得到什么

预先声明的一个月成熟幅度校准分支，两套固定经济映射在历史验证和2026扩展OOS都保有正净收益。edge8历史验证+3.3719%、日Sharpe2.6747、最大回撤1.0251%，OOS+2.4679%、2.1324、回撤0.8919%；edge16分别+2.0458%、2.3522、0.8341%和+2.3210%、2.6017、0.5976%。单边费用4bps，两个窗口各自零仓位启动并在终点付费清仓。没有通过删亏损币、放宽单因子盈利门槛、未来翻号或展示同样本255叶拟合来制造这个结果。

这两条是本次全量预声明分支里的观察存活产品，不能把一个月事后写成原先唯一主假设。R4主协议三个月仍失效；冻结的自动发现推荐一直是wide_ridge_edge8，未被改写成OOS冠军。冻结展示项三个月方向评分平滑也保留其失败。这里给产品卡是为了让可用证据可观察，不是根据OOS重新排序生产参数。

主要成功是把弱信号联合预测的幅度与经济映射重新匹配。保持原强度、只学正负方向的score分支仍亏；直接乘成熟校准系数的cal分支存活。核心池仍包含OLD基础、原公式、增加的旧档案公式和条件函数。新池与模型没有被清空，原失败也没有被覆盖。

收益主要来自2026年8—9月，因此不能称已证明所有月份和所有市场状态稳健。七日时间块重采样的OOS Sharpe区间跨零，说明独立有效性仍有不确定性。项目本期得到的是有交易量、费用余量、跨币广度和固定敏感性支持的历史存活基线，而非未来胜率保证。

## 2. 完整核心比较

以下按事前模型/研究阶段列示，未按OOS收益排序。收益为完整5m账本简单收益复利，Sharpe是UTC日净简单收益和的均值/总体标准差乘sqrt(365.25)，回撤是财富相对历史峰值。百分比列按百分数显示。

{COMPARE}

## 3. 为什么R1能有效，为什么不能只凭R1停止

R1改正了两层错位：预测池不再要求每条因子独立付费赚钱；联合读出不再只用线性系数。106个OLD坐标与OLD/混合各48个函数完整进入共享非线性读出，条件/不稳定函数有独立名额，15/31叶捕捉它们在价格、成交、波动和时点状态中的共同作用。过去12月成熟标签的月度更新再将它们映射到直接raw-return，而不是先预测一个误差、再未经说明当成收益预测。

其固定证据有边界：原条件48条输入置零，组合验证财富由5.5922%降至3.9340%；混合48条置零降至3.8769%，OLD106置零降至0.6455%。弱条件信息有当前联合用途，普通基础坐标同样重要。核心48条置零略改善；单条稳定排名不能代表当前联合价值。所有置零保持原模型和权重，并改变联合输入分布，不能作可加因果PnL归因。

R1的edge16比机械RMS映射少放大弱预测、少付费，因此同模型策略可以有更高Sharpe而收益幅度不大。组合/15叶平均绝对暴露仅3.84%/5.72%，低于目标上限0.5很多；12币分摊资本进一步降低组合波动。原身份one-hot置零几乎不改收益，共享市场信息的使用无需逐币独立搜参。不能反过来把共享有效解释成所有币都拥有同等可预测性。

R2真正加入2026后，原两条分别亏3.6932%和4.5947%，说明七个月高Sharpe不能替代下一时期证据。此后继续围绕有界、可检验的组合估计和成熟状态问题迭代，而不是再次增加万级公式或只写为什么不能成功。

## 4. 数据、时钟和OOS范围

资产固定为ADA/AVAX/BCH/BNB/BTC/DOGE/ETH/LINK/LTC/SOL/TRX/XRP USDT，整个研究没有按后续收益删币。因子由既有分钟/15m表示计算，信号与主目标4h，目标是完成决策价后固定份额简单价格收益减未来资金费价格代理；策略持仓每4h形成，承担之后完整5m变化。资金费先按旧份额结算，再形成新订单；5m间份额不变、风险暴露会漂移。自然持有的4h/8h统计与预测目标不同，不假装每次都严格四小时平仓。

发现期至2025-06，历史验证为2025-07至2026-01。OOS主收益窗口为2026-03-01至缓存最晚完成时钟2026-09-25 00:00 UTC，原始1m数据最后行为2026-09-24 23:59 UTC，已核验Parquet尾部时间统计。9月为部分月份，没有伪造9月30日。2026-02只作训练桥接，不混入3—9月收益表。

各期分别从零仓位启动、末尾平仓。验证和OOS财富不直接拼为连续实盘净值，2月模型学习也不意味着2月交易收益被计入。末日00:00终点结算会进入UTC日Sharpe计数；全年前后对照均使用同一账本实现。

R2历史阶段物理截断因子/价格数值时钟至验证终点；OOS阶段先匹配协议、源散列和代码散列，才加载之后的数据。标签必须origin+4h严格早于更新月初，等于边界也不算成熟。衍生品文件读取的字节范围与数值样本入模范围不同，不能声称文件中holdout字节完全未读；候选排序与数值筛选始终只用发现期允许列和发现数值。

## 5. 扩池：增加独立线索，保留条件层

保留原202市场输入，新增纳入96个旧档案公式，无新公式搜索：OLD流48、混合流48；每流核心24、条件/不稳定24。106基础+192公式=298个市场输入，加12个静态身份列为310列。原池顺序和初始归一化复现为完全相同数值，未为大池重新归一化原输入。

每流最多读取256个R5冻结候选，候选排序只读取发现字段；新增公式与旧/已新增公式的发现期绝对相关小于0.95。旧池即使相互高度相关也不倒过来删除。此规则针对公式，不宣称全部公式与106基础坐标都小于0.95。新增名额按家族/层级交替，相关别名和预算档案逐条保存；数量不足会明确失败，不静默补入回退。

{FAMILY}

核心/条件标签继承源研究目标，不能直接认证它在当前raw-return目标上稳健或盈利。当前逐输入IC独立保存，某函数边际IC弱、平均反号或者不能独立付费，均不导致其从预测池删除。新增96条提升了可供联合学习的信息覆盖，但并没有自动提高策略质量：同容量15叶验证由7.1711%降至6.3074%，Sharpe由2.1491降至1.7954，OOS仍退化。

这个对照明确纠正“数量多就更有效”的误解。新公式仍是有效研究资产；可以保留更多弱信息，同时让有界读出、先验组合和幅度校准决定其经济使用程度。不能因为大池整体不胜，就按OOS逐条删成完美小池。

## 6. 公平模型与币种适配

同池同目标对照Ridge(lambda10)、15叶160树/最小叶256、31叶240树/最小叶128，学习率0.05、L2=10、训练标签截断±2000bps、过去12月、月更。共享模型同容量的小池/大池比较不混入改阈值或改费用。大池15叶另有2025-07冻结的静态版本，OOS继续使用该模型而非2026重训。

每币额外7叶120树、最小叶128、L2=10的直接预测器，和共享15叶固定25%/75%混合；币种样本一年仅约2190个4h标签，不能把全市场2万余标签的复杂度原封不动视为单币安全容量。新适配保留所有币，未为币种挑窗口/阈值。

实证不支持把“特异适配”当成默认升级：该分支验证5.4179%、Sharpe1.7711，OOS−4.3372%。R3把同一四成员的权重向共享/先验收缩，单币局部权重再与全局各半，验证Sharpe2.3131但OOS仍弱。真正应保留的是可测、受约束的适配接口和全部失败，而非假称多一个币种模型就已经实现特异Alpha。

## 7. 为什么误差协方差权重没有解决问题

R2以过去成熟的三成员误差协方差做非负和为1的最小方差，50%向对角收缩，最优权重再50%向等权收缩。误差e_j=p_j−y，各列共有−y；目标方差远大于弱预测方差。和为1时目标共同方差原本是常数，而对整个误差矩阵作对角收缩，会额外引入很大的人工分散惩罚。因此实际权重接近1/3：Ridge平均33.84%、15叶33.31%、31叶32.85%。

这并非三模型都有同样信息，也不是权重估计充分有效。误差彼此高度相关可能主要说明真实收益噪声共有，而不能直接用于删掉信息。R2收缩权重和等权的验证结果近似，没有大幅升级；同样不借此抛弃普通线性成员的诊断价值。

## 8. R3：在预测尺度上学习，向已有解法收缩

R3直接估计A=E[pp']、b=E[py]，A在预测尺度上25%向对角收缩，求非负、和为1的w'Aw−2b'w，再75%向[0.40,0.35,0.20,0.05]先验收缩。成员是原15叶、原等权组合、扩池静态15叶、扩池单币收缩模型。弱标签的共同方差不再进入正则的尺度，已有有效共享产品的先验占75%，估计噪声不能轻易改写整个组合。

另保留原15叶与扩池在线15叶75%/25%、在线/静态各半预测、两成员先映射仓位再平均三种小型对照。后两者因tanh/阈值非线性不同；仓位平均在净变仓上只收一次费用，不把成员各自费用直接相加。

在线/静态预测各半edge16历史验证6.4278%、Sharpe2.5560、回撤3.7709%，七个月全部正，比原组合更合理地兼顾费用与模型差异；OOS仍−0.6840%。这一步改善了已见历史，是有用的组合进展，但不是2026退化已经解决的证明。R4取的正是这条事先固定预测，不根据OOS选一个基线作方便的误差投影。

## 9. R4：成熟幅度校准的可复现公式

令p_it为在线15叶与静态扩池15叶各半的直接raw-return预测。每月m，取过去一个月满足t+4h<m、预测/标签有限且|p_it|>16bps的样本集合M_m。固定最小成熟样本300，足够时：

$$
\gamma_m=\operatorname{clip}_{[-1,1]}\left(\frac{\sum_{(i,t)\in M_m}p_{it}\,\operatorname{clip}(y_{it},-0.2,0.2)}{\sum_{(i,t)\in M_m}p_{it}^{2}}\right),\qquad \widehat\mu_{it}=\gamma_m p_{it}.
$$

样本不足时gamma=1，是事前声明的原基线暖启动，不能叫“估计仍充分可信”。阈值样本只用于组合校准，不是给因子池重新加交易盈利门禁。训练y截断只用于拟合，评价标签和账本不截断。所有币共用系数，系数可以为负，但只能由更新时点之前已成熟的预测标签产生。

固定执行为：

$$
q_{it}=0.5\,\operatorname{sign}(\widehat\mu_{it})\tanh\left(\frac{\max(|\widehat\mu_{it}|-\theta,0)}{0.005}\right),\qquad\theta\in\{0.0008,0.0016\}.
$$

这里预测是原始收益单位，避免按自身小RMS放大噪声。阈值是固定经济映射，并不保证每次交易实际盈利。另行score=sign(gamma)*p，|gamma|<0.05为0，它保持原强度，明确只叫方向评分；不能把它当校准后的条件期望收益。

2026一个月系数与成熟截止完整如下。8月245个样本不足300回到1，9月样本足够但估计触及上界1，两者性质不同。末次可用origin例如2026-06-30 16:00+4h=20:00，严格早于7月1日更新边界。

{STATES}

## 10. 为什么这次存活，不应归因于未来翻号

原pair预测edge16 OOS−0.6840%；一个月score edge16−1.2510%；一个月cal edge16+2.3210%。score和cal使用相同的成熟状态信息，后者按估计幅度缩放、再进入固定阈值。它在低经济强度的时点更容易保持零或小仓位，避免只要模型预测非零就把不确定信号转成有费用的仓位。

普通预测的强度、弱信号的条件信息、尾部样本的成熟校准和执行成本是共同解法，不能只概括成“翻转了一下所以盈利”。固定负gamma月份置零，edge16 OOS仍+2.3851%，略高于原+2.3210%；收益并不依赖负向状态交易。旧池/静态模型仍提供可用信号，校准控制了哪些强度值得经济使用。

也不能声称一月永远胜过三月。三月cal edge8 OOS−1.2303%，score和平滑分支更差，表明状态记忆长短敏感；一月是事先声明的敏感性分支的实际存活，不是把主协议偷偷改成一月。研究问题走通，参数普适性仍需下一真正未见时期。

## 11. 预测表现与交易效用必须分别判断

{PREDICTION}

一个月cal历史验证IC0.0463、相对零预测R²+0.1841%，OOS IC0.0331、R²−0.0198%，并未证明对所有4h收益的均方误差稳健胜过零。OOS大量原始收益仍不可预测，信号幅度也很小。但固定映射只使用部分超过阈值的预测，完整经济账本有实际交易和净收益。不能因为总体R²略负就把信息池或交易证据清空，也不能因策略盈利就谎称全样本预测损失已解决。

边际IC的低强度与联合使用并不矛盾；需要看条件读出、预测幅度、稳定性与经济目标，而不是要求每条输入同时过全样本稳定符号、独立净收益和独立对冲腿盈利三道门。

## 12. 收益不大而Sharpe较高的具体原因

一个月cal edge16 OOS平均绝对暴露0.5226%，edge8为1.2398%；目标上限50%不等于长期50%投入。幅度校准改变了交易集合和经济强度，而非仅把原仓位整体除以常数。被低强度预测筛掉的月份几乎平仓，剩余区间由多币共享信号捕捉，日均收益/波动之比可以较高，绝对净收益仍只有几个百分点。

降低暴露不是自动提升Sharpe；它主要减少实际付费的噪声交易并改变尾部分布。单纯加杠杆既不能补出新信息，也会放大同向市场和滑点风险。本轮不以加杠杆追高绝对收益。固定gamma绝对上限改为0.5的只读探针，edge16 OOS仍+0.5908%、Sharpe2.4903，但暴露更低，这体现的是低幅度经济使用，而非一个新选中的主策略。

## 13. 持仓、完成频率、币种广度

{BREADTH}

活跃5m比例定义为任一币实际绝对暴露>1e−5，不等于全组合持续满仓。自然持有是连续有符号暴露片段，末尾强制平仓右删失；不是交易所逐笔lot、胜率或每一笔完整可分配PnL。两条OOS分别1786/614个自然结束片段，右删失5/1，中位4h、90分位8h，足以排除空仓微正的解释。

币种财富贡献按当期真实组合资本加权，可加到组合总收益；每币net分量的诊断复利不等于重新模拟的独立资金账户。edge16 OOS财富贡献11/12正，而独立分量复利/算术口径为10/12正，ADA的细小差异由资金路径加权产生。不能混用这两个定义夸大广度。

最大正贡献占比edge16 OOS约22.03%，edge8约24.78%，没有删ADA/TRX或把亏损腿单独去掉。主要用途仍是共同方向和条件择时，非12个独立Alpha同时成立。

## 14. 月、季度与时间集中

{MONTHS}

edge16 OOS四个月为正、两个月零、一个月负；Q2仅微正0.0220%，Q3部分季度+2.2984%。edge8主要8/9月正，3/5月零，4/6/7月负，Q2−0.1875%。不能用总Sharpe掩盖这一区别。两条验证期的两个完整季度和1月均正，两个发现反馈折正；一月cal并非三个反馈折全正。

2026年收益集中8—9月，对市场状态、暖启动和短期尾部有依赖。高Sharpe来自这段实际风险收益形状，不意味着已经消除月份集中。这是当前产品最应保留的限制，不把它升级成新的强硬单因子淘汰门槛。

## 15. 费用与真实变仓余量

{COSTS}

每项费用测试重放真实份额/现金，而非从原收益直接减常数；费用改变资产金额及后续实际暴露，微小价格/资金费差异是这个路径的正常结果。0/4/8bps都保持相同预测、阈值、预算及终点清仓。两条OOS 8bps仍+1.4833%/+1.8992%，后者Sharpe2.2169。

成本盈亏平衡14.15/26.28bps是价格+资金费算术和除以变仓的诊断，不能等同于已精确验证到该费用的复利策略。价差、滑点、冲击和真实mark资金费仍未包含；结算使用完成价格代理。这里只交当前成本合同下的可复现历史策略。

## 16. 共同方向、相对部分与长短腿

{EXPOSURE}

{LONGSHORT}

edge16 OOS共同价格财富贡献+2.2448个百分点，相对部分+0.5036，资金费−0.0116、费用−0.4159，合成净财富+2.3210%。相对部分是等权价格分解的剩余，不是重新对冲后的纯Alpha回测。暴露平均接近零也不能自动叫市场中性。

长腿价格贡献+2.9383、短腿−0.1898：signed long-short产品有亏损的短侧，整体仍可有效；这不是单独对冲腿必须赚钱的理由。本轮不据OOS剪掉短腿。若研究纯残差产品，应明确另一个目标、风险暴露与全部对冲费用，不能把本报告的raw-return换标签。

## 17. 固定暖启动、方向和幅度探针

{PROBES}

探针不重训模型、校准系数或未来月度状态，也不改阈值/预算；它们仅回答当前冻结产品对某模块的依赖，不是新增挑选的交易策略。完整重放误差在1e−5个百分点/指标阈值内。

8月暖启动是主要依赖：把样本不足月份预测置零，edge8 OOS由2.4679%降至0.4644%，edge16由2.3210%降至0.7484%，Sharpe降至0.8452/1.8134。收益仍正，说明不是全靠这一暖启动月，但不能因此忽略其贡献。把负gamma月份置零不伤本期收益，说明保持原预测强度的方向翻转不是核心成功；gamma绝对上限减半也仍正，表明有一定幅度敏感性余量。

这些信息应该进入产品卡和下一实验的问题，而不应偷偷把暖启动改成空仓、抹掉负系数月份或另挑上限再宣布“原策略”更好。

## 18. 时间块不确定性

{UNCERTAINTY}

采用UTC日净收益和，1000次、7日循环共同时间块重采样，保留策略之间同一日和块内共同波动；区间为5%/95%分位，positive_mean_fraction是重采样正均值比例，不是未来赚钱概率。没有重新搜索/训练、没有多重试验校正，且依赖条件历史平稳近似。

OOS edge8 Sharpe区间−1.4223至3.9947，edge16−0.1675至4.2269。当前点估计、费用和固定探针支持历史存活；时间集中使严格统计独立性仍未确认。不能为了形式上的显著性，反过来删除全部弱输入、把目前真实经济用途说成不存在；也不能把点估计2.60直接写成未来参数。

## 19. 样本内过拟合的正确位置

R1保留255叶同样本强拟合IC0.9749、R²89.0%以及夸张诊断财富，说明模型表达能力足以拟合已有标签；同容量按时间外推仍亏。用户允许一定样本内过拟合，因此本轮不把所有高容量或不稳定输入排除，但仍把它们标成诊断产物。

本次存活分支没有使用255叶同样本预测，也没有把已发生收益当当前状态特征。它消费的是原先已按时间产生的冻结预测；每月gamma只用已成熟标签。发现反馈OOF仍受上游全发现候选搜索影响，不能冒充独立OOF；2026更不能因为后续研究出现正分支就重新标成未见盲测。

## 20. 经验沉淀：以后按什么顺序解决

第一，先保留有信息的弱/条件池并生成可观察直接预测，不把单条收费回测当研究入池门禁。第二，让有界联合读出学习状态，原普通坐标与复杂公式并存，不用结构维数代替实证。第三，在统一目标、时钟和执行下比较数量、容量、更新、权重与特异适配，每次保留原有效锚。

第四，权重必须匹配要优化的量。弱预测的误差协方差包含巨大共同目标项，收缩尺度应拆清楚；组合先向已有解法收缩，少量过去标签不能轻易重写全部权重。第五，校准预测的幅度，再考虑经济阈值。方向正确、边际IC为正和有条件净效用是不同问题。

第六，状态只能从成熟过去估计，并保存每次截止与样本量。暖启动、截断和小样本回退都是模型的一部分，要事前声明、检查依赖。第七，高Sharpe必须与真实暴露、自然持有、月份/币种、费用和时间块不确定性一起交付，六个月近空仓的微正不是有效解法。

第八，主假设失败和其他预声明分支存活可以同时成立。保留全部预测和冻结推荐，不能为了看起来顺畅把三月改写成一月主协议。第九，不用亏损对冲腿否决整个signed产品，不把共同方向盈利冒充残差Alpha。第十，研究完成应交使用卡、完整账本与交互证据；经验不能只剩“失败原因”或工程包装。

## 21. 当前验收与下一边界

R1有效基线已保留并准确复现；原202/扩310列公平对照、共享/单币模型、两类权重、预测/仓位组合、成熟状态敏感性均完成。一个月cal两套在已见验证期与扩展历史OOS均存活，拥有自然持有、币种贡献、8bps余量和固定暖启动/幅度探针。当前已经有可以复查和复现的有效组合产品，不以全亏损解释结束。

未解决的是一月窗口和暖启动在下一真正未见市场的普适性、时间集中与精确成交/mark资金费。它们不被包装成已解决，也不作为阻断本期弱信号产品的新增苛刻门禁。本轮冻结全部参数和证据，下一次必须以新的未见数据或事前单一问题开展，不按本次OOS追调最佳曲线。

## 22. 交付、入口与复现

首先按用户要求将原当前状态推到GitHub：789453/Factor-Digging-Combo，提交e8d2fd699b4ce9a6a5eca82e382eb03a4388a726，经远端main核验后才开始R2研究。原origin和本地普通索引保留；发布用独立索引构建当前工作区快照。

唯一生产入口仍为`python -m src.alpha_mvp.research_cli --config ...`。R2为effective_combo_extension，R3/R4为同一入口的effective_combo_stacking复用阶段，没有新挖掘流水线。六个真实历史/OOS实验输出不可覆盖；源码/配置更改后的复现应使用相应source_snapshot版本和新的输出路径，不绕过冻结校验。逐月模型和状态、原始预测、账本、实际份额路径都在原outputs。

最新173单测通过，既有一个空切片警告；R2/R3/R4均通过模拟生产CLI及有界十二币真实实验。R2原15叶/等权基线复现最大指标误差约9.9e−8；R3/R4全源产品重放通过。只读R4复核确认394个原证据文件未变，历史/OOS的协议和推荐一致；两个53页报告各577本地链接零缺失。额外固定校准探针与最终展示复核独立保存。

主交互入口：[完整扩展总览](../visualizations/effective_combo_20261009_extended_final_review_v1/index.html)，含全部OOS模型、192条公式卡、逐币独立坐标、验证总览链接、费用、状态、统计与固定探针。原R1[41页报告](../visualizations/effective_combo_20261009_full_review_v2/index.html)、R2/R3/R4所有阶段和原五轮报告均保留。轻量公开证据副本在docs/evidence/effective_combo_extension_20261009，SHA对应本机不可覆盖原输出；数值矩阵、大账本和原始市场数据不进入Git仓库。

配置：[R2历史](../configs/research/effective_combo_expansion_20261009_r2_history.yaml)、[R2 OOS](../configs/research/effective_combo_expansion_20261009_r2_oos.yaml)、[R3历史](../configs/research/effective_combo_stacking_20261009_r3_history.yaml)、[R3 OOS](../configs/research/effective_combo_stacking_20261009_r3_oos.yaml)、[R4历史](../configs/research/effective_combo_mature_state_20261009_r4_history.yaml)、[R4 OOS](../configs/research/effective_combo_mature_state_20261009_r4_oos.yaml)。产品使用定义见[新产品卡](EFFECTIVE_COMBO_EXTENDED_PRODUCT_CARDS_20261009.md)，过程见[迭代日志](EFFECTIVE_COMBO_ITERATION_LOG_20261009.md)，初始原因沉淀见[经验页](EFFECTIVE_BASELINE_LESSONS_AND_NEXT_QUESTIONS_20261009.md)。

## 23. 全部预声明执行产品，不筛掉失败

{ALL}
'''
    replacements={'COMPARE':compare,'FAMILY':family,'STATES':states,'PREDICTION':pm[['model','period','ic','r2_vs_zero_pct','rms_bps','observations']],
        'BREADTH':breadth,'MONTHS':months[['id','frequency','segment','net_return_pct','daily_sharpe']],
        'COSTS':costs[['id','period','cost_bps','net_return_pct','sharpe','max_drawdown_pct','turnover','fee_pct']],
        'EXPOSURE':exposure,'LONGSHORT':ls,'PROBES':probe,'UNCERTAINTY':ci,'ALL':merged[['id','net_return_pct_validation','sharpe_validation','net_return_pct_oos','sharpe_oos','mean_abs_position_oos']]}
    for key,frame in replacements.items():report=report.replace('{'+key+'}',md(frame))
    Path('docs/EFFECTIVE_COMBO_EXTENDED_RESEARCH_RESULTS_20261009.md').write_text(report,encoding='utf-8')
    cards='''# 扩展组合产品卡：明确使用、事实与边界\n\n两条一个月cal映射是预声明分支的历史存活产品，不覆盖三个月主协议失败、自动发现推荐wide_ridge_edge8或原展示项。核心是冻结在线/静态预测各半、过去一个月成熟标签幅度校准、固定经济阈值；非纯残差Alpha、非独立未来证明。\n\n'''+md(compare.loc[compare.id.isin(core)])+r'''

## 预测/组合合同

- 106基础坐标与192冻结公式，共298市场列，加12身份列；新增96公式来自旧R5档案、核心/条件各半、发现相关去重，无新大搜索。
- 原池在线15叶与扩池静态15叶各半形成p。共享结构保留原有效信息与更新差异，不为每币单独挑配置。
- 月初仅用过去1月origin+4h严格早于当前边界且|p|>16bps的有限样本，估计gamma=sum(p*clipped_y)/sum(p²)，截断[-1,1]。300样本前gamma=1是声明的暖启动。
- 直接预测mu=gamma*p；它与只保持强度的方向评分不同。训练标签±2000bps，评价不截断；方向只从成熟过去得到。
- q=0.5*sign(mu)*tanh(max(|mu|-theta,0)/0.005)，theta=8或16bps；4h更新，5m实际份额/现金账本，资金费付旧份额，单边4bps，终点收费平仓，5m净简单收益复利。

## 活跃度和广度

'''+md(breadth)+r'''

任一币有实际暴露是活跃，平均绝对暴露才表示组合资本使用程度。自然持有为连续方向暴露，不等于逐笔lot；末尾右删失。每币分量诊断复利与组合资本财富贡献定义不同，不混用正币数。

## 可保留效果

两条OOS8bps仍+1.4833%/+1.8992%，edge16 Sharpe2.2169；费用余量不是未实测26bps的承诺。暖启动月份固定置零，OOS仍+0.4644%/+0.7484%，效果变弱；负gamma置零不伤本期收益，上限0.5固定探针仍正，均不升级为新选中策略。

## 必须带着使用的限制

2026年收益主要8—9月，3/5月平仓；一月暖启动8月样本245<300回原预测，9月系数足够样本但触及1上界。三个月主协议亏损，六个月edge16 OOS只有0.0593%而暴露0.0056%，不能因其Sharpe2.55把它认证为有用策略。长腿OOS贡献主要正、短腿负，不删亏腿，也不叫市场中性。

历史OOS已被研究查看，7日块重采样区间跨零、没有选择校正；当前是有实际交易的历史存活，不是所有月份/下一未来稳健保证。零预测基线R²在OOS仍略负，经济选择效用与全样本预测精度分开。数据只到原始2026-09-24末分钟/完成9月25日零点；9月部分，不代表月底完整覆盖。

## 使用与复现

保持完整12币、原始单位、阈值、费率、暖启动和成熟截止；不要为了复制表面Sharpe删币、改变强度、提前取标签、把score当mu或者用OOS重新选窗。生产配置R4历史/OOS保留全部48执行产品，本卡两条存活分支的输出前缀是state_cal1__edge8/state_cal1__edge16，不另建第二流水线。

配置、source_snapshot、state_coefficients、逐时点预测、5m ledger和asset_path、trades、cost_sensitivity、asset_contribution、period_breadth均可追溯。完整报告/交互总览说明所有对照及边界；原R1与主协议失败不改写。
'''
    Path('docs/EFFECTIVE_COMBO_EXTENDED_PRODUCT_CARDS_20261009.md').write_text(cards,encoding='utf-8')
    shutil.copytree(Path('visualizations/effective_combo_mature_state_20261009_r4_oos'),final)
    (final/'index.html').rename(final/'oos_complete.html')
    # New display layer preserves source report/data; clarify per-asset diagnostics.
    for p in final.rglob('*.html'):
        text=p.read_text(encoding='utf-8').replace('R3 有效基线、共享收缩权重与','R4 成熟幅度、共享组合与')
        text=text.replace('独立币账本与组合财富贡献','币种分量诊断复利；真实组合贡献见表')
        if p.parent.name=='assets':text=text.replace('<main>','<main><p>币种分量净收益按组合资本归一化；图示分量诊断复利，不是重模拟独立币账户。预测/方向评分/价格/实际持仓量纲分开。</p>',1)
        p.write_text(text,encoding='utf-8')
    nav=[('index.html','研究总览'),('oos_complete.html','全部OOS'),('comparison.html','全模型比较'),('factors.html','完整公式卡'),('states.html','成熟状态'),
         ('qualifications.html','广度与不确定性'),('calibration_probes.html','固定校准探针'),('assets.html','十二币路径'),('costs.html','费用与暴露'),('lineage.html','研究迭代')]
    table=lambda x:_table(x,rows=len(x))
    def page(name,title,body):
        n=' · '.join(f'<a href="{f}">{label}</a>' for f,label in nav)
        text=f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{_CSS}</style><script src="plotly.min.js"></script></head><body><nav>{n}</nav><main><h1>{html.escape(title)}</h1><p>固定12币、4h预测/决策、5m份额账本、单边4bps、简单收益复利、UTC日Sharpe。历史OOS复核，非新盲测；末时钟2026-09-25 00:00 UTC，9月部分覆盖。</p>{body}</main></body></html>'
        (final/name).write_text(text,encoding='utf-8')
    fig=make_subplots(rows=1,cols=2,subplot_titles=['2025-07至2026-01验证','2026-03至9月数据终点'])
    for col,(root,period) in enumerate([(history,'validation'),(oos,'oos')],1):
        for key in ['base15__edge16','pair_prediction__edge16','state_cal1__edge8','state_cal1__edge16','state_cal3__edge8']:
            f=pd.read_parquet(root/(key+'_'+period+'_ledger.parquet'));dt=pd.to_datetime(f.completed_5m,format='%Y%m%d%H%M');ii=np.unique(np.r_[np.arange(0,len(f),12),len(f)-1])
            fig.add_trace(go.Scatter(x=dt.iloc[ii],y=np.cumprod(1+f.net.to_numpy())[ii]*100,name=key,legendgroup=key,showlegend=col==1),row=1,col=col)
    fig.update_layout(height=470)
    body='<p>全部预声明产品完整保留。一个月cal分支两套映射跨段存活；三个月主假设失败，自动发现推荐和原展示均不改。图按小时抽点，财富用完整5m账本计算。</p>'+_plot(fig,'保留原基线、历史升级与状态校准',height=470)
    body+=table(compare)+'<h2>成熟幅度的效果，不是只翻方向</h2><p>一个月score edge16 OOS −1.2510%，cal edge16 +2.3210%。低幅度预测经固定经济映射减少付费噪声；两条都有实际持有，8bps仍正。收益主要8—9月，暖启动依赖和跨零时间块区间如实展示。</p>'
    body+='<p><a href="../effective_combo_mature_state_20261009_r4_history/index.html">53页完整历史验证</a> · <a href="../../docs/EFFECTIVE_COMBO_EXTENDED_RESEARCH_RESULTS_20261009.md">完整研究报告</a> · <a href="../../docs/EFFECTIVE_COMBO_EXTENDED_PRODUCT_CARDS_20261009.md">产品使用卡</a></p>'
    page('index.html','弱信号、多因子组合与成熟幅度：完整扩展总览',body)
    page('qualifications.html','实际持有、月份/币种和时间块不确定性',table(breadth)+table(months)+table(ci)+'<p>七日循环块、1000次，5%/95%分位；重采样正均值比例不是未来赚钱概率。2026收益集中8—9月，区间跨零，未作选择校正。广度与费用支持历史存活，不保证独立未来。</p>')
    page('calibration_probes.html','固定系数探针：暖启动、负向状态与幅度',table(probe)+'<p>固定模型/原系数和未来月度状态，不重训、不改阈值预算、不用探针另选策略。暖启动移除后效果下降但仍正；负gamma移除不伤本期收益。不存在可加PnL因果归因。</p>')
    page('lineage.html','从有效基线到跨期解法：迭代和证据', '<p>先GitHub同步e8d2fd6，再R2冻结扩池和OOS；R3预测尺度收缩组合改进历史，OOS仍弱；R4预声明1/3/6月成熟幅度/方向对照，一个月cal存活，三月主假设失败。</p>'+table(family)+
        '<p><a href="../effective_combo_20261009_full_review_v2/index.html">R1完整报告</a> · <a href="../effective_combo_expansion_20261009_r2_history/index.html">R2历史</a> · <a href="../effective_combo_expansion_20261009_r2_oos/index.html">R2 OOS</a> · <a href="../effective_combo_stacking_20261009_r3_history/index.html">R3历史</a> · <a href="../effective_combo_stacking_20261009_r3_oos/index.html">R3 OOS</a></p><p>新增96公式来源、全部失败、原始数值和代码/源散列均保留。完整Python生产入口与本机outputs是权威证据，Git公开轻量副本不代替原账本。</p>')
    pages=sorted(str(p.relative_to(final)).replace('\\','/') for p in final.rglob('*.html'))
    missing=[];links=0
    for name in pages:
        p=final/name
        for href in re.findall(r'(?:href|src)=[\"\x27]([^\"\x27]+)',p.read_text(encoding='utf-8')):
            if href.startswith(('http:','https:','data:','#')):continue
            links+=1
            if not (p.parent/href.split('#')[0]).exists():missing.append(name+' -> '+href)
    if missing:raise ValueError('missing final links '+str(missing[:5]))
    meta={'pages':pages,'page_count':len(pages),'full_rows':True,'automatic_selection':'wide_ridge__edge8','production_showcase':'state_score3_smooth__edge16',
        'observed_predeclared_survivors':core,'original_recommendation_unchanged':True,'role':'new read-only display layer preserving original completed reports',
        'sources':[str(history),str(oos),str(qa),str(probes)],'local_links':links,'missing_links':missing}
    (final/'visualization_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    (delivery/'delivery_qa.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    (delivery/'manifest.json').write_text(json.dumps({'status':'COMPLETED','role':'read-only core uncertainty/decomposition and final display QA','page_count':len(pages),'missing_links':missing},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'pages':len(pages),'links':links,'report_bytes':Path('docs/EFFECTIVE_COMBO_EXTENDED_RESEARCH_RESULTS_20261009.md').stat().st_size},ensure_ascii=False))


if __name__=='__main__':build()
