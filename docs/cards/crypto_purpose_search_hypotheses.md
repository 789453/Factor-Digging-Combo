# 万级用途分轨机制假设卡

## 残差方向 alpha 候选

价格与主动流不一致可能代表短暂价格冲击、吸收或延迟反应；同向持续则可能代表信息延续。角色为价格/偏移 A、主动流 B、成交集中/冲击或非对称风险 C。`Sub` 是背离，`GatePos/GateNeg` 分别测试确认与吸收，`TsCorr` 的水平/变化表示响应结构。所有输入无量纲或先滚动 z-score；`Div` 分母加 1 处理极值。方向仅由 discovery 推断，不给“机构交易”因果解释。对照为已知时点价格动量、波动、成交量、流动性、NY 时钟基线和留一币市场 beta 残差标签。失败模式为短噪声、共同市场暴露未去净、单币驱动、换窗过拟合。

## 共同风险／beta 候选

市场同步下行风险可能由波动持续性、跳跃、成交集中和流动性收缩共同表征；24/7 纽约时段改变基线，但并无开收盘。候选先跨币聚合为一个市场状态，再预测等权市场未来下行半方差，仅按时间计样本。对照为当前市场下行平方收益和 NY 时钟。合格需要基线外增量及跨时间段稳定；仅与风险同向不证明风险溢价。失败模式为波动简单持续性重复、少数币驱动、局部高波动期支配。

## 混合候选

方向反应可能依赖共同风险状态：流价背离在流动性压力下改变方向延续或吸收的含义。A 为方向模块，B 为主动流/确认，C 为市场或币种风险状态。必须对残差方向和共同风险**分别**给出证据，排序使用较弱轴，不能把单一维度强者称混合代理。失败模式为两目标方向不稳定、条件门控无新增信息、伪重复市场样本。

## 边界

模板只声明经济角色、公式族、参数范围和最大复杂度；实际假设需要经历万级确定性生成、三阶段筛选和独立迁移。5m 原生版本另用已完成 bar 和已完成 15m as-of 特征，不复用 1h 标签。OI、funding、basis 在原始数据缺失时不得臆造。
# Liquidity concentration style proxy

The observable target is the mean `micro5_volume_hhi` of the next four *complete* 1h bars after a one-bar entry lag. Each hourly HHI uses twelve native 5m quote-volume shares, summed after squaring; missing or incomplete hours invalidate the target. Its natural bounds are 1/12 through 1. This target describes future within-hour volume concentration, not a return or a trade direction.

The six bounded template families test persistence/transition, order-flow fragmentation, price-impact coupling, New York session transfer, activity divergence, and simple controls. The predicted mechanism is that a burst concentrated into few 5m slices may persist or normalize depending on flow, price impact, trade size and session state. For an incremental claim, a candidate must improve validation prediction beyond known-at-decision volume HHI, trade HHI, volume shock, trade-count shock and New York clock flags, using discovery-fitted coefficients and identical validation rows. A positive IC alone is insufficient. Invalidation includes negative incremental fit in later years, unstable direction, or alias-level overlap with a baseline field.
