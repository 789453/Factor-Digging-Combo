# 字段知识卡：`cluster_stress_affinity`

- **角色**：流—跳跃—流动性联合压力状态的连续代理，可用于下行风险门控；不是机构交易或纯 alpha 的直接观测。
- **版本**：`2026-09-28-cluster-stress-affinity-v1`，实现在 `src/alpha_mvp/research/cluster_state.py`，模型参数单独保存在每次实验的 `cluster_state_model.json`。
- **依赖**：`micro5_taker_imbalance`、`micro5_jump_share`、`session_illiquidity_surprise`、`trade_date`、`ts_code`。前三项必须为当前已完成 1h bar 可得的值；后两项确定时间和币种面板。
- **公式**：对每个币种和三个输入分别做因果 96 bar 滚动 z-score，裁剪至 [-8,8]。仅取 discovery 中每 4 个 bar 的完整向量，按固定等距子样本上限拟合 K=2 的欧氏 k-means；以中心 `-flow + jump + illiquidity` 最大者标记压力中心。输出 `exp(-||z-c_stress||²/(2τ²))`，其中 `τ` 为发现期压力簇内距离中位数（下限 0.25）。配置可更改窗口、步长、上限和 K，均写入实验签名。
- **数值语义**：无量纲，有限值属于 [0,1]；越接近 1 表示越靠近发现期压力中心。它不是簇号或概率。任一输入缺失、滚动历史不足时为 NaN；极端有限输入经 z-score 后裁剪，输出保持有限。发现期完整样本不足、空簇或簇占比低于门槛时明确失败。
- **训练边界**：滚动变换只使用当前及历史 bar；中心、压力簇标签及带宽只用 discovery 拟合，validation/holdout 只应用冻结参数。候选评分不得读取 holdout。
- **单位/检验**：单变量无量纲标准化后组合；测试覆盖缺失、极端值、确定性、发现期拟合边界及 holdout 改动不改变中心。新数据集需检查簇占比和跨币覆盖后再解释状态含义。
