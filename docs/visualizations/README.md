# 冻结研究的离线可视化

[一体化万级因子研究总览](aligned_crypto_v9/index.html)是 2026-09-30 完成的
`crypto_aligned_11000_native5_20260930_v9` 实验的只读页面快照。包含 48 个 HTML：
搜索漏斗、入选因子、Alpha/Beta、风险与原生 5m、三段净值与执行、12 币逐段页面。
页面复用旧 factor combo 的样式和本地 `plotly.min.js`，克隆仓库后可离线打开。

原始 5m 账本、Parquet 缓存和大量候选 CSV 不在 Git 仓库中。页面已保存图表
所需的汇总点，但要复算数值，应使用本机完整实验目录和
[研究报告](../CRYPTO_ALIGNED_FACTOR_RESEARCH_20260930.md)描述的口径。
生成代码为 `src/alpha_mvp/research/aligned_visualization.py`；它只读取已完成实验，
输出到新目录，不改写原证据，也不参与因子选择或测试期调参。
