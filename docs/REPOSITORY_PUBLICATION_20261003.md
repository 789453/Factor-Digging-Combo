# GitHub 仓库版本与数据边界

2026-10-09更新：目标仍为`789453/Factor-Digging-Combo`，旧origin不改。新增当前源码、配置、测试、完整研究与经验文档，以及有效基线41页、复杂研究总览／R5／固定移除的发布快照；原始行情、模型矩阵、缓存和完整账本仍在本机outputs，不上传。`docs/evidence/effective_combo_20261009_round1`仅保存冻结来源的轻量指标与定义副本。发布提交基于远端当前main，不覆盖其他远端提交，不改工作区原有暂存状态。

新仓库 `789453/Factor-Digging-Combo` 从当前工作树建立独立的干净根提交，
不继承旧仓库可能包含大文件的历史；原仓库及其 `origin` 保留。
发布版本标记为 `v0.1.0`，研究性质为实验性基线，不代表实盘可用策略。

提交包含源码、YAML 配置、单元测试、研究文档，以及
[`visualizations/aligned_crypto_v9/index.html`](visualizations/aligned_crypto_v9/index.html)
离线可视化快照。`.gitignore` 排除原始行情、Parquet 研究账本、缓存、
`outputs/` 全量实验和其他二进制数据。可视化中的数字来自已完成的本机
`outputs/crypto_aligned_11000_native5_20260930_v9/`，Git 仓库只保留页面所需的
图表点，不能替代完整逐 5m 账本或原始数据。

真实数据配置中存在本机数据根目录路径；克隆后需要改成自己的数据位置、
为复跑指定新的输出目录，继续通过唯一入口
`python -m src.alpha_mvp.research_cli --config ...` 启动。
测试期已用于历史诊断，后续不能作为全新独立 OOS 反复调参。
