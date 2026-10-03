# 文档导航与交接顺序

本目录是项目的长期研究记忆。新窗口、智能体或研究者开始工作前，按以下顺序阅读：

1. 根目录 `AGENTS.md`：不可违反的工程与研究约束。
2. 根目录 `README.md`：当前项目定位、唯一入口和快速运行方式。
3. `PROJECT_STATUS_AND_ROADMAP_20260928.md`：阶段性结论、已冻结证据、已知局限和下一阶段优先级。
4. `MIDTERM_RESEARCH_PROGRAM_20260928.md`：最近聊天形成的定位纠偏、中期阶段任务、当前阶段停点；近期工作先读此文。
5. `CRYPTO_MINING_THEORY_V1_V3_20260928.md`：最新校正后的万级三轨挖掘、原生 5m、机制和数理设计三版推演。
6. `CRYPTO_PURPOSE_MINING_RESULTS_20260928.md` 与 `CRYPTO_PURPOSE_MINING_DASHBOARD_20260928.html`：本阶段五轨万级实证、冻结跨年迁移、可视化与用途边界；当前先看此处的结果。
7. `RESEARCH_REQUIREMENTS.md`：长期有效的研究需求、筛选口径、可视化验收与未来设计原则。
8. `ARCHITECTURE.md` 与 `RESEARCH_PROTOCOL.md`：模块边界、holdout 防火墙和实验协议。
9. `CRYPTO_MULTISCALE_SESSION_RUN_20260926.md`：较早 30,000 表达式实验的结果事实。
10. `CRYPTO_FINAL20_FACTOR_IMPLEMENTATION_GUIDE.md`：在其他项目复现较早最终 20 因子的公式与策略口径。
11. `CHANGELOG.md`：阶段性变更记录。
12. `CRYPTO_MECHANISM_RESEARCH_DESIGN_20260928.md`：机制表达、半方差目标和半结构化流程。
13. `CRYPTO_PHASE_A_INCREMENTAL_EVIDENCE_20260928.md`：早期 180 条机制探索、基线外信息和跨年迁移检查。
14. `CRYPTO_EXECUTION_AND_CLUSTER_RESEARCH_20260928.md`：较早的执行诊断和聚类可行性记录。
15. `FACTOR_COMBO_BASELINE_DESIGN_20260929.md` 与 `FACTOR_COMBO_BASELINE_RESULTS_20260929.md`：原生 5m 因子组合首轮冻结设计、预测/风险双输出与 2023–2025 历史复核；当前 combo 工作先读此处。
16. `FACTOR_COMBO_DYNAMIC_REDESIGN_20260929.md` 与 `FACTOR_COMBO_DYNAMIC_RESULTS_20260929.md`：动态因子出入、Alpha/Beta/Risk 分层交易账本、独立年度及逐币 HTML 页面；继续 combo 工作时接着阅读。
17. `FACTOR_COMBO_RESEARCH_STANDARD.md`：后续 factor combo 的挖掘来源合同、去冗余、动态组合、触发和报告验收标准。
18. `FACTOR_COMBO_SIGNAL_ENGINE_V2_DESIGN_20260929.md` 与 `FACTOR_COMBO_SIGNAL_ENGINE_V2_RESULTS_20260929.md`：多尺度 5m 快/中/慢执行、训练期交易频率校准和真实数据复核。
19. `FACTOR_MINING_AND_COMBO_RESEARCH_DIAGNOSIS_20260930.md` 与 `FACTOR_RESEARCH_REDESIGN_HANDOFF_20260930.md`：因子质量、期限迁移、组合尺度与 Alpha/Beta 用途的深度诊断，含冻结账本审计、有限因子复算、固定执行消融及下一轮研究交接。继续收益改进工作前先读；未改生产策略。
20. `ALIGNED_CRYPTO_RESEARCH_DESIGN_20260930.md` 与 `CRYPTO_ALIGNED_FACTOR_RESEARCH_20260930.md`：2023-01 至 2025-06 发现期、2025-07 至 2026-01 验证期、2026-02 起测试期的一体化万级因子研究设计与实证。后者是本轮统一收益合同、冻结选择、实际交易账本和弱基线结论的主报告。

## 当前权威对象

- 生产研究入口：`python -m src.alpha_mvp.research_cli --config ...`
- 最新一体化真实数据研究：`configs/research/crypto_aligned_11000_native5_20260930_v9.yaml`；完成输出 `outputs/crypto_aligned_11000_native5_20260930_v9/`，结论见 `CRYPTO_ALIGNED_FACTOR_RESEARCH_20260930.md`。该测试期已用于诊断复核，不能作为后续参数调优的全新独立测试集。
- 最新离线交互总览：`visualizations/aligned_crypto_v9/index.html`，48 个页面；Git 仓库保留可视化快照，原始大账本仍留在本机 `outputs/`。
- 最新目的化研究配置：`configs/research/crypto_residual_alpha_10000_v2_2023.yaml`、`crypto_common_risk_10000_v2_2023.yaml`、`crypto_hybrid_10000_v2_2023.yaml`、`crypto_native5_residual_alpha_10000_2023.yaml`、`crypto_style_hhi_10000_2023.yaml`（后四项同在 `configs/research/`）。
- 最新完成实验与结论：`CRYPTO_PURPOSE_MINING_RESULTS_20260928.md`；逐轨 `outputs/` 目录和冻结年份索引见报告。
- 最新综合可视化：`CRYPTO_PURPOSE_MINING_DASHBOARD_20260928.html`。
- 最新 factor combo 标准：`FACTOR_COMBO_RESEARCH_STANDARD.md`；动态触发设计：`FACTOR_COMBO_SIGNAL_ENGINE_V2_DESIGN_20260929.md`。新实验完成后，以其独立 `index.html`、逐年触发页、逐币页和账本为证据；旧 v7 保留在 `outputs/crypto_factor_combo_dynamic_20260929_v7/`。
- 较早的 30k signed-return 实验 `outputs/crypto_multiscale_sessions_30000_robust20_20260926` 仍是其原目标下的不可覆盖证据，不能替代当前目的化研究。

`outputs/` 下已完成目录是不可覆盖的研究证据。任何复跑、改口径或扩展都必须使用新的输出目录。
