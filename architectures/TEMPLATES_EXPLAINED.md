# Alpha MVP Phase 2 数据结构与模板说明

## 1. 核心数据结构

### 面板数据 (Panels)
项目使用矩阵化的面板数据进行高效运算。
- **结构**: `(T, N)` 矩阵，其中 `T` 是交易日数量，`N` 是股票数量。
- **存储**: 在内存中以 `numpy.ndarray (float64)` 存储。
- **优势**: 
    - 避开了 Pandas 索引对齐的巨大开销。
    - 完美适配 Numba 的并行矩阵运算。
    - 极大地降低了内存碎片和 IO 传输损耗。

### 结果数据库 (DuckDB)
所有的中间过程和结果都存储在 `phase2_results.duckdb` 中。
- `runs`: 记录每一次挖掘运行的配置、Signature 和状态。
- `expression_catalog`: 存储所有生成的唯一表达式及其元数据（AST 结构）。
- `expression_jobs`: 任务队列，记录哪些表达式已计算，哪些待计算（支持断点续传）。
- `factor_results`: 详细的因子表现指标，包括 IC、IR、稳定性、覆盖率等。

---

## 2. 挖掘模板 (Templates)

当前挖掘引擎支持以下 6 类核心模板，旨在覆盖不同的 Alpha 逻辑：

| 模板名称 | 家族 | 阶数 | 逻辑说明 | 复杂度 |
| :--- | :--- | :--- | :--- | :--- |
| `single_ts_outer_rank` | `single` | 1 | 基础一阶变换，如 `Rank(TsMean(SLog1p($f), w))` | 极低 |
| `binary_same_ts` | `binary` | 2 | 相同时间窗口下的二元互动，如 `Rank(Div(TsMean($f1,w), TsMean($f2,w)))` | 中 |
| `binary_mixed_ts` | `binary_mixed` | 2 | 混合算子的二元组合，增强特征非线性表达能力 | 中 |
| `triple_modulation` | `triple` | 3 | 三元调制逻辑，模拟复杂的资金流向互动 | 高 |
| `quad_balanced` | `quad` | 4 | 四元平衡结构，适用于深度非线性因子 | 极高 |
| `multi_window_gap` | `multi_window` | 2 | 跨周期动量/反转，如 `Rank(Div(TsMean($f,5), TsMean($f,60)))` | 中 |

---

## 3. 核心算子集

### 一元算子 (Unary)
- `Abs`: 绝对值
- `SLog1p`: 符号对数变换（处理长尾分布）
- `Rank`: 截面百分比排名（标准化核心）
- `Inv`: 倒数
- `Sign`: 符号

### 二元算子 (Binary)
- `Add`, `Sub`, `Mul`, `Div`: 基础四则运算
- `Greater`, `Less`: 比较取大/小（非线性触发）

### 时间序列算子 (Rolling)
- `TsMean`: 滚动均值
- `TsStd`: 滚动标准差
- `TsIr`: 滚动信息比（均值/标差）
- `TsRank`: 滚动序
- `TsDelta`: 滚动差分
- `TsPctChange`: 滚动涨跌幅
- `TsWMA`: 加权移动平均
- `TsEMA`: 指数移动平均

---

## 4. 评价指标体系

系统采用多维度加权评分逻辑：
1. **IC Edge (30%)**: 因子在测试集上的 RankIC 均值。
2. **IR Stability (20%)**: IC 的信息比率，代表收益的稳定性。
3. **Hit Rate (15%)**: IC 为正的交易日占比。
4. **Yearly Stability (15%)**: 年度稳定性，要求在绝大多数年份都有正向收益。
5. **Coverage (10%)**: 因子对全市场的覆盖程度。
6. **Complexity Penalty (5%)**: 鼓励简洁的表达式，惩罚过度拟合。
