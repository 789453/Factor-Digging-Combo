from __future__ import annotations
from collections import OrderedDict
import numpy as np
from .parser import Node, parse_expr, canonical
from . import ops
from . import fastops
from .validator import Validator

class EvalError(RuntimeError):
    pass

class BatchEvaluator:
    def __init__(self, panels: dict[str, np.ndarray], dates: list[str], codes: list[str], windows=(10,20,30,40,50),
                 max_depth=4, max_nodes=10, max_ts_ops=2,
                 max_pair_ops=1, max_binary_ops=1, max_cache_items=256):
        self.panels = panels
        self.dates = dates
        self.codes = codes
        self.windows = tuple(windows)
        self.validator = Validator(
            set(panels.keys()),
            set(windows),
            max_depth=max_depth,
            max_nodes=max_nodes,
            max_ts_ops=max_ts_ops,
            max_pair_ops=max_pair_ops,
            max_binary_ops=max_binary_ops,
        )
        self.max_cache_items = max(0, int(max_cache_items))
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def eval_expr(self, expr: str) -> tuple[np.ndarray | None, str]:
        try:
            node = parse_expr(expr)
            vr = self.validator.validate(node)
            if not vr.ok:
                return None, vr.reason
            return self._eval_node(node), "OK"
        except Exception as e:
            return None, f"EVAL_ERROR:{type(e).__name__}:{e}"

    def eval_batch(self, exprs: list[str], max_workers: int = 4) -> list[tuple[np.ndarray | None, str]]:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self.eval_expr, exprs))

    def _eval_node(self, node: Node) -> np.ndarray:
        key = canonical(node)
        if key in self._cache:
            cached = self._cache.pop(key)
            self._cache[key] = cached
            return cached
        if node.kind == "field":
            if node.value not in self.panels:
                raise EvalError(f"unknown field {node.value}")
            return self.panels[node.value]
        elif node.kind == "const":
            out = np.full_like(next(iter(self.panels.values())), float(node.value), dtype=float)
        elif node.kind == "op":
            op = node.value
            if op in ops.UNARY:
                if op == "Rank":
                    out = fastops.fast_rank_cs(self._eval_node(node.args[0]))
                else:
                    out = ops.UNARY[op](self._eval_node(node.args[0]))
            elif op in ops.BINARY:
                out = ops.BINARY[op](self._eval_node(node.args[0]), self._eval_node(node.args[1]))
            elif op in ops.ROLLING:
                out = ops.ROLLING[op](self._eval_node(node.args[0]), int(node.args[1].value))
            elif op in ops.PAIR_ROLLING:
                if op == "TsCorr":
                    out = fastops.fast_rolling_corr(
                        self._eval_node(node.args[0]), self._eval_node(node.args[1]), int(node.args[2].value)
                    )
                else:
                    out = ops.PAIR_ROLLING[op](self._eval_node(node.args[0]), self._eval_node(node.args[1]), int(node.args[2].value))
            else:
                raise EvalError(f"unknown op {op}")
        else:
            raise EvalError(f"unknown node {node.kind}")
        # Some accelerated/Pandas results can expose read-only buffers.
        # Evaluation always normalizes into an owned writable array.
        out = np.array(out, dtype=float, copy=True)
        out[~np.isfinite(out)] = np.nan
        if self.max_cache_items:
            self._cache[key] = out
            while len(self._cache) > self.max_cache_items:
                self._cache.popitem(last=False)
        return out

def make_panels(df: pd.DataFrame, feature_cols: list[str], value_col="close"):
    """
    高效构建面板数据，避免多次 pivot
    """
    # 确保日期和代码有序
    dates = sorted(df["trade_date"].astype(str).unique().tolist())
    codes = sorted(df["ts_code"].unique().tolist())
    
    date_map = {d: i for i, d in enumerate(dates)}
    code_map = {c: i for i, c in enumerate(codes)}
    
    n_dates = len(dates)
    n_codes = len(codes)
    
    # 提取索引位置
    row_idx = df["trade_date"].astype(str).map(date_map).values
    col_idx = df["ts_code"].map(code_map).values
    
    pivots = {}
    for c in feature_cols + [value_col]:
        if c not in df.columns:
            continue
        # 预分配矩阵
        arr = np.full((n_dates, n_codes), np.nan, dtype=np.float64)
        # 填充数据
        arr[row_idx, col_idx] = df[c].values
        pivots[c] = arr
        
    return pivots, dates, codes
