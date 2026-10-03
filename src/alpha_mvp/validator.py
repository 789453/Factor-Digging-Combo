from __future__ import annotations
from dataclasses import dataclass
from .parser import Node
from .ops import UNARY, BINARY, ROLLING, PAIR_ROLLING, ALL_OPERATORS

@dataclass
class ValidationResult:
    ok: bool
    reason: str = ""
    depth: int = 0
    nodes: int = 0
    ts_ops: int = 0
    pair_ops: int = 0
    binary_ops: int = 0

class Validator:
    def __init__(self, fields: set[str], windows: set[int], max_depth=4, max_nodes=10,
                 max_ts_ops=2, max_pair_ops=1, max_binary_ops=1):
        self.fields = fields
        self.windows = windows
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.max_ts_ops = max_ts_ops
        self.max_pair_ops = max_pair_ops
        self.max_binary_ops = max_binary_ops

    def validate(self, node: Node) -> ValidationResult:
        try:
            depth, nodes, ts_ops, pair_ops, bin_ops = self._walk(node)
        except ValueError as e:
            return ValidationResult(False, str(e))
        if depth > self.max_depth:
            return ValidationResult(False, f"DEPTH_EXCEEDED:{depth}", depth, nodes, ts_ops, pair_ops, bin_ops)
        if nodes > self.max_nodes:
            return ValidationResult(False, f"NODES_EXCEEDED:{nodes}", depth, nodes, ts_ops, pair_ops, bin_ops)
        if ts_ops > self.max_ts_ops:
            return ValidationResult(False, f"TS_OPS_EXCEEDED:{ts_ops}", depth, nodes, ts_ops, pair_ops, bin_ops)
        if pair_ops > self.max_pair_ops:
            return ValidationResult(False, f"PAIR_OPS_EXCEEDED:{pair_ops}", depth, nodes, ts_ops, pair_ops, bin_ops)
        if bin_ops > self.max_binary_ops:
            return ValidationResult(False, f"BINARY_OPS_EXCEEDED:{bin_ops}", depth, nodes, ts_ops, pair_ops, bin_ops)
        return ValidationResult(True, "OK", depth, nodes, ts_ops, pair_ops, bin_ops)

    def _walk(self, node: Node) -> tuple[int, int, int, int, int]:
        if node.kind == "field":
            if node.value not in self.fields:
                raise ValueError(f"UNKNOWN_FIELD:{node.value}")
            return 1, 1, 0, 0, 0
        if node.kind == "const":
            return 1, 1, 0, 0, 0
        if node.kind != "op":
            raise ValueError(f"UNKNOWN_NODE:{node.kind}")
        op = node.value
        if op not in ALL_OPERATORS:
            raise ValueError(f"UNKNOWN_OPERATOR:{op}")
        n = len(node.args)
        if op in UNARY and n != 1:
            raise ValueError(f"ARITY:{op}")
        if op in BINARY and n != 2:
            raise ValueError(f"ARITY:{op}")
        if op in ROLLING:
            if n != 2 or node.args[1].kind != "const":
                raise ValueError(f"ROLLING_NEEDS_WINDOW:{op}")
            w = int(node.args[1].value)
            if w not in self.windows:
                raise ValueError(f"BAD_WINDOW:{op}:{w}")
        if op in PAIR_ROLLING:
            if n != 3 or node.args[2].kind != "const":
                raise ValueError(f"PAIR_ROLLING_NEEDS_WINDOW:{op}")
            w = int(node.args[2].value)
            if w not in self.windows:
                raise ValueError(f"BAD_WINDOW:{op}:{w}")
        stats = [self._walk(a) for a in node.args if not (a.kind == "const" and op in (set(ROLLING) | set(PAIR_ROLLING)))]
        depth = 1 + max((s[0] for s in stats), default=0)
        nodes = 1 + sum(s[1] for s in stats)
        ts_ops = (1 if op in ROLLING else 0) + sum(s[2] for s in stats)
        pair_ops = (1 if op in PAIR_ROLLING else 0) + sum(s[3] for s in stats)
        bin_ops = (1 if op in BINARY else 0) + sum(s[4] for s in stats)
        if op == "Log":
            child = node.args[0]
            if not (child.kind == "op" and child.value in {"Abs", "SLog1p", "Greater"}):
                raise ValueError("LOG_INPUT_UNSAFE")
        if op == "Pow":
            rhs = node.args[1]
            if rhs.kind != "const" or rhs.value not in {0.5, 2.0, 3.0}:
                raise ValueError("POW_ONLY_CONSTANT_0.5_2_3")
        return depth, nodes, ts_ops, pair_ops, bin_ops
