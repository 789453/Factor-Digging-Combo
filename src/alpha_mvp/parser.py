from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class Node:
    kind: str
    value: Any
    args: tuple["Node", ...] = ()
    def __str__(self) -> str:
        if self.kind == "field":
            return f"${self.value}"
        if self.kind == "const":
            if float(self.value).is_integer():
                return str(int(self.value))
            return str(self.value)
        if self.kind == "op":
            return f"{self.value}({','.join(str(a) for a in self.args)})"
        raise ValueError(self.kind)

class ParseError(ValueError):
    pass

def split_args(s: str) -> list[str]:
    args, cur, depth = [], [], 0
    for ch in s:
        if ch == "(":
            depth += 1; cur.append(ch)
        elif ch == ")":
            depth -= 1; cur.append(ch)
        elif ch == "," and depth == 0:
            args.append("".join(cur).strip()); cur = []
        else:
            cur.append(ch)
    if cur:
        args.append("".join(cur).strip())
    return args

def parse_expr(s: str) -> Node:
    s = s.strip()
    if not s:
        raise ParseError("empty expression")
    if s.startswith("$"):
        return Node("field", s[1:])
    try:
        return Node("const", float(s))
    except ValueError:
        pass
    idx = s.find("(")
    if idx <= 0 or not s.endswith(")"):
        raise ParseError(f"invalid expression: {s}")
    op = s[:idx]
    inner = s[idx + 1:-1]
    return Node("op", op, tuple(parse_expr(a) for a in split_args(inner)))

def canonical(node: Node) -> str:
    if node.kind in ("field", "const"):
        return str(node)
    args = [canonical(a) for a in node.args]
    if node.value in {"Add", "Mul", "TsCorr", "TsCov"} and len(args) == 2:
        args = sorted(args)
    return f"{node.value}({','.join(args)})"
