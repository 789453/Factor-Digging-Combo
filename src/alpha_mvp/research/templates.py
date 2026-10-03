from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import combinations, permutations, product
from pathlib import Path
from typing import Iterable

import yaml

from ..parser import canonical, parse_expr
from ..validator import Validator
from .proposal_prior import proposal_bonus


@dataclass(frozen=True)
class TemplateFamily:
    name: str
    kind: str
    order: int
    target_modes: tuple[str, ...] = ()
    hypothesis_card: str | None = None
    enabled: bool = True
    unary_pre: tuple[str, ...] = ("Id",)
    ts_ops: tuple[str, ...] = ()
    left_ts_ops: tuple[str, ...] = ()
    right_ts_ops: tuple[str, ...] = ()
    pair_ops: tuple[str, ...] = ()
    binary_ops: tuple[str, ...] = ()
    outer_transforms: tuple[str, ...] = ("Rank",)
    short_windows: tuple[int, ...] = ()
    long_windows: tuple[int, ...] = ()
    forms: tuple[str, ...] = ()
    fields_a: tuple[str, ...] = ()
    fields_b: tuple[str, ...] = ()
    fields_c: tuple[str, ...] = ()
    max_count: int | None = None
    max_depth: int = 7
    max_nodes: int = 24
    max_ts_ops: int = 4
    max_pair_ops: int = 1
    max_binary_ops: int = 4


@dataclass(frozen=True)
class ExpressionRecord:
    expr: str
    canonical: str
    expr_hash: str
    rank_equivalence_hash: str
    template_name: str
    template_family: str
    template_order: int
    fields: tuple[str, ...]
    operators: tuple[str, ...]
    windows: tuple[int, ...]
    depth: int
    nodes: int
    priority_score: float


_SEQUENCE_KEYS = {
    "unary_pre", "ts_ops", "left_ts_ops", "right_ts_ops", "pair_ops",
    "binary_ops", "outer_transforms", "short_windows", "long_windows", "forms",
    "fields_a", "fields_b", "fields_c", "target_modes",
}


def load_template_families(path: str) -> tuple[list[TemplateFamily], dict]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "version" not in raw or not isinstance(raw.get("families"), list):
        raise ValueError("Template config requires version and families")
    families = []
    for item in raw["families"]:
        values = dict(item)
        for key in _SEQUENCE_KEYS:
            if key in values:
                values[key] = tuple(values[key])
        if values.get("hypothesis_card") and not Path(values["hypothesis_card"]).is_file():
            raise ValueError(f"Template hypothesis_card not found: {values['hypothesis_card']}")
        if values.get("target_modes") and not set(values["target_modes"]) <= {
            "return", "upside_semivariance", "downside_semivariance", "total_variance", "liquidity_concentration"
        }:
            raise ValueError(f"Invalid target_modes for template: {values.get('name')}")
        families.append(TemplateFamily(**values))
    return families, raw


def _stable_value(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:15], 16) / float(16**15)


def _outer(expr: str, transform: str) -> str:
    if transform == "Id":
        return expr
    if transform == "RankSLog1p":
        return f"Rank(SLog1p({expr}))"
    return f"{transform}({expr})"


def _walk_meta(expr: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...], int, int]:
    node = parse_expr(expr)
    fields: list[str] = []
    operators: list[str] = []
    windows: list[int] = []

    def walk(n):
        if n.kind == "field":
            fields.append(n.value)
            return 1, 1
        if n.kind == "const":
            return 1, 1
        operators.append(n.value)
        child_stats = []
        for index, arg in enumerate(n.args):
            is_window = (
                arg.kind == "const"
                and index == len(n.args) - 1
                and n.value.startswith(("Ts", "Ref"))
            )
            if is_window:
                windows.append(int(arg.value))
            else:
                child_stats.append(walk(arg))
        return (
            1 + max((x[0] for x in child_stats), default=0),
            1 + sum(x[1] for x in child_stats),
        )

    depth, nodes = walk(node)
    return tuple(fields), tuple(operators), tuple(windows), depth, nodes


def _rank_equivalence(node) -> str:
    """Canonical key for transforms that are identical under daily RankIC."""
    current = node
    while (
        current.kind == "op"
        and current.value in {"Rank", "SLog1p"}
        and len(current.args) == 1
    ):
        current = current.args[0]
    return canonical(current)


def _candidate_expressions(
    spec: TemplateFamily,
    fields: tuple[str, ...],
    windows: tuple[int, ...],
) -> Iterable[str]:
    if spec.kind == "single":
        for field in fields:
            for window in windows:
                for ts_op in spec.ts_ops:
                    for unary in spec.unary_pre:
                        source = f"${field}" if unary == "Id" else f"{unary}(${field})"
                        base = f"{ts_op}({source},{window})"
                        for outer in spec.outer_transforms:
                            yield _outer(base, outer)

    elif spec.kind == "binary_same":
        for left, right in combinations(fields, 2):
            for window in windows:
                for ts_op in spec.ts_ops:
                    a = f"{ts_op}(${left},{window})"
                    b = f"{ts_op}(${right},{window})"
                    for binary in spec.binary_ops:
                        for outer in spec.outer_transforms:
                            yield _outer(f"{binary}({a},{b})", outer)

    elif spec.kind == "binary_mixed":
        left_ops = spec.left_ts_ops or spec.ts_ops
        right_ops = spec.right_ts_ops or spec.ts_ops
        for left, right in permutations(fields, 2):
            for window in windows:
                for left_op in left_ops:
                    for right_op in right_ops:
                        for binary in spec.binary_ops:
                            a = f"{left_op}(${left},{window})"
                            b = f"{right_op}(${right},{window})"
                            for outer in spec.outer_transforms:
                                yield _outer(f"{binary}({a},{b})", outer)

    elif spec.kind == "pair_rolling":
        for left, right in combinations(fields, 2):
            for window in windows:
                for pair_op in spec.pair_ops:
                    for outer in spec.outer_transforms:
                            yield _outer(f"{pair_op}(${left},${right},{window})", outer)

    elif spec.kind == "pair_change":
        for left, right in combinations(fields, 2):
            for long in spec.long_windows or windows:
                for short in spec.short_windows or windows:
                    if short >= long:
                        continue
                    for pair_op in spec.pair_ops:
                        base = f"TsDelta({pair_op}(${left},${right},{long}),{short})"
                        for outer in spec.outer_transforms:
                            yield _outer(base, outer)

    elif spec.kind == "cross_window_binary":
        left_ops = spec.left_ts_ops or spec.ts_ops
        right_ops = spec.right_ts_ops or spec.ts_ops
        for left, right in permutations(fields, 2):
            for short in spec.short_windows or windows:
                for long in spec.long_windows or windows:
                    if short >= long:
                        continue
                    for left_op in left_ops:
                        for right_op in right_ops:
                            a = f"{left_op}(${left},{short})"
                            b = f"{right_op}(${right},{long})"
                            for binary in spec.binary_ops:
                                for outer in spec.outer_transforms:
                                    yield _outer(f"{binary}({a},{b})", outer)

    elif spec.kind == "normalized_binary":
        for left, right in combinations(fields, 2):
            for window in windows:
                for ts_op in spec.ts_ops:
                    a = f"{ts_op}(${left},{window})"
                    b = f"{ts_op}(${right},{window})"
                    normalized = f"Div(Sub({a},{b}),Add(Abs({a}),Abs({b})))"
                    for outer in spec.outer_transforms:
                        yield _outer(normalized, outer)

    elif spec.kind == "multi_window":
        for field in fields:
            for short in spec.short_windows:
                for long in spec.long_windows:
                    if short >= long:
                        continue
                    for ts_op in spec.ts_ops:
                        a = f"{ts_op}(${field},{short})"
                        b = f"{ts_op}(${field},{long})"
                        for binary in spec.binary_ops:
                            for outer in spec.outer_transforms:
                                yield _outer(f"{binary}({a},{b})", outer)

    elif spec.kind in {"triple", "quad"}:
        arity = 3 if spec.kind == "triple" else 4
        letters = "ABCD"[:arity]
        for selected in combinations(fields, arity):
            for window in windows:
                for ts_op in spec.ts_ops:
                    replacements = {
                        letter: f"{ts_op}(${field},{window})"
                        for letter, field in zip(letters, selected)
                    }
                    for form in spec.forms:
                        expr = form
                        for letter, replacement in replacements.items():
                            expr = expr.replace("{" + letter + "}", replacement)
                        yield expr
    elif spec.kind == "time_series_formula":
        role_pools = []
        configured = (spec.fields_a, spec.fields_b, spec.fields_c)
        for index in range(spec.order):
            pool = configured[index] or fields
            role_pools.append(tuple(field for field in pool if field in fields))
        if any(not pool for pool in role_pools):
            return
        short_windows = spec.short_windows or windows
        long_windows = spec.long_windows or windows
        for selected in product(*role_pools):
            if len(set(selected)) != len(selected):
                continue
            for short in short_windows:
                for long in long_windows:
                    if short >= long:
                        continue
                    replacements = {
                        "A": f"${selected[0]}",
                        "S": str(short),
                        "L": str(long),
                    }
                    if spec.order >= 2:
                        replacements["B"] = f"${selected[1]}"
                    if spec.order >= 3:
                        replacements["C"] = f"${selected[2]}"
                    for form in spec.forms:
                        expr = form
                        for name, replacement in replacements.items():
                            expr = expr.replace("{" + name + "}", replacement)
                        yield expr
    else:
        raise ValueError(f"Unknown template kind: {spec.kind}")


def generate_expressions(
    fields: list[str] | tuple[str, ...],
    windows: list[int] | tuple[int, ...],
    families: list[TemplateFamily],
    max_expressions: int,
    max_per_family: int,
    seed: int = 42,
    priority_fields: Iterable[str] = (),
    priority_operators: Iterable[str] = (),
    priority_templates: Iterable[str] = (),
    priority_windows: Iterable[int] = (),
    diversity_share: float = 0.50,
    evidence_prior: dict[str, float] | None = None,
    evidence_strength: float = 0.0,
) -> list[ExpressionRecord]:
    unique_fields = tuple(dict.fromkeys(fields))
    window_tuple = tuple(sorted(set(windows)))
    if not unique_fields:
        raise ValueError("At least one field is required")
    priority_field_set = set(priority_fields)
    priority_operator_set = set(priority_operators)
    priority_template_set = set(priority_templates)
    priority_window_set = set(priority_windows)
    field_tuple = tuple(sorted(
        unique_fields,
        key=lambda name: (
            name not in priority_field_set,
            _stable_value(name, seed),
        ),
    ))

    family_records: list[list[ExpressionRecord]] = []
    global_seen: set[str] = set()

    for spec in families:
        if not spec.enabled:
            continue
        valid_windows = set(window_tuple) | set(spec.short_windows) | set(spec.long_windows)
        validator = Validator(
            set(field_tuple),
            valid_windows,
            max_depth=spec.max_depth,
            max_nodes=spec.max_nodes,
            max_ts_ops=spec.max_ts_ops,
            max_pair_ops=spec.max_pair_ops,
            max_binary_ops=spec.max_binary_ops,
        )
        limit = min(spec.max_count or max_per_family, max_per_family)
        candidates = _candidate_expressions(spec, field_tuple, window_tuple)
        sampled: list[tuple[float, ExpressionRecord]] = []
        family_seen: set[str] = set()
        # Bound grammar expansion. The deterministically shuffled field order
        # makes this a reproducible stratified scan instead of a full Cartesian
        # materialization (which is especially costly for triple/quad forms).
        scan_limit = max(2_000, limit * 40)
        for scanned, expr in enumerate(candidates, start=1):
            if scanned > scan_limit:
                break
            try:
                node = parse_expr(expr)
                validation = validator.validate(node)
                if not validation.ok:
                    continue
                can = canonical(node)
                equivalence = _rank_equivalence(node)
                if equivalence in global_seen or equivalence in family_seen:
                    continue
                family_seen.add(equivalence)
                flds, operators, expr_windows, depth, nodes = _walk_meta(expr)
                priority = (
                    3.0 * int(
                        spec.name in priority_template_set
                        or spec.kind in priority_template_set
                    )
                    + 1.5 * len(set(flds) & priority_field_set)
                    + 1.0 * len(set(operators) & priority_operator_set)
                    + 0.5 * len(set(expr_windows) & priority_window_set)
                )
                if evidence_prior:
                    priority += proposal_bonus(
                        evidence_prior, spec.name, flds, operators,
                        expr_windows, evidence_strength,
                    )
                record = ExpressionRecord(
                    expr=expr,
                    canonical=can,
                    expr_hash=hashlib.sha256(can.encode("utf-8")).hexdigest(),
                    rank_equivalence_hash=hashlib.sha256(
                        equivalence.encode("utf-8")
                    ).hexdigest(),
                    template_name=spec.name,
                    template_family=spec.kind,
                    template_order=spec.order,
                    fields=flds,
                    operators=operators,
                    windows=expr_windows,
                    depth=depth,
                    nodes=nodes,
                    priority_score=priority,
                )
                sampled.append((_stable_value(can, seed), record))
            except (ValueError, TypeError):
                continue
        sampled.sort(key=lambda item: (-item[1].priority_score, item[0]))
        selected = [item[1] for item in sampled[:limit]]
        global_seen.update(
            _rank_equivalence(parse_expr(record.expr))
            for record in selected
        )
        family_records.append(selected)

    output: list[ExpressionRecord] = []
    selected_hashes: set[str] = set()
    diversity_target = min(
        max_expressions,
        int(round(max_expressions * diversity_share)),
    )
    index = 0
    while len(output) < diversity_target:
        added = False
        for records in family_records:
            if index < len(records):
                record = records[index]
                output.append(record)
                selected_hashes.add(record.expr_hash)
                added = True
                if len(output) >= diversity_target:
                    break
        if not added:
            break
        index += 1
    remainder = sorted(
        (
            record
            for records in family_records
            for record in records
            if record.expr_hash not in selected_hashes
        ),
        key=lambda record: (
            -record.priority_score,
            _stable_value(record.canonical, seed),
        ),
    )
    output.extend(remainder[:max(0, max_expressions - len(output))])
    return output
