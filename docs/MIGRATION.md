# Migration and Cleanup

## New source of truth

- configuration: `configs/research/`
- orchestration: `research/workflow.py`
- templates: `research/templates.py`
- evaluation: `research/evaluation.py`
- selection and pairing: `research/selection.py`
- CLI: `research_cli.py`

## Retained proven infrastructure

- DuckDB extraction in `data.py`;
- domain field construction in `fields.py`;
- expression parser and canonicalization in `parser.py`;
- operator implementation in `ops.py` and `fastops.py`;
- AST validation in `validator.py`;
- panel evaluator in `evaluator.py`.

These modules remain implementation dependencies and will be hardened in later
iterations without changing the research contract.

## Retired concepts

The old MVP, Phase2, and validation entry points are historical
implementations. New development must not import them. Redundant modules and
scripts were removed after the new kernel and tests were established.

Historical output directories are deliberately retained. They are evidence of
past experiments, even where their manifests or metrics are incomplete.
