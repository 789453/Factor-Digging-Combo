import numpy as np
import pandas as pd
import json
import pytest

from src.alpha_mvp.research.factor_combo import (
    _registry, _robust_fit, align_completed_hourly, deduplicate_factors,
)


def test_hourly_becomes_available_only_after_completion():
    values = np.array([[1.0], [2.0]], dtype=np.float32)
    out = align_completed_hourly(values, ["202301010000", "202301010100"],
                                 ["202301010055", "202301010100", "202301010155", "202301010200"])
    assert np.isnan(out[0, 0])
    assert out[:, 0][1:].tolist() == [1, 1, 2]


def test_alias_pruning_and_train_only_extreme_scaling():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(100, 4))
    x = np.stack([base, base * 1.000001, rng.normal(size=base.shape)], axis=-1)
    registry = pd.DataFrame({"expr_hash": ["a", "b", "c"], "family": ["a", "b", "c"]})
    train = np.zeros(base.shape, dtype=bool)
    train[:60] = True
    retained, selected, notes = deduplicate_factors(x, registry, train, .97)
    assert selected.expr_hash.tolist() == ["a", "c"]
    assert notes.iloc[1].alias_of == "a"
    z, scaler = _robust_fit(retained, train)
    changed = retained.copy()
    changed[60:] = 1e20
    z_changed, changed_scaler = _robust_fit(changed, train)
    assert scaler == changed_scaler
    assert np.max(z_changed) <= 5
    assert np.allclose(z[:60], z_changed[:60])


def test_mining_manifest_lineage_and_holdout_column_firewall(tmp_path):
    sources = []
    for clock, timeframe in (("native5", "5m"), ("hourly", "1h")):
        folder = tmp_path / clock
        folder.mkdir()
        chosen = folder / "selected_factors.csv"
        pd.DataFrame([{"expr": "$x", "expr_hash": clock,
                       "template_family": "trend", "direction": 1,
                       "rank_equivalence_hash": clock, "template_name": "simple",
                       "fields": "x", "operators": "", "windows": "",
                       "target_mode": "return", "target_scope": "absolute",
                       "holdout_rank_ic": 999.}]).to_csv(chosen, index=False)
        manifest = {"status": "COMPLETED", "experiment_id": clock,
                    "framework_version": "test", "artifacts": {"selected_factors": chosen.name},
                    "data_fingerprint": {"primary_timeframe": timeframe},
                    "config": {"evaluation": {"factor_mode": "time_series",
                                               "entry_lag": 0 if clock == "native5" else 1,
                                               "horizon": 1 if clock == "native5" else 4,
                                               "target_mode": "return",
                                               "target_scope": "absolute"}}}
        (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        sources.append({"clock": clock, "role": "alpha", "path": str(chosen), "limit": 1})
    registry = _registry({"factor_sources": sources})
    assert registry.horizon_bars.tolist() == [1, 48]
    assert registry.lag_bars.tolist() == [0, 12]
    assert "holdout_rank_ic" not in registry
    changed = pd.read_csv(sources[0]["path"])
    changed["holdout_rank_ic"] = -999.
    changed.to_csv(sources[0]["path"], index=False)
    again = _registry({"factor_sources": sources})
    assert again.expr_hash.tolist() == registry.expr_hash.tolist()
    assert again.source_selected_sha256.iloc[0] != registry.source_selected_sha256.iloc[0]
    bad_path = tmp_path / "hourly" / "manifest.json"
    bad = json.loads(bad_path.read_text(encoding="utf-8"))
    bad["status"] = "RUNNING"
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible mining manifest"):
        _registry({"factor_sources": sources})
