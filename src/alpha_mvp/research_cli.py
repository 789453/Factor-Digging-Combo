from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .research.config import load_research_config
from .research.long_only import load_long_only_config, run_long_only_report
from .research.crypto_return_patch import (
    load_crypto_return_patch_config,
    run_crypto_return_patch,
)
from .research.workflow import run_research


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the unified factor research workflow."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config",
        help="Research YAML configuration path.",
    )
    group.add_argument(
        "--long-only-config",
        help="Frozen Top50 full-market long-only report configuration.",
    )
    group.add_argument(
        "--crypto-return-patch-config",
        help="Report-only crypto cumulative-return patch configuration.",
    )
    args = parser.parse_args()
    if args.crypto_return_patch_config:
        manifest = run_crypto_return_patch(
            load_crypto_return_patch_config(args.crypto_return_patch_config)
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    elif args.long_only_config:
        manifest = run_long_only_report(
            load_long_only_config(args.long_only_config)
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        if raw.get("mode") == "factor_combo":
            from .research.factor_combo import run_factor_combo
            manifest = run_factor_combo(raw, args.config)
            print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
        elif raw.get("mode") == "aligned_crypto":
            from .research.aligned_workflow import run_aligned_crypto

            manifest = run_aligned_crypto(raw, args.config)
            print(json.dumps(manifest["results"], ensure_ascii=False, indent=2))
        else:
            manifest = run_research(load_research_config(args.config))
            print(json.dumps(manifest["research_summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
