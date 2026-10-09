"""Update navigation and publish lightweight copies; never edit research outputs."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    for path in [root / 'README.md', *list((root / 'docs').glob('*.md'))]:
        text = path.read_text(encoding='utf-8')
        updated = text.replace('effective_combo_20261009_extended_final_review_v1', 'effective_combo_20261009_extended_final_review_v2').replace('57页', '69页')
        if updated != text:
            path.write_text(updated, encoding='utf-8')
    report = root / 'docs/EFFECTIVE_COMBO_EXTENDED_RESEARCH_RESULTS_20261009.md'
    text = report.read_text(encoding='utf-8')
    marker = '大池15叶另有2025-07冻结的静态版本，OOS继续使用该模型而非2026重训。'
    clarification = '静态版本使用扩展训练窗口，在线版本使用过去12个月；这个对照同时改变训练窗口和更新频率，不能把差异单独归因于月更。'
    if clarification not in text:
        text = text.replace(marker, marker + clarification)
    marker2 = '固定最小成熟样本300，足够时：'
    text = text.replace(marker2, '固定最小成熟样本300；这是相关的币种/时点观测数量，不是300个独立样本。足够时：')
    report.write_text(text, encoding='utf-8')
    evidence = root / 'docs/evidence/effective_combo_extension_20261009'
    versions_path = evidence / 'runtime_versions.json'
    versions = json.loads(versions_path.read_text(encoding='utf-8'))
    versions.update(pandas=pd.__version__, python=sys.version)
    versions_path.write_text(json.dumps(versions, ensure_ascii=False, indent=2), encoding='utf-8')
    contract_path = evidence / 'PUBLICATION_CONTRACT.json'
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    for name in ['delivery_qa.json', 'manifest.json', 'core_long_short_decomposition.csv', 'core_temporal_uncertainty.csv']:
        source = root / 'outputs/effective_combo_20261009_extended_delivery_review_v2' / name
        target = evidence / ('final_v2_' + name)
        shutil.copy2(source, target)
        record = {'source': str(source.relative_to(root)), 'copy': str(target.relative_to(root)), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
        contract['records'] = [r for r in contract['records'] if r['copy'] != record['copy']]
        contract['records'].append(record)
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'public_evidence_records': len(contract['records']), 'pandas': pd.__version__, 'pages': 69}))


if __name__ == '__main__':
    main()
