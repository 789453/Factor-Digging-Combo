"""Read-only report and integrity checks; no generation, fitting or selection.

Inputs must be completed experiments. The final source must already have its
selection frozen before any audit metric is read. Output must be a new folder.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.alpha_mvp.research.complex_alpha_comparison import render_comparison


def inspect_evidence(source):
    source=Path(source)
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    freeze=json.loads((source/'selection_freeze.json').read_text(encoding='utf-8'))
    if manifest['status']!='COMPLETED' or freeze['status']!='FROZEN':
        raise ValueError('only completed, frozen evidence can be reviewed')
    if any('holdout' in x for x in freeze['selection_uses']):
        raise ValueError('holdout metric declared as research selection input')
    code=hashlib.sha256(b''.join(p.read_bytes() for p in sorted((source/'source_snapshot').glob('complex_alpha_*.py')))).hexdigest()
    if code!=manifest['code_sha256']:raise ValueError('source snapshot hash mismatch')
    for item in manifest.get('source_files',[]):
        if 'path' not in item:continue
        st=Path(item['path']).stat()
        if st.st_size!=item['size'] or st.st_mtime_ns!=item['mtime_ns']:raise ValueError('frozen input fingerprint changed: '+item['path'])
    cfg=yaml.safe_load((source/'config.yaml').read_text(encoding='utf-8'))
    signature=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()+code.encode()).hexdigest()
    if signature!=freeze['run_signature']:raise ValueError('frozen configuration signature mismatch')
    ledger=pd.read_parquet(source/'combo_5m_ledger.parquet')
    expected=ledger.price_gross+ledger.funding-ledger.fee
    np.testing.assert_allclose(ledger.net,expected,atol=1e-13)
    equity=ledger.equity_before_cashflows.to_numpy()
    np.testing.assert_allclose(equity[1:]/equity[:-1]-1,ledger.net.to_numpy()[:-1],atol=2e-12)
    pnl=pd.read_parquet(source/'combo_asset_pnl.parquet')
    np.testing.assert_allclose(pnl.mean(axis=1),ledger.net,atol=1e-13)
    dates=pd.to_datetime(ledger.completed_5m,format='%Y%m%d%H%M')
    if not np.all(np.diff(dates.asi8 if hasattr(dates,'asi8') else dates.to_numpy().astype('datetime64[ns]').astype('int64'))==300_000_000_000):
        raise ValueError('native5 evidence clock is not continuous')
    metrics=json.loads((source/'combo_metrics.json').read_text())
    for period,frame in ledger.groupby('period',sort=False):
        measured=(np.prod(1+frame.net.to_numpy())-1)*100
        np.testing.assert_allclose(measured,metrics[period]['net_return_pct'],atol=1e-10)
    return {'source':str(source.resolve()),'status':'PASS','source_hash':code,
            'rows':len(ledger),'selected':freeze['selected'],'checks':[
                'frozen source/config signature','declared selection excludes holdout',
                'continuous native5 clock','price+funding-fee identity',
                'self-financing equity recursion','asset aggregate','period wealth reconciliation']}


def inspect_links(root):
    errors=[];count=0
    for file in Path(root).rglob('*.html'):
        body=file.read_text(encoding='utf-8');count+=1
        if '<meta charset=' not in body or '<title>' not in body:errors.append(str(file)+': missing document metadata')
        for ref in re.findall(r'(?:href|src)=[\"\']([^\"\']+)',body):
            if ref.startswith(('http:','https:','data:','#','mailto:')):continue
            target=(file.parent/ref.split('#')[0]).resolve()
            if not target.exists():errors.append(str(file)+': broken '+ref)
    if errors:raise ValueError('\n'.join(errors))
    return {'html_pages':count,'local_links_and_scripts':'PASS','visual_render_claim':'static checks only; no browser rendering'}


def finalize_review_navigation(root,include_ablation=False):
    """Compose diagnostic pages into this new atlas before its final review stamp."""
    root=Path(root)
    declared=[('index.html','研究结论'),('readouts.html','读出与消融'),
        ('uncertainty.html','不确定性'),('structure.html','结构辨识'),
        ('economics.html','成本与执行'),('iterations.html','迭代档案'),
        ('decay.html','期限与波动触发'),('audit.html','全部通道迁移'),
        ('additive.html','修正基线的贡献'),('coins.html','逐币归因')]
    pages=[(file,label) for file,label in declared if (root/file).exists()]
    navigation='<nav>'+' · '.join(f'<a href="{file}">{label}</a>' for file,label in pages)+'</nav>'
    extra=[(p.name,p.stem) for p in sorted(root.glob('coin_*.html'))]
    for file,_ in pages+extra:
        path=root/file;body=path.read_text(encoding='utf-8')
        body=re.sub(r'<nav>.*?</nav>',navigation,body,count=1,flags=re.S)
        if "name='viewport'" not in body and 'name="viewport"' not in body:
            body=body.replace('</head>',"<meta name='viewport' content='width=device-width,initial-scale=1'></head>",1)
        if file=='index.html':
            note='<p>主表mean_loss_gain_fraction及横轴来自中筛OOF；修正版按UTC每日轮换1h起点，旧轮次实际固定相位的时间单位勘误见采样合同。不确定性页区间来自完整15m预测，不与中筛采样混同。期限、波动分层和基线修正是冻结后解释诊断，不用于重新选取通道。</p>'
            documents=[('COMPLEX_ALPHA_RESEARCH_RESULTS_20261008.md','完整研究报告'),
                ('COMPLEX_ALPHA_CLOCK_CORRIGENDUM_20261008.md','时钟勘误'),
                ('COMPLEX_ALPHA_PRODUCT_CARD_20261008.md','历史弱正联合产品卡')]
            refs=[f'<a href="../../docs/{name}">{label}</a>' for name,label in documents
                if (root.parent.parent/'docs'/name).exists()]
            if refs:note+='<p>'+' · '.join(refs)+'</p>'
            ablation=root.parent/'complex_alpha_20261009_fixed_readout_ablation_v3'
            if include_ablation and (ablation/'manifest.json').exists():
                record=json.loads((ablation/'manifest.json').read_text(encoding='utf-8'))
                if record['status']!='COMPLETED' or record['holdout_used'] or record['new_fits']:
                    raise ValueError('related frozen predictor audit is incomplete or changes research scope')
                note+='<p><a href="../complex_alpha_20261009_fixed_readout_ablation_v3/index.html">原弱正产品：十一类固定移除的验证期诊断</a>（独立只读页面，不重新选择产品）</p>'
            body=body.replace('</h1>','</h1>'+note,1)
        path.write_text(body,encoding='utf-8')
    metadata=root/'visualization_manifest.json'
    manifest=json.loads(metadata.read_text(encoding='utf-8'))
    manifest['pages']=[file for file,_ in pages+extra];manifest['page_count']=len(pages+extra)
    manifest['completion_authority']='review_provenance.json and evidence_integrity.json'
    metadata.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',nargs='+',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--decay',action='store_true',help='Predeclared decay diagnostic of the final native-contract source')
    parser.add_argument('--all-frozen-audits',action='store_true',help='Audit every declared final-source channel without reselection')
    parser.add_argument('--additive-baseline',action='store_true',help='Restore saved OLD forecasts and compare OLD+increment without refitting')
    args=parser.parse_args();out=Path(args.out)
    if args.all_frozen_audits and not args.decay:raise ValueError('all frozen audits require the predeclared decay review')
    if args.additive_baseline and not args.all_frozen_audits:raise ValueError('additive comparison requires full frozen audit')
    if out.exists():raise ValueError('review output must be a new directory')
    if not (Path(args.sources[-1])/'selection_freeze.json').exists():
        raise ValueError('final research decision must be frozen before cross-round review')
    audits=[inspect_evidence(s) for s in args.sources]
    if args.decay:
        original=Path(args.sources[-1])/'source_snapshot';current=Path(__file__).resolve().parents[1]/'src/alpha_mvp/research'
        for name in ['complex_alpha_evaluation.py','complex_alpha_workflow.py','manual_alpha_derivatives.py','complex_alpha_time.py']:
            if (original/name).read_bytes()!=(current/name).read_bytes():raise ValueError('numerical replay source differs from frozen experiment: '+name)
        out.mkdir(parents=True)
        from scripts.complex_alpha_decay_review import render_decay
        render_decay(args.sources[-1],out,audit_all=args.all_frozen_audits,additive=args.additive_baseline)
    models=render_comparison(args.sources,out)
    from scripts.complex_alpha_coin_review import render_coin_review
    render_coin_review(args.sources,out)
    finalize_review_navigation(out,include_ablation=any(Path(s).name=='complex_alpha_20261008_round3' for s in args.sources))
    fig,axes=plt.subplots(1,3,figsize=(20,6),layout='constrained')
    for round_,frame in models.groupby('round'):
        axes[0].scatter(frame.mean_loss_gain_fraction*100,frame.validation_net_return_pct,label=round_,alpha=.7)
        axes[1].scatter(frame.mean_loss_gain_fraction*100,frame.validation_net_return_pct,label=round_,alpha=.7)
        active=frame[frame.validation_turnover>1e-8]
        axes[2].scatter(active.validation_turnover,active.validation_break_even_bps,label=round_,alpha=.7)
    axes[0].axvline(0,color='grey',ls=':');axes[0].axhline(0,color='grey',ls=':')
    axes[0].set(xlabel='Discovery screening-sample OOF loss improvement (%)',ylabel='Validation net wealth (%)',title='Information and economic value')
    axes[1].axvline(0,color='grey',ls=':');axes[1].axhline(0,color='grey',ls=':')
    axes[1].set(xlabel='Discovery screening-sample OOF loss improvement (%)',ylabel='Validation net wealth (%)',
        title='Same observations: detail near zero wealth',ylim=(-.5,.5))
    axes[2].axhline(4,color='firebrick',ls='--',label='Declared fee: 4 bps')
    axes[2].set(xlabel='Validation capital turnover',ylabel='Break-even cost (bps)',title='Forecast edge versus actual trading friction')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Complex Alpha: completed research rounds\nDifferent target contracts across rounds; these are diagnostic comparisons')
    fig.savefig(out/'research_evidence.png',dpi=160);plt.close(fig)
    links=inspect_links(out)
    for source in args.sources:
        viz=Path('visualizations')/Path(source).name
        if viz.exists():links[str(viz)]=inspect_links(viz)
    (out/'evidence_integrity.json').write_text(json.dumps({'status':'PASS','experiments':audits,'visualization_checks':links},ensure_ascii=False,indent=2),encoding='utf-8')
    models.to_csv(out/'all_readouts_review.csv',index=False)
    snapshot=out/'review_source_snapshot';snapshot.mkdir()
    hashes={}
    for name in ['render_complex_alpha_review.py','complex_alpha_decay_review.py','complex_alpha_additive_review.py','complex_alpha_coin_review.py']:
        script=Path(__file__).with_name(name);shutil.copyfile(script,snapshot/name);hashes[name]=hashlib.sha256(script.read_bytes()).hexdigest()
    protocol=Path('docs/COMPLEX_ALPHA_DECAY_DIAGNOSTIC_PROTOCOL_20261008.md')
    if args.decay:shutil.copyfile(protocol,snapshot/protocol.name)
    (out/'review_provenance.json').write_text(json.dumps({'status':'COMPLETED','report_code_sha256':hashes,'python':sys.version,
        'numpy':np.__version__,'pandas':pd.__version__,'source_decisions_unchanged':True,
        'sources':[str(Path(s).resolve()) for s in args.sources]},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'PASS','out':str(out.resolve()),'experiments':len(audits),'comparison_pages':links['html_pages']},ensure_ascii=False))


if __name__=='__main__':main()
