"""Read-only temporal uncertainty and delivery QA for completed R2 evidence.

No model fit, candidate selection, direction change or portfolio alteration.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,hashlib,re,shutil,yaml
from urllib.parse import unquote
import numpy as np
import pandas as pd


def daily_block_diagnostics(matrix,seed=20261009,trials=1000,block=7):
    """Joint circular blocks preserve within-week co-movement across strategies.

    Conditional historical resampling only: no model retrain, selection correction
    or claim that the return process is stationary. Daily sums match Sharpe units.
    """
    if matrix.ndim!=2 or len(matrix)<30 or not np.isfinite(matrix).all():raise ValueError('finite daily matrix with >=30 days required')
    rng=np.random.default_rng(seed);n=len(matrix)
    starts=rng.integers(0,n,size=(trials,int(np.ceil(n/block))))
    index=((starts[...,None]+np.arange(block))%n).reshape(trials,-1)[:,:n]
    samples=matrix[index];mean=samples.mean(axis=1);sd=samples.std(axis=1)
    sharpe=np.divide(mean,sd,out=np.zeros_like(mean),where=sd>0)*np.sqrt(365.25)
    return dict(sharpe_low=np.quantile(sharpe,.05,axis=0),sharpe_high=np.quantile(sharpe,.95,axis=0),
        positive_mean_fraction=(mean>0).mean(axis=0),mean_low=np.quantile(mean,.05,axis=0),mean_high=np.quantile(mean,.95,axis=0))


def review(history,oos,out):
    history=Path(history);oos=Path(oos);out=Path(out)
    if out.exists():raise ValueError('new immutable review directory required')
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    for p in [history,oos]:
        if load(p/'manifest.json')['status']!='COMPLETED':raise ValueError('completed source required')
    if (history/'protocol_freeze.json').read_bytes()!=(oos/'protocol_freeze.json').read_bytes():raise ValueError('protocol freeze changed')
    if (history/'selection_freeze.json').read_bytes()!=(oos/'selection_freeze.json').read_bytes():raise ValueError('recommendation changed')
    tracked=[p for root in [history,oos] for p in root.rglob('*') if p.is_file()]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    out.mkdir(parents=True);shutil.copyfile(__file__,out/'review_source.py');confidence=[];paired=[];qa=[]
    for root,period in [(history,'validation'),(oos,'oos')]:
        stats=pd.read_csv(root/'portfolio_metrics.csv');keys=sorted(stats.loc[(stats.period==period)&stats.id.str.endswith('__edge16'),'id'])
        daily=[]
        for k in keys:
            f=pd.read_parquet(root/(k+'_'+period+'_ledger.parquet'));stamp=pd.to_datetime(f.completed_5m,format='%Y%m%d%H%M',utc=True)
            daily.append(pd.Series(f.net.to_numpy(),index=stamp).resample('D').sum())
            check=float(abs(f.price_gross+f.funding-f.fee-f.net).max())
            path=np.load(root/(k+'_'+period+'_asset_path.npz'));forward=np.zeros_like(path['price']);forward[:-1]=path['price'][1:]/path['price'][:-1]-1
            pricecheck=float(abs(np.mean(path['position']*forward,axis=1)-f.price_gross).max())
            contrib=pd.read_csv(root/'asset_contribution.csv');capitalcheck=float(abs(contrib.loc[contrib.id==k,'wealth_contribution_pct'].sum()-stats.loc[(stats.id==k)&(stats.period==period),'net_return_pct'].iloc[0]))
            qa.append({'source':str(root),'id':k,'ledger_identity_error':check,'price_identity_error':pricecheck,'wealth_contribution_error_pp':capitalcheck})
            if max(check,pricecheck)>1e-12 or capitalcheck>1e-7:raise ValueError('accounting identity mismatch')
        frame=pd.concat(daily,axis=1);frame.columns=keys
        if frame.isna().any().any():raise ValueError('different strategy clock')
        diag=daily_block_diagnostics(frame.to_numpy())
        for j,k in enumerate(keys):confidence.append({'id':k,'period':period,'days':len(frame),'block_days':7,'trials':1000,**{name:float(values[j]) for name,values in diag.items()}})
        base=frame['base_ensemble__edge16'].to_numpy()
        for k in keys:
            diff=frame[k].to_numpy()-base;result=daily_block_diagnostics(diff[:,None])
            paired.append({'id':k,'period':period,'vs':'base_ensemble__edge16','mean_daily_difference_bps':float(diff.mean()*1e4),
                'difference_mean_low_bps':float(result['mean_low'][0]*1e4),'difference_mean_high_bps':float(result['mean_high'][0]*1e4),
                'positive_difference_fraction':float(result['positive_mean_fraction'][0])})
    pd.DataFrame(confidence).to_csv(out/'temporal_uncertainty.csv',index=False);pd.DataFrame(paired).to_csv(out/'paired_daily_differences.csv',index=False)
    pd.DataFrame(qa).to_csv(out/'accounting_checks.csv',index=False)
    visual=[]
    for experiment in [history,oos]:
        root=Path(yaml.safe_load((experiment/'config.yaml').read_text(encoding='utf-8'))['output']['visualization_dir']);meta=load(root/'visualization_manifest.json');missing=[];links=0
        for name in meta['pages']:
            p=root/name
            for href in re.findall(r'(?:href|src)=[\"\x27]([^\"\x27]+)',p.read_text(encoding='utf-8')):
                if href.startswith(('http:','https:','data:','#')):continue
                links+=1
                if not (p.parent/unquote(href.split('#')[0])).exists():missing.append(str(p)+' -> '+href)
        if missing:raise ValueError('missing report links: '+str(missing[:5]))
        visual.append({'root':str(root),'pages':meta['page_count'],'local_links':links,'missing':missing,'automatic_selection':meta['automatic_selection']})
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):raise ValueError('read-only review changed evidence')
    (out/'source_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps({'status':'COMPLETED','role':'read-only conditional temporal uncertainty and QA',
        'sources':[str(history),str(oos)],'files_unchanged':len(hashes),'selection_identical':True,'protocol_identical':True,
        'visualizations':visual,'bootstrap_scope':'conditional circular seven-day blocks; no selection/multiple-testing correction or future guarantee'},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(visual,ensure_ascii=False));print(pd.DataFrame(confidence).to_string(index=False))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--history',required=True);p.add_argument('--oos',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();review(a.history,a.oos,a.output)
