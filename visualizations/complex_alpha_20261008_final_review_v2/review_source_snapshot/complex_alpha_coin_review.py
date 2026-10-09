"""Display saved ledgers only; no fitting, target construction or selection."""
from __future__ import annotations
import html,json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.alpha_mvp.research.factor_combo_reporting import _CSS,_plot,_table


def wealth_contributions(ledger,asset_pnl):
    """Additive contributions to period wealth, rather than fake leg NAVs."""
    values=asset_pnl.to_numpy(dtype=float)
    equity=ledger.equity_before_cashflows.to_numpy(dtype=float)
    if len(values)!=len(ledger) or not len(values) or values.shape[1]<1:
        raise ValueError('invalid asset ledger shape')
    if not np.isfinite(values).all() or not np.isfinite(equity).all() or np.any(equity<=0):
        raise ValueError('invalid asset ledger values or capital')
    np.testing.assert_allclose(values.mean(axis=1),ledger.net.to_numpy(),atol=1e-13)
    result=np.zeros_like(values)
    for period in ledger.period.unique():
        mask=ledger.period.eq(period).to_numpy()
        result[mask]=values[mask]*equity[mask,None]/equity[mask][0]/values.shape[1]*100
        expected=(np.prod(1+ledger.net.to_numpy()[mask])-1)*100
        np.testing.assert_allclose(result[mask].sum(),expected,atol=2e-9)
    return pd.DataFrame(result,columns=asset_pnl.columns)


def render_coin_review(sources,out):
    out=Path(out);records=[];legacy=[];pages=[]
    for source in map(Path,sources):
        freeze=json.loads((source/'selection_freeze.json').read_text(encoding='utf-8'))
        if not freeze['selected']:continue
        ledger=pd.read_parquet(source/'combo_5m_ledger.parquet')
        pnl=pd.read_parquet(source/'combo_asset_pnl.parquet')
        contrib=wealth_contributions(ledger,pnl)
        dates=pd.to_datetime(ledger.completed_5m,format='%Y%m%d%H%M',utc=True)
        label=source.name.rsplit('_',1)[-1]
        legacy.append((label,ledger,contrib,dates))
        for period in ledger.period.unique():
            mask=ledger.period.eq(period).to_numpy()
            for code in pnl.columns:records.append({'round':label,'asset':code,'period':period,
                'period_wealth_contribution_pp':float(contrib.loc[mask,code].sum())})
    final=Path(sources[-1]);cfg=pd.read_csv(out/'all_frozen_asset_contributions.csv') if (out/'all_frozen_asset_contributions.csv').exists() else pd.DataFrame()
    assets=sorted(set(cfg.asset if len(cfg) else []).union(r['asset'] for r in records))
    if not assets:return []
    summary=pd.DataFrame(records)
    summary.to_csv(out/'selected_asset_wealth_contributions.csv',index=False)
    links=' · '.join(f'<a href="coin_{html.escape(code)}.html">{html.escape(code)}</a>' for code in assets)
    note='<p>历史入选集合取原保存账本：逐币曲线是该腿对整篮子分段财富的加性贡献（百分点），不是单币独立交易净值。各币贡献之和与该段组合复利收益精确对账。最终轮全部固定通道的逐币柱状值是简单收益贡献之和，二者口径分开展示；对冲腿亏损不能单独否定Alpha。测试已见且仅冻结后审计，这些页面不改变选择。</p>'
    def page(name,title,body):
        path=out/name
        if path.exists():raise ValueError('coin review page already exists')
        path.write_text(f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{_CSS}</style><script src='plotly.min.js'></script></head><body><nav><a href='coins.html'>逐币归因</a></nav><main><h1>{html.escape(title)}</h1>{note}<p>{links}</p>{body}</main></body></html>",encoding='utf-8')
        pages.append(name)
    page('coins.html','逐币贡献与历史联合候选',_table(summary)+('<h2>最终冻结源：'+html.escape(final.name)+'</h2>'+_table(cfg) if len(cfg) else ''))
    for code in assets:
        body=''
        for label,ledger,contrib,dates in legacy:
            if code not in contrib:continue
            fig=go.Figure();monthly=[]
            for period in ledger.period.unique():
                mask=ledger.period.eq(period).to_numpy()
                daily=pd.Series(contrib.loc[mask,code].to_numpy(),index=dates[mask]).resample('1D').sum()
                fig.add_trace(go.Scatter(x=daily.index,y=daily.cumsum(),mode='lines',name=label+'/'+period))
                months=pd.Series(contrib.loc[mask,code].to_numpy(),index=dates[mask]).resample('MS').sum()
                monthly.extend({'round':label,'period':period,'month':str(t.date()),'wealth_contribution_pp':float(v)} for t,v in months.items())
            fig.update_yaxes(title='对该段整篮子财富的加性贡献（百分点）')
            body+=_plot(fig,label+'原冻结集合：逐币财富贡献')+_table(pd.DataFrame(monthly))
        if len(cfg):
            frame=cfg[cfg.asset.eq(code)];fig=go.Figure()
            for period,g in frame.groupby('period',sort=False):fig.add_trace(go.Bar(x=g.id,y=g.net_contribution_sum_pct,name=period))
            fig.update_layout(barmode='group',xaxis_tickangle=-55)
            fig.update_yaxes(title='分期简单净收益贡献之和（百分点），不是复利财富')
            body+=_plot(fig,'最终轮全部固定通道：包含失败与零交易')+_table(frame)
        page('coin_'+code+'.html',code+'：组合腿贡献',body)
    return pages
