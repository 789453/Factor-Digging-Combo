"""Read-only v2: add surviving products' actual per-asset paths to the full review."""
from pathlib import Path
import json,shutil,re,html
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from src.alpha_mvp.research.factor_combo_reporting import _CSS,_plot,_table


def build():
    old=Path('visualizations/effective_combo_20261009_extended_final_review_v1');out=Path('visualizations/effective_combo_20261009_extended_final_review_v2')
    qa=Path('outputs/effective_combo_20261009_extended_delivery_review_v2')
    if out.exists() or qa.exists():raise ValueError('new read-only display/QA required')
    shutil.copytree(old,out);qa.mkdir()
    for name in ['core_temporal_uncertainty.csv','core_long_short_decomposition.csv']:
        shutil.copyfile(Path('outputs/effective_combo_20261009_extended_delivery_review_v1')/name,qa/name)
    roots=[(Path('outputs/effective_combo_mature_state_20261009_r4_history'),'validation'),(Path('outputs/effective_combo_mature_state_20261009_r4_oos'),'oos')]
    names=['state_cal1__edge8','state_cal1__edge16'];codes=json.loads((roots[0][0]/'feature_contract.json').read_text())['assets']
    tables={};frames={};paths={};predictions={};clocks={};targets={}
    for root,period in roots:
        clocks[period]=np.load(root/'dates.npy');targets[period]=np.load(root/'target.npy')
        predictions[period]={k:np.load(root/(k+'_prediction.npy')) for k in ['base15','pair_prediction','state_cal1']}
        tables[period]=pd.read_csv(root/'asset_contribution.csv')
        for name in names:
            frames[period,name]=pd.read_parquet(root/(name+'_'+period+'_ledger.parquet'))
            paths[period,name]=np.load(root/(name+'_'+period+'_asset_path.npz'))
    nav='<nav><a href="../index.html">研究总览</a> · <a href="../assets.html">十二币</a> · <a href="../states.html">成熟系数</a> · <a href="../calibration_probes.html">固定探针</a> · <a href="../costs.html">成本与暴露</a></nav>'
    for j,code in enumerate(codes):
        source=out/'assets'/(code+'.html');source.rename(source.with_name(code+'_state3_original.html'));body=''
        for root,period in roots:
            p=paths[period,names[0]];fast=p['dates'];stamp=pd.to_datetime(fast,format='%Y%m%d%H%M');ii=np.unique(np.r_[np.arange(0,len(fast),12),len(fast)-1])
            dates=clocks[period];dm=(dates>=str(fast[0]))&(dates<=str(fast[-1]));dd=pd.to_datetime(dates[dm],format='%Y%m%d%H%M')
            fig=make_subplots(rows=4,cols=1,shared_xaxes=True,subplot_titles=['完成价格','4h原预测、成熟幅度预测和实现标签（bps）','实际持仓：两套固定映射','对组合净财富的累计贡献（百分点）'])
            fig.add_trace(go.Scatter(x=stamp[ii],y=p['price'][ii,j],name='completed price'),row=1,col=1)
            fig.add_trace(go.Scatter(x=dd,y=targets[period][dm,j]*1e4,name='realized 4h label (ex post)',opacity=.15),row=2,col=1)
            for name,value in predictions[period].items():fig.add_trace(go.Scatter(x=dd,y=value[dm,j]*1e4,name=name+' forecast'),row=2,col=1)
            for name in names:
                path=paths[period,name];f=frames[period,name];capital=np.r_[1,np.cumprod(1+f.net.to_numpy())[:-1]]
                contribution=np.cumsum(capital*path['net'][:,j]/len(codes))*100
                fig.add_trace(go.Scatter(x=stamp[ii],y=path['position'][ii,j],name=name+' actual position'),row=3,col=1)
                fig.add_trace(go.Scatter(x=stamp[ii],y=contribution[ii],name=name+' wealth contribution'),row=4,col=1)
            fig.update_layout(height=1100)
            selected=tables[period].loc[(tables[period].asset==code)&tables[period].id.isin(names)]
            body+='<h2>'+period+'</h2>'+_plot(fig,code+' '+period+'价格/预测/持仓/真实组合贡献',height=1100)+_table(selected,rows=len(selected))
        body+='<p>币种分量诊断复利列不是独立币资金账户；图中累计贡献按真实组合资本加权，可按币求和为组合总净财富。原三个月主协议的逐币失败图仍保存：<a href="'+code+'_state3_original.html">原主协议完整路径</a>。</p>'
        source.write_text(f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>{html.escape(code)} 存活分支与原基线</title><style>{_CSS}</style><script src="../plotly.min.js"></script></head><body>{nav}<main><h1>{html.escape(code)}：预声明一月幅度分支的完整双期路径</h1><p>4h完成预测，之后5m份额账本；单边4bps，终点收费清仓；OOS为已见历史复核。所有量纲分开，实际标签为事后显示。原推荐与主协议未改。</p>{body}</main></body></html>',encoding='utf-8')
    # Clarify the only comparison that changes both update frequency and window.
    for p in out.rglob('*.html'):
        text=p.read_text(encoding='utf-8').replace('同容量静态/月更','同容量静态/月更（训练窗口也不同）')
        p.write_text(text,encoding='utf-8')
    pages=sorted(str(p.relative_to(out)).replace('\\','/') for p in out.rglob('*.html'));missing=[];count=0
    for name in pages:
        p=out/name
        for href in re.findall(r'(?:href|src)=[\"\x27]([^\"\x27]+)',p.read_text(encoding='utf-8')):
            if href.startswith(('http:','https:','data:','#')):continue
            count+=1
            if not (p.parent/href.split('#')[0]).exists():missing.append(name+' -> '+href)
    if missing:raise ValueError('v2 missing links '+str(missing[:5]))
    meta=json.loads((old/'visualization_manifest.json').read_text());meta.update(pages=pages,page_count=len(pages),local_links=count,missing_links=missing,
        asset_paths='both validation and OOS, core calibrated forecasts/actual positions/additive wealth contribution; original state3 preserved')
    (out/'visualization_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    (qa/'delivery_qa.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    (qa/'manifest.json').write_text(json.dumps({'status':'COMPLETED','role':'read-only full dual-period core asset paths and delivery QA','page_count':len(pages),'missing_links':missing},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'pages':len(pages),'links':count,'missing':missing}))


if __name__=='__main__':build()
