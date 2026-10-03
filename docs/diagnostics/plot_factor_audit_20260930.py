"""Render the completed audit tables; no recalculation of research signals."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[2]
out=root/'outputs/factor_audit_figures_20260930_v1'
out.mkdir(exist_ok=False)
audit=root/'outputs/factor_information_audit_20260930_v2'
abl=pd.read_csv(root/'outputs/frozen_execution_ablation_20260930_v1/ablation_metrics.csv')
ic=pd.read_csv(audit/'signal_horizons.csv')
fig,axes=plt.subplots(1,3,figsize=(17,5.4),layout='constrained')
colors=['#215c98','#d28035']
names=['native5_alpha_unit','alpha_unit_target','target','position']
for period,color in zip(['2024','2025'],colors):
    g=ic[(ic.period==period)&(ic.horizon_5m==1)].set_index('signal')
    axes[0].plot(range(4),g.loc[names,'residual_pooled_ic'],marker='o',label=period+' Q1',color=color)
axes[0].set_xticks(range(4),['Native 5m\nalpha','Combined\nalpha','Profile\ntarget','Executed\nposition'])
axes[0].set_ylabel('Correlation with next 5m residual return')
axes[0].set_title('Information survives upstream')
axes[0].axhline(0,color='#999999',lw=.7);axes[0].legend()
g=abl[abl['case']=='original'].set_index('period')
x=np.arange(2);w=.24
for k,col in enumerate(['common_gross_log','relative_gross_log']):
    axes[1].bar(x+(k-.5)*w,g.loc[['2024','2025'],col]*100,width=w,label=['Common position','Relative position'][k],color=colors[k])
axes[1].set_xticks(x,['2024 Q1','2025 Q1']);axes[1].axhline(0,color='#999999',lw=.7)
axes[1].set_ylabel('Additive gross log-return contribution (%)')
axes[1].set_title('Execution is mainly directional');axes[1].legend()
cases=['original','remove_beta','remove_native5','risk_budget_one','min_hold_one_bar']
for k,(period,color) in enumerate(zip(['2024','2025'],colors)):
    g=abl[abl.period==period].set_index('case')
    axes[2].bar(np.arange(len(cases))+(k-.5)*.34,g.loc[cases,'net_return_pct'],width=.34,label=period+' Q1',color=color)
axes[2].set_xticks(range(len(cases)),['Original','No beta','No 5m','Risk=1','Min hold\n5m'],rotation=25,ha='right')
axes[2].axhline(0,color='#999999',lw=.7);axes[2].set_ylabel('Net return at 4 bps (%)')
axes[2].set_title('No single mechanical fix');axes[2].legend()
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.15)
fig.suptitle('Frozen v9 audit | descriptive historical evidence, not an optimized strategy',fontsize=14)
fig.savefig(out/'research_diagnosis.png',dpi=160)
fig.savefig(out/'research_diagnosis.svg')
print(out/'research_diagnosis.png')
