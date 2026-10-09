"""Matured forecasting skill states; never flip using current/future outcomes."""
import numpy as np
import pandas as pd
from .effective_combo import mature_window


def causal_skill_states(prediction,target,dates,periods,horizon,policy):
    calibrated={};scores={};records=[];stamp=pd.to_datetime(dates,format='%Y%m%d%H%M',utc=True)
    for months in policy['windows_months']:
        cal=np.full_like(target,np.nan);score=np.full_like(target,np.nan)
        for start,end in periods:
            boundary=pd.to_datetime(start,format='%Y%m%d%H%M',utc=True);stop=pd.to_datetime(end,format='%Y%m%d%H%M',utc=True)
            while boundary<=stop:
                key=boundary.strftime('%Y%m%d%H%M');next_time=boundary+pd.DateOffset(months=1);past=mature_window(dates,key,horizon,months)
                p=prediction[past].ravel();y=target[past].ravel();good=np.isfinite(p)&np.isfinite(y)&(abs(p)>policy['training_floor_bps']/1e4)
                n=int(good.sum());gamma=float(np.clip(np.dot(p[good],np.clip(y[good],-.2,.2))/max(np.dot(p[good],p[good]),1e-16),-1,1)) if n>=policy['minimum_labels'] else 1.
                test=(stamp>=boundary)&(stamp<next_time)&(dates<=end);cal[test]=gamma*prediction[test]
                score[test]=(np.sign(gamma) if abs(gamma)>=policy['skill_deadzone'] else 0.)*prediction[test]
                records.append({'window_months':months,'update':key,'mature_labels':n,'gamma':gamma,
                    'last_eligible_origin':str(dates[past][-1]),'role':'matured direct prediction calibration; signed score separately declared'})
                boundary=next_time
        calibrated[months]=cal;scores[months]=score
    return calibrated,scores,records


def smooth_targets(target,span,deadband):
    if span<1 or deadband<0:raise ValueError('invalid target smoothing')
    proposal=pd.DataFrame(target).ewm(span=span,adjust=False).mean().to_numpy();result=np.zeros_like(target);held=np.zeros(target.shape[1])
    for i,row in enumerate(proposal):
        update=np.abs(row-held)>=deadband;held[update]=row[update];result[i]=held
    return result
