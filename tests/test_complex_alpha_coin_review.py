import numpy as np
import pandas as pd
import pytest
from scripts.complex_alpha_coin_review import wealth_contributions


def test_wealth_attribution_reconciles_compounding_and_period_capital():
    net=np.array([.1,-.05,.02,-.03])
    equity=np.r_[1,np.cumprod(1+net[:-1])]
    ledger=pd.DataFrame({'net':net,'equity_before_cashflows':equity,'period':['d','d','v','v']})
    pnl=pd.DataFrame({'a':net+.01,'b':net-.01})
    result=wealth_contributions(ledger,pnl)
    assert result.iloc[:2].to_numpy().sum()==pytest.approx(4.5)
    assert result.iloc[2:].to_numpy().sum()==pytest.approx(-1.06)
    assert result.iloc[0,0]==pytest.approx(5.5)
    assert result.iloc[1,1]==pytest.approx(-3.3)


def test_wealth_attribution_rejects_missing_inconsistent_or_invalid_capital():
    ledger=pd.DataFrame({'net':[.1],'equity_before_cashflows':[1.],'period':['d']})
    with pytest.raises(ValueError):wealth_contributions(ledger,pd.DataFrame({'a':[np.nan]}))
    with pytest.raises(AssertionError):wealth_contributions(ledger,pd.DataFrame({'a':[.2]}))
    ledger['equity_before_cashflows']=0
    with pytest.raises(ValueError):wealth_contributions(ledger,pd.DataFrame({'a':[.1]}))
