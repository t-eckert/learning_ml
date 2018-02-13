import statsmodels.formula.api as sm

def backward_elim(x, sl):
  nvariables = len(x[0])
  for i in range(nvariables):
    regressor_ols = sm.OLS(y,x).fit()