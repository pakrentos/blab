from blab.itertools import true_meshgrid
import numpy as np
import pandas as pd

def get_MANOVA_table(data, sub_f, factors, var_names):
    if len(data.shape) != sub_f + 2:
        data = data.copy()[..., None]
    assert data.shape[-1] == len(var_names)
    assert sub_f == len(factors.keys())
    for i, v in enumerate(factors.values()):
        assert data.shape[i] == len(v)
    
    sub_n = data.shape[sub_f]
    subs = np.arange(1, sub_n + 1, dtype=int).tolist()
    factors_raw = true_meshgrid(*([subs] + [v for v in factors.values()])).tolist()
    factor_names = list(factors.keys())
    cols = ['Subject'] + factor_names + var_names
    
    temp = np.rollaxis(data, sub_f).reshape(-1, len(var_names)).tolist()
    
    data_raw = [a + b for a,b in zip(factors_raw, temp)]
    res_df = pd.DataFrame(data_raw, columns=cols)
    return res_df

# def concat_factor_names(factors, sep='_'):
#     return [sep.join(i) for i in true_meshgrid(*factors)]

def get_RM_ANOVA_table(data, sub_f, factors):
    temp = np.rollaxis(data, sub_f).reshape(data.shape[sub_f], -1)
    cols = factors
    print(cols)
    return pd.DataFrame(temp, columns=cols)