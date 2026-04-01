import numpy as np

def compute_rmse(x,y):
    # x = truth, y = computed
    rmse = x-y
    rmse = rmse**2
    rmse = np.sum(rmse)/len(x)
    rmse = np.sqrt(rmse)
    nrmse = rmse/ (max(x)-min(x))
    return nrmse

def interpolate_timeseries(data, target_len=100):
    n = data.shape[0]
    old = np.arange(n)
    new = np.linspace(0, n - 1, target_len)


    # qualora avessimo un solo angolo da plottare
    if len(data.shape) == 2:
        data_interp = np.empty((target_len, data.shape[1]))
        for i in range(data.shape[1]):
            data_interp[:, i] = np.interp(new, old, data[:, i])
    else:
        data_interp = np.interp(new, old, data)

    
    return data_interp, new

import casadi as ca

def LSE(weighted_objectives, t_LSE):
    max_obj = ca.mmax(weighted_objectives)

    # LSE anti overflow / stable (normalized)
    LSE = max_obj + (1/t_LSE)*ca.log(ca.sum(ca.exp(t_LSE*(weighted_objectives - max_obj))))

    return LSE


