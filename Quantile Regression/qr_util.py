import pandas as pd
import numpy as np

import pandas_datareader as pdr
import cvxpy as cvx

def build_data(tickers, start, end, tiingo_key, FF='F-F_Research_Data_Factors_daily', add_const=True):

	# load stock price
	px = pdr.DataReader(tickers, data_source='tiingo', start=start, end=end, access_key=tiingo_key)
	close_px = pd.concat([px.loc[ticker, 'adjClose'] for ticker in tickers], axis=1)
	close_px.columns = tickers
	ret = close_px.pct_change()

	# load factor returns
	FF = pdr.DataReader(FF, 'famafrench', start=start, end=end)
	FF[0] = FF[0] / 100
	rm = FF[0]['Mkt-RF']
	rf = FF[0]['RF']

	# create data: calculate excess returns
	re = pd.concat([FF[0]] + [ret[ticker] - rf for ticker in tickers], axis=1)
	re.columns = FF[0].columns.tolist() + tickers
	re.dropna(inplace=True)

	if add_const: re['Const'] = 1

	return re[tickers].copy(), re[(['Const'] if add_const else []) + ['Mkt-RF', 'SMB', 'HML']].copy()

def quantile_reg(Y, X, quantile=0.5, lamb=0.01):
    
    y, x = np.array(Y), np.array(X)
    T, K = X.shape
    
    b = cvx.Variable(K)
    u = y - x * b
    
    J = (quantile * cvx.sum_entries(cvx.pos(u)) + (1 - quantile) * cvx.sum_entries(cvx.neg(u))) / T
    L2 = cvx.norm(b) ** 2
    
    prob = cvx.Problem(cvx.Minimize(J + lamb * L2))
    prob.solve()
    
    return np.asarray(b.value).reshape(-1)

def pred(X, b):
    return X.dot(b)

def error(Y, Y_pred, quantile):
    u = Y - Y_pred
    return (quantile * u[u > 0].sum() + (quantile - 1) * u[u < 0].sum()) / Y.shape[0]

def quantile(Y, Y_pred):
	# Y: np array
	# Y_pred: np array, int or float
    return (Y < Y_pred).sum() / Y.shape[0]

def time_serires_split(ix, split_ratio=[0.8, 0.2], embargo_len=0):

    n_split = len(split_ratio)
    T = len(ix)
    split = np.array(split_ratio) * (1 - embargo_len * (n_split - 1) / T) / sum(split_ratio)
    idx = 0
    
    partition = []
    for i, r in enumerate(split):
        if i != n_split - 1:
            segment = int(T * r)
            partition.append(ix[idx: idx + segment])
            idx = idx + segment + embargo_len
        else:
            partition.append(ix[idx:])

    return partition