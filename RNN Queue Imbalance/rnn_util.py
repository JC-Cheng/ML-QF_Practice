import pandas as pd
import numpy as np


def process_data(raw_df):
    temp = pd.pivot_table(raw_df, values=['SIZE', 'PRICE'], 
                          index='Time', columns=['BUY_SELL_FLAG', 'LEVEL'])
    
    df = pd.DataFrame(temp['SIZE'].values, 
                      columns=['ask_1', 'ask_2', 'ask_3', 'bid_1', 'bid_2', 'bid_3'],
                      index=temp.index)
    
    df['a'] = temp['PRICE', 'ask', 1]
    df['b'] = temp['PRICE', 'bid', 1]
    
    df['a_move'] = np.sign(df['a'].shift(-1) - df['a'])
    df['b_move'] = np.sign(df['b'].shift(-1) - df['b'])
    
    df.dropna(inplace=True)
    
    LOB = df[['bid_3', 'bid_2', 'bid_1', 'ask_1', 'ask_2', 'ask_3']].copy()
    # label: {-1, 0, 1} -> {0, 1, 2}
    label = df[['b_move', 'a_move']].copy() + 1 
    
    return LOB, label

def create_LOB(path='./Equity_PRL_lv3.csv', tickers=['CS', 'TM']):
    
    raw = pd.read_csv(path)
    LOB_raw = {}
    LOB = {}
    for t in tickers:
        df = raw[[c for c in raw.columns if t in c]].copy()
        df.columns = [c.split('::')[1].replace(t + '.', '').split('.')[0] for c in df.columns]
        df.loc[:, 'BUY_SELL_FLAG'] = df['BUY_SELL_FLAG'].map({0: 'bid', 1: 'ask'})
        df.index = pd.to_datetime(raw['Time'])
        
        LOB[t] = process_data(df)

    return LOB

def get_one_hot(y, C):
    
    S = sorted(set(y))
    assert(len(S) <= C)
    m = {val: i for i, val in enumerate(S)}
    
    return np.eye(C)[[m[val] for val in y.reshape(-1)]]

def extract_RNN_data_from_LOB(LOB, label, n_lag=6):
    
    M, P = LOB.shape
    ts = label.index[n_lag:].tolist()
    
    X = np.zeros((M - n_lag, n_lag, P))
    
    for i in range(n_lag):
        X[:, i, :] = LOB.shift(i).values[n_lag:, :]
    
    Y = []
    for i in range(label.shape[1]):
        Y.append(get_one_hot(label.values[n_lag:, i], 3))
    
    return X, Y, ts

def prepare_RNN_data(LOB, tickers, n_lags=6):
    
    X, Y, TS = [], [[], []], []
    for t in tickers:
        x, y, ts = extract_RNN_data_from_LOB(LOB[t][0], LOB[t][1], n_lags)
        X.append(x)
        Y[0].append(y[0])
        Y[1].append(y[1])
        TS.extend(ts)
    
    X = np.concatenate(X, axis=0)
    Y = [np.concatenate(y, axis=0) for y in Y]
    
    return X, Y, TS

def split_and_suffle(X, Y, ts, split_ratio=0.2):
    
    N = len(ts)
    idx = list(range(N))
    np.random.shuffle(idx)
    
    train_idx = idx[:-int(N * split_ratio)]
    test_idx = idx[-int(N * split_ratio):]
    
    X_train, X_test = X[train_idx, :, :], X[test_idx, :, :]
    Y_train = [y[train_idx, :] for y in Y]
    Y_test = [y[test_idx, :] for y in Y]
    
    return X_train, Y_train, X_test, Y_test
    