import pandas as pd
import numpy as np
from .strategy import Strategy
from .tradingstrategy import TradingStrategy


### HELP FUNCTIONS

def trading_session(d, market, strategy, prev_W, prev_W_hedge, capital, tc_func):
    W, W_hedge = strategy.get_weights(d)
    R, R_hedge = market.get_returns(d)
    begin_px, begin_px_hedge = market.get_begin_price(d)
    tc = tc_func(capital, [prev_W, prev_W_hedge], [W, W_hedge], [begin_px, begin_px_hedge])
    #print(type(R), type(W), type(W_hedge), type(R_hedge))
    ret = ((R if R is not None else 0) * W).sum() + W_hedge * (R_hedge if R_hedge is not None else 0)
    return ret, tc, W, W_hedge

def T_cost_share(capital, h0, h1, P, cost_per_share=0.005):
    '''
    h0: previous holdings
    h1: target holings

    list index : 
    - 0: stocks, 
    - 1: hedging_tool (eg: SPY)
    '''
    share_trade = (capital * (h0[0] - h1[0]).abs() / P[0]).sum() + (capital * abs(h0[1] - h1[1]) / P[1])
    tc = share_trade * cost_per_share
    return tc

def T_cost_ratio(capital, h0, h1, P, commission=0.001):
    '''
    h0: previous holdings
    h1: target holings
    
    list index : 
    - 0: stocks, 
    - 1: hedging_tool (eg: SPY)
    '''
    dollar_trade = capital * (abs(h0[0] - h1[0]) + abs(h0[1] - h1[1]))
    tc = dollar_trade * commission
    return tc

def snapshot_IC_BR(d, signal, ret):
    g = signal.loc[d, :]
    r = ret.loc[d, :]
    
    data = pd.DataFrame({'g': g, 'r': r}).dropna()
    return [data.corr().values[0][1], data.shape[0]]

def snapshot_IC_BR_combined(d, signal_1, ret_1, signal_2, ret_2):
    
    if d not in signal_1.index and d not in signal_2.index:
        return np.nan, np.nan
    elif d not in signal_1.index and d in signal_2.index:
        g = signal_2.loc[d, :]
        r = ret_2.loc[d, :]
    elif d in signal_1.index and d not in signal_2.index:
        g = signal_1.loc[d, :]
        r = ret_1.loc[d, :]
    else:
        g = signal_1.loc[d, :].append(signal_2.loc[d, :])
        r = ret_1.loc[d, :].append(ret_2.loc[d, :])
    
    data = pd.DataFrame({'g': g, 'r': r}).dropna()
    return [data.corr().values[0][1], data.shape[0]]
    
###

### child class

class IO_strategy(TradingStrategy):
    
    def __init__(self, trade_dates, w_intraday, w_overnight, hedge_method='None', hedge_ratio_intraday=1, hedge_ratio_overnight=1, beta=None):
        '''
        hedge_method: 'None', 'Dollar', 'Beta'
        '''
        if isinstance(w_intraday, pd.DataFrame):
            assert(set(trade_dates) >= set(w_intraday.index))
        if isinstance(w_overnight, pd.DataFrame):
            assert(set(trade_dates) >= set(w_overnight.index))
        
        TradingStrategy.__init__(self, trade_dates)
        self.intraday = Strategy(w_intraday, hedge_method, hedge_ratio_intraday, beta)
        self.overnight = Strategy(w_overnight, hedge_method, hedge_ratio_overnight, beta)
        
    
    def trade(self, mkt_intraday, mkt_overnight, initial_capital, tc_func=T_cost_share):
        
        self.result = pd.DataFrame(index=self.trade_dates, columns=['gross_return', 'capital'], dtype=float)
        
        d0 = self.trade_dates[0]
        self.result.loc[d0, 'capital'] = initial_capital
        self.result.loc[d0, 'gross_return'] = 0
        
        prev_W_hedge = 0
        Z_overnight = pd.Series(0, index=self.overnight.W.columns) if isinstance(self.overnight.W, pd.DataFrame) else 0
        Z_intraday = pd.Series(0, index=self.intraday.W.columns) if isinstance(self.intraday.W, pd.DataFrame) else 0
        prev_W = Z_intraday
        for d, d_prev in zip(self.trade_dates[1:], self.trade_dates[:-1]):
            
            capital = self.result.loc[d_prev, 'capital']
            
            # overnight trading
            if d not in self.overnight.W.index:
                o_ret, o_tc = 0, 0
                prev_W, prev_W_hedge = Z_overnight, 0
            else:
                o_ret, o_tc, prev_W, prev_W_hedge = trading_session(d, mkt_overnight, self.overnight, prev_W, prev_W_hedge, capital, tc_func)
                
            capital = capital * (1 + o_ret) - o_tc
            
            # intraday trading
            if d not in self.intraday.W.index:
                i_ret, i_tc = 0, 0
                prev_W, prev_W_hedge = Z_intraday, 0
            else:
                i_ret, i_tc, prev_W, prev_W_hedge = trading_session(d, mkt_intraday, self.intraday, prev_W, prev_W_hedge, capital, tc_func)
            
            capital = capital * (1 + i_ret) - i_tc
            
            total_return = (1 + o_ret) * (1 + i_ret) - 1
            self.result.loc[d, 'gross_return'] = total_return
            self.result.loc[d, 'capital'] = capital
        
        self.result['net_return'] = self.result['capital'].pct_change()
        self.result.loc[d0, 'net_return'] = self.result.loc[d0, 'capital'] / initial_capital - 1
        self.result['capital_pre_tc'] = (initial_capital * ((1 + self.result['gross_return']).cumprod()))
        
        self.result = self.result[['capital', 'capital_pre_tc', 'net_return', 'gross_return']]
        
        return
    
    

    def information_coefficient(self, mkt_intraday, mkt_overnight, return_BR=False):

        '''
        Assume that weights is proportional to signal strength
        '''

        overnight = pd.DataFrame([snapshot_IC_BR(d, self.overnight.W, mkt_overnight.ret) for d in self.overnight.W.index], index=self.overnight.W.index)#, columns=['IC', 'BR'])
        intraday = pd.DataFrame([snapshot_IC_BR(d, self.intraday.W, mkt_intraday.ret) for d in self.intraday.W.index], index=self.intraday.W.index)#, columns=['IC', 'BR'])
        combined = pd.DataFrame([snapshot_IC_BR_combined(d, self.overnight.W, mkt_overnight.ret, self.intraday.W, mkt_intraday.ret) for d in self.trade_dates], index=self.trade_dates)#, columns=['IC', 'BR'])
        
        IC_BR = pd.concat([overnight, intraday, combined], axis=1)
        
        IC_BR.columns = ['Overnight_IC', 'Overnight_BR','Intraday_IC', 'Intraday_BR', 'Combined_IC', 'Combined_BR']
        
        if return_BR:
            return IC_BR[['Overnight_IC', 'Intraday_IC', 'Combined_IC']], IC_BR[['Overnight_BR', 'Intraday_BR', 'Combined_BR']]
        else:
            return IC_BR[['Overnight_IC', 'Intraday_IC', 'Combined_IC']]
    
    def eval_beta(self, beta):
        
        res = pd.concat([self.intraday.eval_beta(beta), self.overnight.eval_beta(beta)], axis=1)
        res.columns = ['Intraday', 'Overnight']
        res = res.reindex(self.trade_dates).fillna(0)
        return res
    
    def eval_exposure(self):
        
        res = pd.concat([self.intraday.eval_exposure(), self.overnight.eval_exposure()], axis=1)
        res.columns = ['Intraday', 'Overnight']
        res = res.reindex(self.trade_dates).fillna(0)
        return res
    
    def __add__(self, other):
        
        new_dates = pd.to_datetime(sorted(set(self.trade_dates).union(other.trade_dates)))
        new = IO_strategy(new_dates, None, None)
        new.intraday = self.intraday + other.intraday
        new.overnight = self.overnight + other.overnight
        
        return new
    
    def __neg__(self):
        
        new = IO_strategy(self.trade_dates, None, None)
        new.intraday = -self.intraday
        new.overnight = -self.overnight
        
        return new 
    
    def __sub__(self, other):
        
        return self + (-other)
    
    def separate_io_result(self, mkt_intraday, mkt_overnight, initial_capital, tc_func=T_cost_share):
        '''
        return: intraday only, overnight only
        '''
        i_only = IO_strategy(self.trade_dates, None, None)
        i_only.intraday.W = self.intraday.W.copy()
        i_only.intraday.W_hedge = self.intraday.W_hedge.copy()
        i_only.overnight.W = self.overnight.W * 0
        i_only.overnight.W_hedge = self.overnight.W_hedge * 0
        o_only = self - i_only
        
        i_only.trade(mkt_intraday, mkt_overnight, initial_capital, tc_func=tc_func)
        o_only.trade(mkt_intraday, mkt_overnight, initial_capital, tc_func=tc_func)
        
        return i_only, o_only
        
    def separate_hedge_result(self, mkt_intraday, mkt_overnight, initial_capital, return_unhedge=False, tc_func=T_cost_share):
        '''
        return: spy part, (optional: unhedge part)
        '''
        unhedge = IO_strategy(self.trade_dates, self.intraday.W, self.overnight.W)
        hedge_only = self - unhedge
        hedge_only.trade(mkt_intraday, mkt_overnight, initial_capital, tc_func=tc_func)
        
        if return_unhedge:
            unhedge.trade(mkt_intraday, mkt_overnight, initial_capital, tc_func=tc_func)
            return hedge_only, unhedge
        else:
            return hedge_only

    def separate_ls_result(self, mkt_intraday, mkt_overnight, initial_capital, tc_func=T_cost_share):
        '''
        return: long only, short only, ignore all hedging
        '''
        l_only = IO_strategy(self.trade_dates, None, None)
        l_only.intraday.W = self.intraday.W.copy()
        l_only.intraday.W_hedge = self.intraday.W_hedge * 0
        l_only.overnight.W = self.overnight.W.copy()
        l_only.overnight.W_hedge = self.overnight.W_hedge * 0

        l_only.intraday.W.loc[l_only.intraday.W<0] = 0
        l_only.overnight.W.loc[l_only.overnight.W<0] = 0

        s_only = self - l_only
        
        l_only.trade(mkt_intraday, mkt_overnight, initial_capital, tc_func=tc_func)
        s_only.trade(mkt_intraday, mkt_overnight, initial_capital, tc_func=tc_func)
        
        return l_only, s_only