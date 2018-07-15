import pandas as pd
from .strategy import Strategy


### HELP FUNCTIONS

def trading_session(d, market, strategy, prev_W, prev_W_hedge, capital, tc_func):
    W, W_hedge = strategy.get_weights(d)
    R, R_hedge = market.get_returns(d)
    begin_px, begin_px_hedge = market.get_begin_price(d)
    tc = tc_func(capital, [prev_W, prev_W_hedge], [W, W_hedge], [begin_px, begin_px_hedge])
    ret = (R * W).sum() + W_hedge * R_hedge
    
    return ret, tc, W, W_hedge

def T_cost(capital, h0, h1, P, cost_per_share=0.005):
    '''
    0: stocks, 1: hedging_tool (SPY)
    '''
    share_trade = (capital * (h0[0] - h1[0]).abs() / P[0]).sum() + (capital * abs(h0[1] - h1[1]) / P[1])
    tc = share_trade * cost_per_share
    
    return tc
    
###

class IO_strategy:
    
    def __init__(self, trade_dates, w_intraday, w_overnight, hedge_method='None', hedge_ratio=1, beta=None):
        '''
        hedge_method: 'None', 'Dollar', 'Beta'
        '''
        if isinstance(w_intraday, pd.DataFrame):
            assert(set(trade_dates) >= set(w_intraday.index))
        if isinstance(w_overnight, pd.DataFrame):
            assert(set(trade_dates) >= set(w_overnight.index))
        
        self.intraday = Strategy(w_intraday, hedge_method, hedge_ratio, beta)
        self.overnight = Strategy(w_overnight, hedge_method, hedge_ratio, beta)
        self.trade_dates = trade_dates
        
        self.result = None
    
    def trade(self, mkt_intrady, mkt_overnight, initial_capital, tc_func=T_cost):
        
        self.result = pd.DataFrame(index=self.trade_dates, columns=['gross_return', 'capital'], dtype=float)
        
        d0 = self.trade_dates[0]
        self.result.loc[d0, 'capital'] = initial_capital
        self.result.loc[d0, 'gross_return'] = 0
        
        prev_W_hedge = 0
        Z_overnight = pd.Series(0, index=self.overnight.W.columns)
        Z_intraday = pd.Series(0, index=self.intraday.W.columns)
        prev_W = Z_intraday
        for d, d_prev in zip(self.trade_dates[1:], self.trade_dates[:-1]):
            
            capital = self.result.loc[d_prev, 'capital']
            
            # overnight trading
            if d not in self.overnight.W.index:
                o_ret, o_tc = 0, 0
                prev_W, prev_W_hedge = Z_overnight, 0
            else:
                o_ret, o_tc, prev_W, prev_W_hedge = trading_session(d, mkt_overnight, self.overnight, prev_W, prev_W_hedge, capital, tc_func)
                #prev_W, prev_W_hedge = W, W_hedge
                
            capital = capital * (1 + o_ret) - o_tc
            
            # intraday trading
            if d not in self.intraday.W.index:
                i_ret, i_tc = 0, 0
                prev_W, prev_W_hedge = Z_intraday, 0
            else:
                i_ret, i_tc, prev_W, prev_W_hedge = trading_session(d, mkt_intrady, self.intraday, prev_W, prev_W_hedge, capital, tc_func)
                #i_tc = tc_func(capital, [prev_W, prev_W_hedge], [W, W_hedge], [begin_px, begin_px_hedge])
                #i_ret = (R * W).sum() + W_hedge * R_hedge
                #prev_W, prev_W_hedge = W, W_hedge
            
            capital = capital * (1 + i_ret) - i_tc
            
            total_return = (1 + o_ret) * (1 + i_ret) - 1
            self.result.loc[d, 'gross_return'] = total_return
            self.result.loc[d, 'capital'] = capital
        
        self.result['net_return'] = self.result['capital'].pct_change()
        self.result.loc[d0, 'net_return'] = self.result.loc[d0, 'capital'] / initial_capital - 1
        self.result['capital_pre_tc'] = (initial_capital * ((1 + self.result['gross_return']).cumprod()))
        
        self.result = self.result[['capital', 'capital_pre_tc', 'net_return', 'gross_return']]
        
        return
    
    def report_sharpe_ratio(self):
        
        if isinstance(self.result, pd.DataFrame):
            SR_net = self.result['net_return'].mean() / self.result['net_return'].std() * (252) ** 0.5
            SR_gross = self.result['gross_return'].mean() / self.result['gross_return'].std() * (252) ** 0.5
            return {'SR_net': SR_net, 'SR_gross': SR_gross}
        else:
            return None
        
    def report_information_ratio(self, bmark):
        if isinstance(self.result, pd.DataFrame) and isinstance(bmark.result, pd.DataFrame):
            pass
        else:
            return None
        
    def save_dataframe(self, name=None, path='output/'):
        if isinstance(self.result, pd.DataFrame):
            self.result.to_csv(path + name + '.csv')
        return
    
    def eval_beta(self, beta):
        
        res = pd.concat([self.intraday.eval_beta(beta), self.overnight.eval_beta(beta)], axis=1)
        res.columns = ['Intraday', 'Overnight']
        res = res.reindex(self.trade_dates).fillna(0)
        return res
    
    def eval_exposure(self, beta):
        
        res = pd.concat([self.intraday.eval_exposure(beta), self.overnight.eval_exposure(beta)], axis=1)
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
    
    def separate_io_result(self, mkt_intrady, mkt_overnight, initial_capital):
        '''
        return: intraday only, overnight only
        '''
        i_only = IO_strategy(self.trade_dates, None, None)
        i_only.intraday.W = self.intraday.W.copy()
        i_only.intraday.W_hedge = self.intraday.W_hedge.copy()
        i_only.overnight.W = self.overnight.W * 0
        i_only.overnight.W_hedge = self.overnight.W_hedge * 0
        o_only = self - i_only
        
        i_only.trade(mkt_intrady, mkt_overnight, initial_capital)
        o_only.trade(mkt_intrady, mkt_overnight, initial_capital)
        
        return i_only, o_only
        
    def separate_hedge_result(self, mkt_intrady, mkt_overnight, initial_capital, return_unhedge=False):
        '''
        return: spy part, (optional: unhedge part)
        '''
        unhedge = IO_strategy(self.trade_dates, self.intraday.W, self.overnight.W)
        hedge_only = self - unhedge
        hedge_only.trade(mkt_intrady, mkt_overnight, initial_capital)
        
        if return_unhedge:
            unhedge.trade(mkt_intrady, mkt_overnight, initial_capital)
            return hedge_only, unhedge
        else:
            return hedge_only