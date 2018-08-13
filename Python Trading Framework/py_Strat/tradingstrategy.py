import pandas as pd
import statsmodels.api as sm
import abc

### parent class 

class TradingStrategy:
    __metaclass__  = abc.ABCMeta

    def __init__(self, trade_dates):
        self.result = None
        self.trade_dates = trade_dates

    @abc.abstractmethod
    def trade(self):
        '''trade the defined strategy in the market'''

    def metric_sharpe_ratio(self, period_mask=None):
        if isinstance(self.result, pd.DataFrame):
            ret = self.result if period_mask is None else self.result.loc[period_mask, :]

            SR_net = ret['net_return'].mean() / ret['net_return'].std() * (252) ** 0.5
            SR_gross = ret['gross_return'].mean() / ret['gross_return'].std() * (252) ** 0.5
            return {'SR_net': SR_net, 'SR_gross': SR_gross}
        else:
            return None

    def residual_returns(self, bmark, OneBeta=False):
        if isinstance(self.result, pd.DataFrame) and isinstance(bmark.result, pd.DataFrame):
            if not OneBeta:
                res_net = sm.OLS(self.result['net_return'], sm.add_constant(bmark.result['net_return'])).fit()
                b_net = float(res_net.params[-1])

                res_gross = sm.OLS(self.result['gross_return'], sm.add_constant(bmark.result['gross_return'])).fit()
                b_gross = float(res_gross.params[-1])
            else:
                b_net, b_gross = 1, 1

            res_net_ret = self.result['net_return'] - b_net * bmark.result['net_return']
            res_gross_ret = self.result['gross_return'] - b_gross * bmark.result['gross_return']

            return res_net_ret, res_gross_ret
        else:
            return None, None

    def metric_information_ratio(self, bmark, OneBeta=False, period_mask=None):
        if isinstance(self.result, pd.DataFrame) and isinstance(bmark.result, pd.DataFrame):

            res_net_ret, res_gross_ret = self.residual_returns(bmark, OneBeta)

            if period_mask is not None:
                res_net_ret = res_net_ret.loc[period_mask]
                res_gross_ret = res_gross_ret.loc[period_mask]

            IR_net = res_net_ret.mean() / res_net_ret.std() * (252) ** 0.5
            IR_gross = res_gross_ret.mean() / res_gross_ret.std() * (252) ** 0.5

            return {'IR_net': IR_net, 'IR_gross': IR_gross}
        else:
            return None

    def metric_maxDD(self, period_mask=None, net_return=True):
        if isinstance(self.result, pd.DataFrame):
            balance = self.result['capital'] if net_return else self.result['capital_pre_tc']
            if period_mask is not None:
                balance = balance.loc[period_mask]
            return min(balance / balance.cummax()) - 1
        else:
            return

    def result_to_csv(self, name=None, path='output/'):
        if isinstance(self.result, pd.DataFrame):
            self.result.to_csv(path + name + '.csv')
        return