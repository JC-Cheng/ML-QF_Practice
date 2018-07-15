import pandas as pd
#import numpy as np

class Strategy:
    
    def __init__(self, weights=None, hedge_method='None', hedge_ratio=1, beta=None):
        
        if isinstance(weights, pd.DataFrame):
            self.W = weights.copy()
            if hedge_method == 'Dollar':
                self.W_hedge = - self.W.sum(axis=1) * hedge_ratio
            elif hedge_method == 'Beta':
                self.W_hedge = - self.W.multiply(beta.loc[weights.index]).sum(axis=1) * hedge_ratio
            else:
                self.W_hedge = pd.Series(0, index=weights.index)
        else:
            self.W = None
            self.W_hedge = None         
            
    def get_weights(self, date):
        if date in self.W.index:
            return self.W.loc[date, :], self.W_hedge.loc[date]
        else:
            return pd.Series(0, index=self.W.columns), 0
            
    def eval_beta(self, beta):
        
        return (self.W * beta.loc[self.W.index]).sum(axis=1) + self.W_hedge
    
    def eval_exposure(self):
        
        return self.W.sum(axis=1) + self.W_hedge
    
    def __add__(self, other):
        
        new = Strategy(self.W + other.W)
        new.W_hedge = self.W_hedge + other.W_hedge
        
        return new
    
    def __neg__(self):
        
        new = Strategy(-self.W)
        new.W_hedge = -self.W_hedge
        
        return new
    
    def __sub__(self, other):
        
        return self + (-other)