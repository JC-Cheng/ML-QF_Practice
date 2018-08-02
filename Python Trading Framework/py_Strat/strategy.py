import pandas as pd
#import numpy as np

class Strategy:
    
    def __init__(self, weights=None, hedge_method='None', hedge_ratio=1, beta=None):
        
        if isinstance(weights, pd.DataFrame) or isinstance(weights, pd.Series):
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

        if isinstance(self.W, pd.DataFrame):
            W = self.W.loc[date, :] if date in self.W.index else None
        elif isinstance(self.W, pd.Series):
            W = self.W.loc[date] if date in self.W.index else None
        else:
            W = None

        if isinstance(self.W_hedge, pd.DataFrame):
            W_hedge = self.W_hedge.loc[date, :] if date in self.W_hedge.index else None
        elif isinstance(self.W_hedge, pd.Series):
            W_hedge = self.W_hedge.loc[date] if date in self.W_hedge.index else None
        else:
            W_hedge = None

        return W, W_hedge
            
    def eval_beta(self, beta):

        if isinstance(self.W, pd.DataFrame):
            B = (self.W * beta.loc[self.W.index]).sum(axis=1)
        elif isinstance(self.W, pd.Series):
            B = self.W * beta.loc[self.W.index]
        else:
            B = 0

        if isinstance(self.W_hedge, pd.DataFrame):
            B_hedge = self.W_hedge
        elif isinstance(self.W_hedge, pd.Series):
            B_hedge = self.W_hedge
        else:
            B_hedge = 0

        return B - B_hedge

    
    def eval_exposure(self):

        if isinstance(self.W, pd.DataFrame):
            W = self.W.sum(axis=1)
        elif isinstance(self.W, pd.Series):
            W = self.W
        else:
            W = 0

        if isinstance(self.W_hedge, pd.DataFrame):
            W_hedge = self.W_hedge.sum(axis=1)
        elif isinstance(self.W_hedge, pd.Series):
            W_hedge = self.W_hedge
        else:
            W_hedge = 0

        return W - W_hedge
    
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