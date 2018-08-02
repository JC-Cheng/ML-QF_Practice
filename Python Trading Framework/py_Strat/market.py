import pandas as pd

class Market:
    
    def __init__(self, begin_price=None, end_price=None, begin_price_hedge=None, end_price_hedge=None):
        
        self.begin_px = begin_price
        self.end_px = end_price
        
        self.begin_px_hedge = begin_price_hedge
        self.end_px_hedge = end_price_hedge
        
        if (isinstance(self.begin_px, pd.DataFrame) or isinstance(self.begin_px, pd.Series)) and (isinstance(self.end_px, pd.DataFrame) or isinstance(self.end_px, pd.Series)):
            self.ret = ((self.end_px - self.begin_px) / self.begin_px).fillna(0)
        else:
            self.ret = None

        if (isinstance(self.begin_px_hedge, pd.DataFrame) or isinstance(self.begin_px_hedge, pd.Series)) and (isinstance(self.end_px_hedge, pd.DataFrame) or isinstance(self.end_px_hedge, pd.Series)):
            self.ret_hedge = ((self.end_px_hedge - self.begin_px_hedge) / self.begin_px_hedge).fillna(0)
        else:
            self.ret_hedge = None
        
    def get_returns(self, date):
        
        if isinstance(self.ret, pd.DataFrame):
            ret = self.ret.loc[date, :] if date in self.ret.index else None
        elif isinstance(self.ret, pd.Series):
            ret = self.ret.loc[date] if date in self.ret.index else None
        else:
            ret = None

        if isinstance(self.ret_hedge, pd.DataFrame):
            ret_hedge = self.ret_hedge.loc[date, :] if date in self.ret_hedge.index else None
        elif isinstance(self.ret_hedge, pd.Series):
            ret_hedge = self.ret_hedge.loc[date] if date in self.ret_hedge.index else None
        else:
            ret_hedge = None

        return ret, ret_hedge
    
    def get_begin_price(self, date):

        if isinstance(self.begin_px, pd.DataFrame):
            begin_px = self.begin_px.loc[date, :] if date in self.begin_px.index else None
        elif isinstance(self.begin_px, pd.Series):
            begin_px = self.begin_px.loc[date] if date in self.begin_px.index else None
        else:
            begin_px = None

        if isinstance(self.begin_px_hedge, pd.DataFrame):
            begin_px_hedge = self.begin_px_hedge.loc[date, :] if date in self.begin_px_hedge.index else None
        elif isinstance(self.begin_px_hedge, pd.Series):
            begin_px_hedge = self.begin_px_hedge.loc[date] if date in self.begin_px_hedge.index else None
        else:
            begin_px_hedge = None

        return begin_px, begin_px_hedge
    
    def get_end_price(self, date):

        if isinstance(self.end_px, pd.DataFrame):
            end_px = self.end_px.loc[date, :] if date in self.end_px.index else None
        elif isinstance(self.end_px, pd.Series):
            end_px = self.end_px.loc[date] if date in self.end_px.index else None
        else:
            end_px = None

        if isinstance(self.end_px_hedge, pd.DataFrame):
            end_px_hedge = self.end_px_hedge.loc[date, :] if date in self.end_px_hedge.index else None
        elif isinstance(self.end_px_hedge, pd.Series):
            end_px_hedge = self.end_px_hedge.loc[date] if date in self.end_px_hedge.index else None
        else:
            end_px_hedge = None

        return end_px, end_px_hedge

