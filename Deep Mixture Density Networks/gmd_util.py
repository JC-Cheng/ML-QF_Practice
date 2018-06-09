from scipy.stats import norm
from scipy.optimize import fsolve
import numpy as np

def vec_func(f):
    vf = np.vectorize(f, excluded=['logit', 'mu', 'sig'])
    return vf

def PDF_gmd(logit, mu, sig):
    
    @vec_func
    def pdf(x):
        return np.dot(logit, [norm.pdf(np.array((x - m) / s)) / s for m, s in zip(mu, sig)])
    
    return pdf


def CDF_gmd(logit, mu, sig):
    
    @vec_func
    def cdf(x):
        return np.dot(logit, [norm.cdf(np.array((x - m) / s)) for m, s in zip(mu, sig)])

    return cdf


def iCDF_gmd(logit, mu, sig):
    
    assert(abs(np.sum(logit) - 1) <= 1e-10)
    cdf = CDF_gmd(logit, mu, sig)
    pdf = PDF_gmd(logit, mu, sig)
    
    @vec_func
    def inverse_cdf(alpha):
        F = lambda x: cdf(x) - alpha
        res = fsolve(F, 0, fprime=pdf)
        return res
    
    return inverse_cdf