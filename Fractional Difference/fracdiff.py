# Fast Fractional Difference Algorithm for Python 2 and 3
#
# Mirza Trokic
# December 08, 2013
#
# fracdiff(x,d) computes the fractional difference of an input vector based on the fast fractional
# differencing algorithm in 
#
#	Jensen and Nielsen (2013). A fast fractional differencing algorithm.
#	QED working paper 1307, Queen's University
#
# Input = vector of data x
#	  scalar fractional difference parameter d
#
# Output = vector (1-L)^d x


# JC Cheng, June 11, 2018

import numpy as np

def fracdiff( x,d ):
    T = len(x)
    np2 = int(2**np.ceil(np.log2(2*T-1)))
    k = np.arange(1,T)
    b = (1,) + tuple(np.cumprod((k-d-1)/k))
    z = (0,)*(np2-T)
    z1 = b + z
    z2 = tuple(x) + z
    dx = np.fft.ifft(np.fft.fft(z1)*np.fft.fft(z2))
    return np.real(dx[0:T])



# legecy code

'''

from pylab import *
from numpy import *
#end imports


def fracdiff( x,d ):
	T=len(x)
	np2=2**ceil(log2(2*T-1))
	k=arange(1,T)
	b=(1,) + tuple(cumprod((k-d-1)/k))
	z = (0,)*(np2-T)
	z1=b + z
	z2=tuple(x) + z
	dx = fft.ifft(fft.fft(z1)*fft.fft(z2))
	return real(dx[0:T])

'''