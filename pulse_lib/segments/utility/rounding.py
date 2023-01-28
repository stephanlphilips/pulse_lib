import numpy as np

def iround(value):
    '''
    Fast implementation of round which uses half-up, i.s.o. half-even rounding.
    '''
    return np.floor(value + 0.5).astype(int)
