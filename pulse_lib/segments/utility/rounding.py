import numpy as np

def iround(value):
    '''
    Fast implementation of round which uses half-up, i.s.o. half-even rounding.
    '''
    return np.int_(value + 0.5)

#def get_effective_point_number(time, time_step):
#    return int(time/time_step + 0.5)