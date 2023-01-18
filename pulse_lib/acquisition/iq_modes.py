import numpy as np

def iq_mode2func(iq_mode):
    '''
    Returns:
        func[np.array]->np.array, or
        list[Tuple[str, func[np.array]->np.array]]
    '''
    func_map = {
        'Complex': lambda x:x,
        'I': np.real,
        'Q': np.imag,
        'abs': np.abs,
        'angle': np.angle,
        'I+Q': [('_I', np.real), ('_Q', np.imag)],
        'abs+angle': [('_abs', np.abs), ('_angle', np.angle)],
        }
    try:
        return func_map[iq_mode]
    except KeyError:
        raise Exception(f'Unknown iq_mode f{iq_mode}')
