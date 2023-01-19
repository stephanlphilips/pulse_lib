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
        'amplitude': np.abs,
        'phase': np.angle,
        'I+Q': [('_I', np.real), ('_Q', np.imag)],
        'amplitude+phase': [('_amp', np.abs), ('_phase', np.angle)],
        }
    try:
        return func_map[iq_mode]
    except KeyError:
        raise Exception(f'Unknown iq_mode {iq_mode}')
