import numpy as np

def iq_mode2func(iq_mode):
    '''
    Returns:
        list[Tuple[str, func[np.array]->np.array], str]
    '''
    func_map = {
        'Complex': [('', lambda x:x, 'mV')],
        'I': [('', np.real, 'mV')],
        'Q': [('', np.imag, 'mV')],
        'amplitude': [('', np.abs, 'mV')],
        'phase': [('', np.angle, 'rad')],
        'I+Q': [('_I', np.real, 'mV'), ('_Q', np.imag, 'mV')],
        'amplitude+phase': [('_amp', np.abs, 'mV'), ('_phase', np.angle, 'rad')],
        }
    try:
        return func_map[iq_mode]
    except KeyError:
        raise Exception(f'Unknown iq_mode {iq_mode}')
