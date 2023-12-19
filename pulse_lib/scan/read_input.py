

def read_channels(pulselib, t_measure, channels=None, sample_rate=None, iq_mode='Complex'):
    '''
    Reads the input channels without pulsing the AWG channels.

    Args:
        pulselib: pulselib object.
        t_measure (float): measurement time [ns].
        channels (Optional[list[name]]): pulselib names of channels to read.
        sample_rate (Optional[float]): sample rate for time trace [Hz]
        iq_mode (str):
            when channel contains IQ data, i.e. iq_input=True or frequency is not None,
            then this parameter specifies how the complex I/Q value should be returned:
                'Complex': return IQ data as complex value.
                'I': return only I value.
                'Q': return only Q value.
                'amplitude': return amplitude.
                'phase:' return phase [radians],
                'I+Q', return I and Q using channel name postfixes '_I', '_Q'.
                'amplitude+phase'. return amplitude and phase using channel name postfixes '_amp', '_phase'.

    Note:
        The frequency and phase for IQ modulation is configured on the pulselib digitizer channel.
    '''
    if sample_rate and int(t_measure) < int(1e9/sample_rate):
        raise Exception(f't_measure ({t_measure} ns) < 1/sample_rate ({int(1e9/sample_rate)} ns)')

    # set sample rate for Keysight upload.
    if t_measure > 2_000_000:
        awg_sample_rate = 1e6
    elif t_measure > 200_000:
        awg_sample_rate = 1e7
    elif t_measure > 20_000:
        awg_sample_rate = 1e8
    else:
        awg_sample_rate = 1e9

    if channels is None:
        channels = pulselib.digitizer_channels.keys()

    seg = pulselib.mk_segment(sample_rate=awg_sample_rate)
    for ch in channels:
        seg[ch].acquire(0, t_measure, wait=True)

    sequence = pulselib.mk_sequence([seg])
    sequence.n_rep = None
    if sample_rate is not None:
        sequence.set_acquisition(sample_rate=sample_rate)

    param = sequence.get_measurement_param(upload='auto', iq_mode=iq_mode)
    return param

