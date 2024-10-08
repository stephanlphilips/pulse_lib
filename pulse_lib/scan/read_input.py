import numpy as np


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

    Returns:
        qcodes parameter to read channels

    Note:
        The frequency and phase for IQ modulation is configured on the pulselib digitizer channel.
    '''
    if sample_rate and int(t_measure) < int(1e9/sample_rate):
        raise Exception(f't_measure ({t_measure} ns) < 1/sample_rate ({int(1e9/sample_rate)} ns)')

    if pulselib._backend in ["Keysight", "Keysight_QS", "M3202A"]:
        if t_measure > 42e9:
            raise Exception("Keysight backend implementation does not support t_measure > 42 s")
        if (pulselib._backend == "Keysight_QS"
            and sample_rate is not None
            and t_measure*1e-9*sample_rate > 2e6):
                raise Exception(f"Too many samples for Keysight_QS ({int(t_measure*1e-9*sample_rate)} > 2e6)")
        if sample_rate is not None and sample_rate < 1000:
            factor = 1000/sample_rate
            max_output = 0.25 / factor
            print(f"Warning: sample rate of {sample_rate} Sa/s is {factor:.1f} times the safe limit. "
                  f"Numerical overflow could occur. Check if all measured data is < {max_output:.2%} of configured "
                  "digitizer input range.")
        # set sample rate for Keysight upload.
        if t_measure > 20_000_000:
            awg_sample_rate = 1e5
        elif t_measure > 2_000_000:
            awg_sample_rate = 1e6
        elif t_measure > 200_000:
            awg_sample_rate = 1e7
        elif t_measure > 20_000:
            awg_sample_rate = 1e8
        else:
            awg_sample_rate = 1e9
    else:
        # let the driver set the sample rate.
        awg_sample_rate = None

    if (pulselib._backend == "Qblox"
        and sample_rate is not None
        and t_measure*1e-9*sample_rate > 130_000):
            raise Exception(f"Too many samples for Qblox QRM ({int(t_measure*1e-9*sample_rate)} > 130_000)")

    if channels is None:
        channels = pulselib.digitizer_channels.keys()

    seg = pulselib.mk_segment(sample_rate=awg_sample_rate)
    for ch in channels:
        seg[ch].acquire(0, t_measure, wait=True, ref=ch)

    sequence = pulselib.mk_sequence([seg])
    sequence.n_rep = None
    if sample_rate is not None:
        sequence.set_acquisition(sample_rate=sample_rate)

    param = sequence.get_measurement_param(upload='auto', iq_mode=iq_mode)

    parameters = dict(
        t_measure=dict(label="t_measure", value=t_measure, unit="ns"),
        iq_mode=iq_mode,
        )
    if sample_rate is not None:
        parameters['sample_rate'] = dict(label="sample_rate", value=sample_rate, unit="Sa/s")
    param.update_snapshot(snapshot_extra={"parameters": parameters})

    return param


def scan_resonator_frequency(
        pulselib,
        channel: str,
        t_measure: int,
        f_start: float,
        f_stop: float,
        f_step: float,
        n_rep=None,
        average_repetitions=False,
        iq_mode='Complex',
        ):
    '''
    Reads the input channel while stepping the frequency from start to stop (inclusive).

    Args:
        pulselib: pulselib object
        channel (str): digitizer channel (sensor name) to read
        t_measure (float): measurement time per point
        f_start (float): start frequency
        f_stop (float): stop frequency, inclusive if (f_stop - f_start)/f_step is (close to) an integer number
        f_step (float): frequency
        n_rep (int or None): if n_rep > 1 then number of repetitions
        average_repetitions (bool): if True the repetitions will be averaged (in hardware)
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

    Returns:
        qcodes parameter to read channels
    '''
    if pulselib._backend != 'Qblox':
        raise Exception("Resonator scan is only supported for Qblox")

    sample_rate = 1e9/t_measure
    n_freq = np.floor((f_stop - f_start) / f_step + 1.001)
    t_total = n_freq * t_measure
    # fix f_stop to match last f_step
    f_stop = f_start + (n_freq-1) * f_step

    seg = pulselib.mk_segment()
    seg[channel].acquire(0, t_total, wait=True)

    sequence = pulselib.mk_sequence([seg])
    sequence.n_rep = None if n_rep is None or n_rep <= 1 else n_rep

    sequence.set_acquisition(
        sample_rate=sample_rate,
        average_repetitions=average_repetitions,
        f_sweep=(f_start, f_stop),
        )

    param = sequence.get_measurement_param(upload='auto', iq_mode=iq_mode)
    parameters = dict(
        t_measure=dict(label="t_measure", value=t_measure, unit="ns"),
        f_start=dict(label="f_start", value=f_start, unit="Hz"),
        f_stop=dict(label="f_stop", value=f_stop, unit="Hz"),
        f_step=dict(label="f_step", value=f_step, unit="Hz"),
        iq_mode=iq_mode,
        )
    if average_repetitions:
        parameters['average_repetitions'] = average_repetitions
        parameters['n_rep'] = n_rep
    param.update_snapshot(snapshot_extra={"parameters": parameters})

    return param
