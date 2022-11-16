from qcodes import MultiParameter

import numpy as np
import logging


def fast_scan1D_param(pulse_lib, gate, swing, n_pt, t_step,
                      biasT_corr=False,
                      acquisition_delay_ns=200,
                      line_margin=0,
                      channels=None,
                      channel_map=None,
                      enabled_markers=[],
                      pulse_gates={},
                      n_avg=1):
    """
    Creates a parameter to do a 1D fast scan.

    Args:
        pulse_lib : pulse library object, needed to make the sweep.
        gate (str) : gate/gates that you want to sweep.
        swing (double) : swing to apply on the AWG gates. [mV]
        n_pt (int) : number of points to measure
        t_step (double) : time in ns to measure per point. [ns]
        biasT_corr (bool) : correct for biasT by taking data in different order.
        acquisition_delay_ns (float):
                Time in ns between AWG output change and digitizer acquisition start.
        line_margin (int): number of points to add to sweep 1 to mask transition effects due to voltage step.
            The points are added to begin and end for symmetry (bias-T).
        channels List[str]: digitizer channels to read
        channel_map (Dict[str, Tuple(str, Callable[[np.ndarray], np.ndarray])]):
            defines new list of derived channels to display. Dictionary entries name: (channel_name, func).
            E.g. {(ch1-I':(1, np.real), 'ch1-Q':('ch1', np.imag), 'ch3-Amp':('ch3', np.abs), 'ch3-Phase':('ch3', np.angle)}
        enabled_markers (List[str]): marker channels to enable during scan
        pulse_gates (Dict[str, float]):
            Gates to pulse during scan with pulse voltage in mV.
            E.g. {'vP1': 10.0, 'vB2': -29.1}
        n_avg (int): number of times to scan and average data.

    Returns:
        Parameter (QCODES multiparameter) : parameter that can be used as input in a conversional scan function.
    """
    logging.info(f'fast scan 1D: {gate}')

    vp = swing/2
    line_margin = int(line_margin)

    # set up timing for the scan
    acquisition_delay = max(100, acquisition_delay_ns)
    step_eff = t_step + acquisition_delay

    if t_step < 1000:
        msg = f'Measurement time too short. Minimum is 1000'
        logging.error(msg)
        raise Exception(msg)

    n_ptx = n_pt + 2*line_margin
    vpx = vp * (n_ptx-1)/(n_pt-1)

    # set up sweep voltages (get the right order, to compenstate for the biasT).
    voltages_sp = np.linspace(-vp,vp,n_pt)
    voltages_x = np.linspace(-vpx,vpx,n_ptx)
    if biasT_corr:
        m = (n_ptx+1)//2
        voltages = np.zeros(n_ptx)
        voltages[::2] = voltages_x[:m]
        voltages[1::2] = voltages_x[m:][::-1]
    else:
        voltages = voltages_x

    if channel_map is None:
        if channels is None:
            acq_channels = list(pulse_lib.digitizer_channels.keys())
        else:
            acq_channels = channels
        channel_map = {i:(i, np.real) for i in acq_channels}
    else:
        acq_channels = set(v[0] for v in channel_map.values())

    seg  = pulse_lib.mk_segment()

    g1 = seg[gate]
    pulse_channels = []
    for ch,v in pulse_gates.items():
        pulse_channels.append((seg[ch], v))

    for i,voltage in enumerate(voltages):
        g1.add_block(0, step_eff, voltage)
        if 0 <= i-line_margin < n_pt:
            for acq_ch in acq_channels:
                seg[acq_ch].acquire(acquisition_delay, t_step)

        for gp,v in pulse_channels:
            gp.add_block(0, step_eff, v)
            # compensation for pulse gates
            if biasT_corr:
                gp.add_block(step_eff, 2*step_eff, -v)
        seg.reset_time()

    end_time = seg.total_time[0]
    for marker in enabled_markers:
        marker_ch = seg[marker]
        marker_ch.reset_time(0)
        marker_ch.add_marker(0, end_time)

    # generate the sequence and upload it.
    my_seq = pulse_lib.mk_sequence([seg])
    my_seq.n_rep = n_avg
    # Note: uses hardware averaging with Qblox modules
    my_seq.set_acquisition(t_measure=t_step, channels=acq_channels, average_repetitions=True)

    logging.info(f'Upload')
    my_seq.upload()

    return _scan_parameter(pulse_lib, my_seq, t_step,
                           (n_pt, ), (gate, ), (tuple(voltages_sp), ),
                           biasT_corr, channel_map=channel_map)


def fast_scan2D_param(pulse_lib, gate1, swing1, n_pt1, gate2, swing2, n_pt2, t_step,
                      biasT_corr=True,
                      acquisition_delay_ns=200,
                      line_margin=0,
                      channels=None,
                      channel_map=None,
                      enabled_markers=[],
                      pulse_gates={},
                      n_avg=1):
    """
    Creates a parameter to do a 2D fast scan.

    Args:
        pulse_lib : pulse library object, needed to make the sweep.
        gates1 (str) : gate that you want to sweep on x axis.
        swing1 (double) : swing to apply on the AWG gates.
        n_pt1 (int) : number of points to measure (current firmware limits to 1000)
        gate2 (str) : gate that you want to sweep on y axis.
        swing2 (double) : swing to apply on the AWG gates.
        n_pt2 (int) : number of points to measure (current firmware limits to 1000)
        t_step (double) : time in ns to measure per point.
        biasT_corr (bool) : correct for biasT by taking data in different order.
        acquisition_delay_ns (float):
                Time in ns between AWG output change and digitizer acquisition start.
                This also increases the gap between acquisitions.
        line_margin (int): number of points to add to sweep 1 to mask transition effects due to voltage step.
            The points are added to begin and end for symmetry (bias-T).
        channels List[str]: digitizer channels to read
        channel_map (Dict[str, Tuple(str, Callable[[np.ndarray], np.ndarray])]):
            defines new list of derived channels to display. Dictionary entries name: (channel_name, func).
            E.g. {(ch1-I':(1, np.real), 'ch1-Q':('ch1', np.imag), 'ch3-Amp':('ch3', np.abs), 'ch3-Phase':('ch3', np.angle)}
        enabled_markers (List[str]): marker channels to enable during scan
        pulse_gates (Dict[str, float]):
            Gates to pulse during scan with pulse voltage in mV.
            E.g. {'vP1': 10.0, 'vB2': -29.1}
        n_avg (int): number of times to scan and average data.

    Returns:
        Parameter (QCODES multiparameter) : parameter that can be used as input in a conversional scan function.
    """
    logging.info(f'Fast scan 2D: {gate1} {gate2}')

    # set up timing for the scan
    acquisition_delay = max(100, acquisition_delay_ns)
    step_eff = t_step + acquisition_delay

    if t_step < 1000:
        msg = f'Measurement time too short. Minimum is 1000'
        logging.error(msg)
        raise Exception(msg)

    if channel_map is None:
        if channels is None:
            acq_channels = list(pulse_lib.digitizer_channels.keys())
        else:
            acq_channels = channels
        channel_map = {i:(i, np.real) for i in acq_channels}
    else:
        acq_channels = set(v[0] for v in channel_map.values())

    line_margin = int(line_margin)
    add_pulse_gate_correction = biasT_corr and len(pulse_gates) > 0

    # set up sweep voltages (get the right order, to compenstate for the biasT).
    vp1 = swing1/2
    vp2 = swing2/2

    voltages1_sp = np.linspace(-vp1,vp1,n_pt1)
    voltages2_sp = np.linspace(-vp2,vp2,n_pt2)

    n_ptx = n_pt1 + 2*line_margin
    vpx = vp1 * (n_ptx-1)/(n_pt1-1)

    if biasT_corr:
        m = (n_pt2+1)//2
        voltages2 = np.zeros(n_pt2)
        voltages2[::2] = voltages2_sp[:m]
        voltages2[1::2] = voltages2_sp[m:][::-1]
    else:
        voltages2 = voltages2_sp

    start_delay = line_margin * step_eff
    if biasT_corr:
        # prebias: add half line with +vp2
        prebias_pts = (n_ptx)//2
        t_prebias = prebias_pts * step_eff
        start_delay += t_prebias

    line_delay = 2 * line_margin * step_eff
    if add_pulse_gate_correction:
        line_delay += n_ptx*step_eff

    seg  = pulse_lib.mk_segment()

    g1 = seg[gate1]
    g2 = seg[gate2]
    pulse_channels = []
    for ch,v in pulse_gates.items():
        pulse_channels.append((seg[ch], v))

    if biasT_corr:
        # correct voltage to ensure average == 0.0 (No DC correction pulse needed at end)
        total_duration = prebias_pts + n_ptx*n_pt2 * (2 if add_pulse_gate_correction else 1)
        g2.add_block(0, -1, -(prebias_pts * vp2)/total_duration)
        g2.add_block(0, t_prebias, vp2)
        for g,v in pulse_channels:
            g.add_block(0, t_prebias, -v)
        seg.reset_time()

    for v2 in voltages2:

        g1.add_ramp_ss(0, step_eff*n_ptx, -vpx, vpx)
        g2.add_block(0, step_eff*n_ptx, v2)
        for acq_ch in acq_channels:
            seg[acq_ch].acquire(step_eff*line_margin+acquisition_delay, n_repeat=n_pt1, interval=step_eff)
        for g,v in pulse_channels:
            g.add_block(0, step_eff*n_ptx, v)
        seg.reset_time()

        if add_pulse_gate_correction:
            # add compensation pulses of pulse_channels
            # sweep g1 onces more; best effect on bias-T
            # keep g2 on 0
            g1.add_ramp_ss(0, step_eff*n_ptx, -vpx, vpx)
            for g,v in pulse_channels:
                g.add_block(0, step_eff*n_ptx, -v)
            seg.reset_time()

    end_time = seg.total_time[0]
    for marker in enabled_markers:
        marker_ch = seg[marker]
        marker_ch.reset_time(0)
        marker_ch.add_marker(0, end_time)

    # generate the sequence and upload it.
    my_seq = pulse_lib.mk_sequence([seg])
    my_seq.n_rep = n_avg
    # Note: uses hardware averaging with Qblox modules
    my_seq.set_acquisition(t_measure=t_step, channels=acq_channels, average_repetitions=True)

    logging.info(f'Seq upload')
    my_seq.upload()

    return _scan_parameter(pulse_lib, my_seq, t_step,
                           (n_pt2, n_pt1), (gate2, gate1),
                           (tuple(voltages2_sp), (tuple(voltages1_sp),)*n_pt2),
                           biasT_corr, channel_map)



class _scan_parameter(MultiParameter):
    """
    generator for the parameter f
    """
    def __init__(self, pulse_lib, my_seq, t_measure, shape, names, setpoint,
                 biasT_corr, channel_map):
        """
        args:
            pulse_lib (pulselib): pulse library object
            my_seq (sequencer) : sequence of the 1D scan
            t_measure (int) : time to measure per step
            shape (tuple<int>): expected output shape
            names (tuple<str>): name of the gate(s) that are measured.
            setpoint (tuple<np.ndarray>): array witht the setpoints of the input data
            biasT_corr (bool): bias T correction or not -- if enabled -- automatic reshaping of the data.
            channel_map (Dict[str, Tuple(str, Callable[[np.ndarray], np.ndarray])]):
                defines new list of derived channels to display. Dictionary entries name: (channel_name, func).
                E.g. {(ch1-I':(1, np.real), 'ch1-Q':('ch1', np.imag), 'ch3-Amp':('ch3', np.abs), 'ch3-Phase':('ch3', np.angle)}
        """
        self.my_seq = my_seq
        self.pulse_lib = pulse_lib
        self.t_measure = t_measure
        self.n_rep = np.prod(shape)
        self.channel_map = channel_map
        self.channel_names = tuple(self.channel_map.keys())
        self.biasT_corr = biasT_corr
        self.shape = shape

        n_out_ch = len(self.channel_names)
        super().__init__(name='fastScan', names = self.channel_names,
                        shapes = tuple([shape]*n_out_ch),
                        labels = self.channel_names, units = tuple(['mV']*n_out_ch),
                        setpoints = tuple([setpoint]*n_out_ch), setpoint_names=tuple([names]*n_out_ch),
                        setpoint_labels=tuple([names]*n_out_ch), setpoint_units=(("mV",)*len(names),)*n_out_ch,
                        docstring='Scan parameter for digitizer')

    def get_raw(self):

        # play sequence
        self.my_seq.play(release = False)
        raw_dict = self.my_seq.get_channel_data()

        # get the data
        data = []
        for setting in self.channel_map.values():
            ch, func = setting
            # channel data already is in mV
            ch_data = raw_dict[ch]
            data.append(func(ch_data))

        # make sure that data is put in the right order.
        data_out = [np.zeros(self.shape) for i in range(len(data))]

        for i in range(len(data)):
            d = data[i]
            ch_data = d.reshape(self.shape)
            if self.biasT_corr:
                data_out[i][:len(ch_data[::2])] = ch_data[::2]
                data_out[i][len(ch_data[::2]):] = ch_data[1::2][::-1]
            else:
                data_out[i] = ch_data

        return tuple(data_out)

    def stop(self):
        if not self.my_seq is None and not self.pulse_lib is None:
            logging.info('stop: release memory')
            # remove pulse sequence from the AWG's memory, unload schedule and free memory.
            self.my_seq.close()
            self.my_seq = None
            self.pulse_lib = None


    def __del__(self):
        if not self.my_seq is None and not self.pulse_lib is None:
            logging.debug(f'Automatic cleanup in __del__(); Calling stop()')
            self.stop()

