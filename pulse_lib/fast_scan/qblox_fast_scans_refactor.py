import logging
from typing import Callable, Iterable, cast
from collections.abc import Mapping
from dataclasses import dataclass, field
import numpy as np
from qcodes import MultiParameter
from pulse_lib.base_pulse import pulselib as PulseLib
from pulse_lib.sequencer import sequencer as Sequencer


DataProcesser = Callable[[np.ndarray], np.ndarray]
_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FastScan1dDef:
    """
    Dataclass for definition of fast 1D scan

    Attributes:
        pulse_lib: pulse library object, needed to make the sweep.
        gate: gate/gates that you want to sweep.
        swing: swing to apply on the AWG gates. [mV]
        n_pt: number of points to measure
        t_step: time in ns to measure per point. [ns]
        biasT_corr: correct for biasT by taking data in different order.
        acquisition_delay_ns (float): Time in ns between AWG output change and digitizer acquisition start.
        line_margin: number of points to add to sweep 1 to mask transition effects due to voltage step.
            The points are added to begin and end for symmetry (bias-T).
        acq_channels: digitizer channels to read
        acq_channel_map: defines new list of derived channels to display. Dictionary entries name: (channel_name, func).
            E.g. {(ch1-I':(1, np.real), 'ch1-Q':('ch1', np.imag), 'ch3-Amp':('ch3', np.abs),
            'ch3-Phase':('ch3', np.angle)}
        enabled_markers: marker channels to enable during scan
        pulse_gates: Gates to pulse during scan with pulse voltage in mV.
            E.g. {'vP1': 10.0, 'vB2': -29.1}
        n_avg: number of times to scan and average data.
    """

    pulselib: PulseLib
    gate: str
    swing: float
    n_pt: int
    t_step: int
    bias_t_corr: bool = False
    acquisition_delay_ns: int = 100
    line_margin: int = 0
    acq_channels: list[str] = field(default_factory=list)
    acq_channel_map: dict[str, tuple[str, DataProcesser]] = field(default_factory=dict)
    enabled_markers: list[str] = field(default_factory=list)
    pulse_gates: dict[str, float] = field(default_factory=dict)
    n_avg: int = 1

    def __post_init__(self):
        if self.t_step < 1000:
            raise ValueError("Measurement time too short. Minimum is 1000")


def fast_scan1D_param(pulse_lib: PulseLib, gate: str, swing: float, n_pt: int, t_step: int, **kwargs) -> MultiParameter:
    kwargs["pulselib"] = pulse_lib
    kwargs["gate"] = gate
    kwargs["swing"] = swing
    kwargs["n_pt"] = n_pt
    kwargs["t_step"] = t_step
    if "biasT_corr" in kwargs:
        kwargs["bias_t_corr"] = kwargs.pop("biasT_corr")
    return fast_scan1d_param(FastScan1dDef(**kwargs))


def fast_scan1d_param(scan_def: FastScan1dDef) -> MultiParameter:
    """
    Creates a parameter to do a 1D fast scan.

    Args:
        scan_def: Definition of the scan

    Returns:
        Parameter: parameter that can be used as input in a conventional scan function.
    """
    _logger.info(f"fast scan 1D: {scan_def.gate}")

    # timing for the scan
    acquisition_delay = max(100, scan_def.acquisition_delay_ns)
    step_eff = scan_def.t_step + acquisition_delay

    # add margin, if requested
    line_margin = int(scan_def.line_margin)
    n_ptx = scan_def.n_pt + 2 * line_margin
    vp = scan_def.swing / 2
    vpx = vp * (n_ptx - 1) / (scan_def.n_pt - 1)

    # calculate setpoints for sweep
    voltages_sp = np.linspace(-vp, vp, scan_def.n_pt)
    voltages = np.linspace(-vpx, vpx, n_ptx)
    if scan_def.bias_t_corr:
        voltages = _bias_t_shuffle(voltages)

    # administration for acquistion channels
    if not scan_def.acq_channel_map:
        if not scan_def.acq_channels:
            acq_channels = list(scan_def.pulselib.digitizer_channels.keys())
        else:
            acq_channels = scan_def.acq_channels
        acq_channel_map = {str(name): (str(name), cast(DataProcesser, np.real)) for name in acq_channels}
    else:
        acq_channel_map = scan_def.acq_channel_map
        acq_channels = list(
            set(v[0] for v in acq_channel_map.values())
        )  # set to remove duplicates

    # Construct the pulse sequence
    seg = scan_def.pulselib.mk_segment()
    sweep_channel = seg[scan_def.gate]
    pulse_channels = [(seg[ch], v) for ch, v in scan_def.pulse_gates.items()]
    for i, voltage in enumerate(voltages):
        # sweep channel
        sweep_channel.add_block(0, step_eff, voltage)
        if 0 <= i - line_margin < scan_def.n_pt:
            for acq_ch in acq_channels:
                seg[acq_ch].acquire(0, scan_def.t_step)
        # DC channels
        for gp, v in pulse_channels:
            gp.add_block(0, step_eff, v)
            if scan_def.bias_t_corr:  # also compensation for DC pulse gates
                gp.add_block(step_eff, 2 * step_eff, -v)
        seg.reset_time()
    # Marker channels
    end_time = seg.total_time[0]
    for marker_ch in [seg[m] for m in scan_def.enabled_markers]:
        marker_ch.reset_time(0)
        marker_ch.add_marker(0, end_time)

    # generate the sequence and upload it.
    my_seq = scan_def.pulselib.mk_sequence([seg])
    my_seq.n_rep = scan_def.n_avg
    # Note: uses hardware averaging with Qblox modules
    my_seq.set_acquisition(
        t_measure=scan_def.t_step, channels=acq_channels, average_repetitions=True
    )
    _logger.info("Upload")
    my_seq.upload()

    return _ScanParameter(
        pulselib=scan_def.pulselib,
        my_seq=my_seq,
        t_measure=scan_def.t_step,
        shape=(scan_def.n_pt,),
        names=(scan_def.gate,),
        setpoint=((voltages_sp,), ),
        bias_t_corr=scan_def.bias_t_corr,
        channel_map=acq_channel_map,
    )


def _bias_t_shuffle(arr_in: np.ndarray) -> np.ndarray:
    # Shuffle array indices as: 0, N-1, 1, N-2, etc
    # If the input is a ramp, this avoids bias T loading
    assert len(arr_in.shape) == 1
    n_ptx = len(arr_in)
    m = (n_ptx + 1) // 2
    arr_out = np.zeros(n_ptx)
    arr_out[::2] = arr_in[:m]  # Even indices contain first half of the input
    arr_out[1::2] = arr_in[m:][
        ::-1
    ]  # Odd indices contain reversed second half of the input
    return arr_out


class _ScanParameter(MultiParameter):
    """
    Scan parameter for digitizer
    """

    def __init__(
        self,
        *,
        pulselib: PulseLib,
        my_seq: Sequencer,
        t_measure: int,
        shape: tuple[int],
        names: Iterable[str],
        setpoint: Iterable[Iterable[np.ndarray]],
        bias_t_corr: bool,
        channel_map: Mapping[str, tuple[str, DataProcesser]],
    ):
        """
        args:
            pulselib: pulse library object
            my_seq: sequence of the 1D scan
            t_measure: time to measure per step
            shape: expected output shape
            names: name of the gate(s) that are measured.
            setpoint: array witht the setpoints of the input data
            bias_t_corr: bias T correction or not -- if enabled -- automatic reshaping of the data.
            channel_map:
                defines new list of derived channels to display. Dictionary entries name: (channel_name, func).
                E.g. {(ch1-I':(1, np.real), 'ch1-Q':('ch1', np.imag), 'ch3-Amp':('ch3', np.abs),
                'ch3-Phase':('ch3', np.angle)}
        """
        self.my_seq = my_seq
        self.pulse_lib = pulselib
        self.t_measure = t_measure
        self.n_rep = np.prod(shape)
        self.channel_map = dict(channel_map)
        self.channel_names = tuple(self.channel_map.keys())
        self.bias_t_corr = bias_t_corr
        self.shape = shape

        n_out_ch = len(self.channel_names)
        names_qc = tuple([tuple(names)] * n_out_ch)
        setpoints_qc = tuple([tuple(s) for s in setpoint] * n_out_ch)
        super().__init__(
            name="fastScan",
            names=self.channel_names,
            shapes=tuple([shape] * n_out_ch),
            labels=self.channel_names,
            units=tuple(["mV"] * n_out_ch),
            setpoints=setpoints_qc,
            setpoint_names=names_qc,
            setpoint_labels=names_qc,
            setpoint_units=(("mV",) * len(names_qc[0]),) * n_out_ch,
            docstring="Scan parameter for digitizer",
        )

    def get_raw(self):
        # play sequence
        self.my_seq.play(release=False)
        raw_dict = self.my_seq.get_measurement_data()

        # get the data, converted to mV
        data = [func(raw_dict[ch] * 1000.0) for ch, func in self.channel_map.values()]

        # make sure that data is put in the right order.
        data_out = []
        for ch_data in data:
            out = ch_data.reshape(self.shape)
            if self.bias_t_corr:
                out = _bias_t_shuffle(out)
            data_out.append(out)

        return tuple(data_out)

    def stop(self):
        if self.my_seq is not None and self.pulse_lib is not None:
            # remove pulse sequence from the AWG's memory, unload schedule and free memory.
            _logger.info("stop: release memory")
            self.my_seq.close()
            self.my_seq = None
            self.pulse_lib = None

    def __del__(self):
        if self.my_seq is not None and self.pulse_lib is not None:
            _logger.warning("Automatic cleanup in __del__(); Calling stop()")
            self.stop()
