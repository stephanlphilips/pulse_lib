import logging
import os
import math
from collections.abc import Sequence
from functools import partial
from numbers import Number
from typing import Dict, List, Union
import matplotlib.pyplot as pt
import numpy as np

import qcodes as qc
import qcodes.logger as qc_logger
from qcodes.logger import start_all_logging

from ruamel.yaml import YAML

from pulse_lib.base_pulse import pulselib
from pulse_lib.tests.hw_schedule_mock import HardwareScheduleMock
from pulse_lib.schedule.tektronix_schedule import TektronixSchedule

try:
    import core_tools as ct
    from core_tools.sweeps.scans import sweep, Scan
    _ct_imported = True
    _ct_configured = False
except Exception:
    _ct_imported = False

_qcodes_initialized = False

logger = logging.getLogger(__name__)


def init_logging():
    start_all_logging()
    qc_logger.get_console_handler().setLevel(logging.WARN)
    qc_logger.get_file_handler().setLevel(logging.DEBUG)


class Context:
    def __init__(self):
        self._dir = os.path.dirname(__file__)
        self._load_configurations()
        cfg = self.configurations
        self.load_configuration(cfg['default'])

    def _load_configurations(self):
        with open(self._dir + '/configurations.yaml') as fp:
            yaml = YAML()
            self.configurations = yaml.load(fp)

    def load_configuration(self, config_name):
        self.configuration_name = config_name
        self._configuration = self.configurations[config_name]
        self._load_station(self._configuration['station'])

    def close_station(self):
        if qc.Station.default is not None:
            qc.Station.default.close_all_registered_instruments()

    def _load_station(self, config):
        self.close_station()
        config_file = os.path.join(self._dir, config)
        station = qc.Station(default=True, config_file=config_file)
        self.station = station

        station.load_all_instruments()

        awgs = []
        digs = []
        # map Qblox Cluster to AWG1 and digitizer
        if 'Qblox' in station.components:
            cluster = station.Qblox
            cluster.reset()
            for module in cluster.modules:
                if module.present():
                    rf = 'RF' if module.is_rf_type else ''
                    print(f'  Add {module.name}: {module.module_type}{rf}')
                    station.add_component(module, module.name)
#                    try:
#                        module.config('trace', True)
#                        module.config('render_repetitions', False)
#                    except:
#                        pass
                    if module.is_qcm_type:
                        awgs.append(module)
                    else:
                        digs.append(module)
        else:
            for name, component in station.components.items():
                if name.startswith('AWG'):
                    awgs.append(component)
                if name.startswith('Dig'):
                    digs.append(component)

        cfg = self._configuration
        backend = cfg['backend']
        if backend in ['Keysight', 'Keysight_QS']:
            # for awg in awgs:
            #     # anti-ringing filter
            #     awg.set_digital_filter_mode(3)
            for dig in digs:
                # Set mode AVERAGE
                dig.set_acquisition_mode(1)

        if backend == 'Tektronix_5014':
            import pyspcm
            # assume M4i digitizer
            for dig in digs:
                dig.timeout(10_000)
                dig.clock_mode(pyspcm.SPC_CM_INTPLL)
                dig.reference_clock(int(1e7))
                # 50 Ohm termination
                dig.initialize_channels(mV_range=2000, termination=1, lp_filter=1)
                dig.set_ext0_OR_trigger_settings(1, 0, 0, 1600)  # POS, 1 MOhm, DC, 1.6V
                dig.sample_rate(40e6)

        self.awgs = awgs
        self.digitizers = digs
        if backend == 'Keysight_QS':
            self._configure_pxi()

    def _configure_pxi(self):
        # TODO Fix for mock.
        import keysightSD1 as SD1
        from keysight_fpga.sd1.sd1_utils import check_error

        pxi_triggers = [
                SD1.SD_TriggerExternalSources.TRIGGER_PXI6,
                SD1.SD_TriggerExternalSources.TRIGGER_PXI7,
            ]

        # configure AWG PXI trigger in
        for awg in self.awgs:
            print('pxi', awg.name)
            with awg._lock:
                for pxi in pxi_triggers:
                    check_error(awg.awg.FPGATriggerConfig(
                            pxi,
                            SD1.SD_FpgaTriggerDirection.IN,
                            SD1.SD_TriggerPolarity.ACTIVE_LOW,
                            SD1.SD_SyncModes.SYNC_NONE,
                            0))

        # configure digitizer PXI trigger out
        for pxi in pxi_triggers:
            check_error(self.digitizers[0].SD_AIN.FPGATriggerConfig(
                    pxi,
                    SD1.SD_FpgaTriggerDirection.INOUT,
                    SD1.SD_TriggerPolarity.ACTIVE_LOW,
                    SD1.SD_SyncModes.SYNC_NONE,
                    0))


    def init_pulselib(self, n_gates=0, n_qubits=0, n_markers=0,
                      n_sensors=0, rf_sources=False,
                      virtual_gates=False, finish=True,
                      no_IQ=False,
                      drive_with_plungers=False):
        self.n_plots = 0
        cfg = self._configuration
        station = self.station
        backend = cfg['backend']
        if backend == 'Qblox':
            from pulse_lib.qblox.pulsar_uploader import UploadAggregator
            UploadAggregator.verbose = True

        pulse = pulselib(backend=backend)
        self.pulse = pulse

        gate_map = {}
        for awg_name, gates in cfg['awg_channels'].items():
            for i, gate in enumerate(gates):
                if backend != 'Qblox':
                    i += 1
                gate_map[gate] = (awg_name, i)

        gates = []
        for i in range(n_gates):
            gate = f'P{i+1}'
            gates.append(gate)
            awg_name, channel = gate_map[gate]
            if awg_name not in pulse.awg_devices:
                pulse.add_awg(getattr(station, awg_name))
            pulse.define_channel(gate, awg_name, channel)
            pulse.add_channel_compensation_limit(gate, (-100, 50))
            # pulse.add_channel_attenuation(gate, 0.1)

        if virtual_gates:
            n_gates = len(gates)
            self.virtual_matrix = np.diag([0.9]*n_gates) + 0.1
            pulse.add_virtual_matrix(
                name='virtual-gates',
                real_gate_names=gates,
                virtual_gate_names=['v'+gate for gate in gates],
                matrix=self.virtual_matrix
            )

        for i in range(n_markers):
            self._add_marker(f'M{i+1}')

        if drive_with_plungers:
            for i in range(n_qubits):
                qubit = i+1
                if backend == 'Keysight_QS':
                    # Use a new awg channel on the same output to drive the qubit.
                    drive_channel_name = f"P{qubit}_drive"
                    awg_channel = pulse.awg_channels[f"P{qubit}"]
                    pulse.define_channel(drive_channel_name, awg_channel.awg_name, awg_channel.channel_number)
                else:
                    drive_channel_name = f"P{qubit}"
                iq_channel_name = f"drive_q{qubit}"
                pulse.define_iq_channel(iq_channel_name, i_name=drive_channel_name)
                pulse.set_iq_lo(iq_channel_name, 0.0)
                # qubit freqs: 50, 100, 150, 200 MHz
                resonance_frequency = qubit*0.050e9
                pulse.define_qubit_channel(f"q{qubit}", iq_channel_name, resonance_frequency)
        else:
            n_iq = math.ceil(n_qubits/2)
            for i in range(n_iq):
                I, Q = f'I{i+1}', f'Q{i+1}',
                awg, channel_I = gate_map[I]
                awg, channel_Q = gate_map[Q]
                if awg not in pulse.awg_devices:
                    pulse.add_awg(station.components[awg])
                pulse.define_channel(I, awg, channel_I)
                pulse.add_channel_delay(I, -20)
                if not no_IQ:
                    pulse.define_channel(Q, awg, channel_Q)
                    pulse.add_channel_delay(Q, -20)
                sig_gen = station.components[f'sig_gen{i+1}']

                iq_channel_name = f'IQ{i+1}'
                if i == 0:
                    iq_marker = 'M_IQ1'
                    self._add_marker(iq_marker, setup_ns=100, hold_ns=20)
                    pulse.add_channel_delay(iq_marker, -20)
                elif i == 1:
                    iq_marker = 'M_IQ2'
                    self._add_marker(iq_marker, setup_ns=100, hold_ns=20)
                    pulse.add_channel_delay(iq_marker, -20)
                else:
                    iq_marker = ''

                if not no_IQ:
                    pulse.define_iq_channel(iq_channel_name, i_name=I, q_name=Q,
                                            marker_name=iq_marker)
                    pulse.set_iq_lo(iq_channel_name, sig_gen.frequency)

                    # LO freqs: 2.400, 2.800
                    sig_gen.frequency(2.400e9 + i*0.400e9)
                    # qubit freqs: 2.450, 2.550, 2.650, 2.750
                    for j in range(2):
                        qubit = 2*i+j+1
                        if qubit < n_qubits+1:
                            resonance_frequency = 2.350e9 + qubit*0.100e9
                            pulse.define_qubit_channel(f"q{qubit}", iq_channel_name, resonance_frequency)
                else:
                    pulse.define_iq_channel(iq_channel_name, i_name=I,
                                            marker_name=iq_marker)
                    pulse.set_iq_lo(iq_channel_name, 0.0)
                    # qubit freqs: 50, 100, 150, 200 MHz
                    for j in range(2):
                        qubit = 2*i+j+1
                        if qubit < n_qubits+1:
                            resonance_frequency = qubit*0.050e9
                            pulse.define_qubit_channel(f"q{qubit}", iq_channel_name, resonance_frequency)

        if n_sensors > 0:
            pulse.configure_digitizer = True

        for i in range(n_sensors):
            sensor = f'SD{i+1}'
            digitizer_name, channel = cfg['sensors'][sensor]
            if digitizer_name not in pulse.digitizers:
                pulse.add_digitizer(getattr(station, digitizer_name))
            if isinstance(channel, int):
                pulse.define_digitizer_channel(sensor, digitizer_name, channel)
            else:
                iq_out = rf_sources and sensor in cfg['rf']
                pulse.define_digitizer_channel_iq(sensor, digitizer_name, channel,
                                                  iq_out=iq_out)
            if backend == 'Qblox':
                pulse.add_channel_delay(sensor, 152)

        if n_sensors > 0 and backend == 'Tektronix_5014':
            self._add_marker('M_M4i')
            pulse.add_digitizer_marker('Dig1', 'M_M4i')

        if rf_sources:
            for sensor, params in cfg['rf'].items():
                if sensor not in pulse.digitizer_channels:
                    continue
                if backend == 'Qblox':
                    pulse.digitizer_channels[sensor].iq_out = True
                    pulse.set_digitizer_frequency(sensor, params['frequency'])
                    pulse.set_digitizer_rf_source(sensor,
                                                  output=params['output'],
                                                  amplitude=params['amplitude'],
                                                  mode='pulsed',
                                                  startup_time_ns=params['startup_time'],
                                                  prolongation_ns=params.get('prolongation_time', 0))
                else:
                    output = params['output']
                    if not isinstance(output, str):
                        output = tuple(output)
                        if len(output) == 2:
                            awg_name = output[0]
                            if awg_name not in pulse.awg_devices:
                                pulse.add_awg(station.components[awg_name])

                    channel_conf = pulse.digitizer_channels[sensor]
                    channel_conf.iq_out = True
                    dig = pulse.digitizers[channel_conf.module_name]
                    dig.set_channel_acquisition_mode(channel_conf.channel_number, 2)
                    pulse.set_digitizer_frequency(sensor, params.get('frequency', None))
                    pulse.set_digitizer_rf_source(sensor,
                                                  output=output,
                                                  amplitude=params.get('amplitude', None),
                                                  mode='pulsed',
                                                  startup_time_ns=params['startup_time'],
                                                  prolongation_ns=params.get('prolongation_time', 0))
                    pulse.set_digitizer_hw_input_channel(sensor, params.get('hw_input_channel'))

        if backend == 'Tektronix_5014':
            # pulselib always wants a digitizer for Tektronix
            if 'Dig1' not in pulse.digitizers:
                pulse.add_digitizer(station.components['Dig1'])

        self.set_default_hw_schedule(cfg.get('schedule', None))

        if finish:
            pulse.finish_init()

        return pulse

    def _add_marker(self, name, setup_ns=0, hold_ns=0):
        cfg = self._configuration
        pulse = self.pulse
        awg, channel = cfg['markers'][name]
        if isinstance(channel, Sequence):
            channel = tuple(channel)
        if awg not in pulse.awg_devices:
            pulse.add_awg(self.station.components[awg])
        pulse.define_marker(name, awg, channel, setup_ns=setup_ns, hold_ns=hold_ns)

    def set_default_hw_schedule(self, schedule):
        if schedule == 'Mock':
            hw_schedule_creator = HardwareScheduleMock
        elif schedule == 'HVI2':
            from core_tools.HVI2.hvi2_schedule_loader import Hvi2ScheduleLoader
            hw_schedule_creator = partial(Hvi2ScheduleLoader, script_name='SingleShot')
        elif schedule == 'TektronixM4i':
            hw_schedule_creator = TektronixSchedule
        else:
            hw_schedule_creator = None
        self.pulse.set_default_hw_schedule_creator(hw_schedule_creator)

    def launch_databrowser(self):
        global _ct_configured
        if not _ct_imported:
            raise Exception('core_tools import failed')
        if not _ct_configured:
            ct.configure(os.path.join(self._dir, 'ct_config.yaml'))
            _ct_configured = True
        ct.launch_databrowser()

    def init_coretools(self):
        global _ct_configured
        if not _ct_imported:
            raise Exception('core_tools import failed')
        if not _ct_configured:
            ct.configure(os.path.join(self._dir, 'ct_config.yaml'))
            _ct_configured = True
        ct.set_sample_info(sample=self.configuration_name)

    def _init_qcodes_data(self):
        global _qcodes_initialized
        if not _qcodes_initialized:
            try:
                from qcodes.data.data_set import DataSet
                from qcodes.data.io import DiskIO
            except ImportError:
                from qcodes_loops.data.data_set import DataSet
                from qcodes_loops.data.io import DiskIO

            _qcodes_initialized = True
            path = 'C:/measurements/test_pulselib'
            DataSet.default_io = DiskIO(path)

    def _play_loop(self, sequence, i, wait):
        if i < len(sequence.params):
            param = sequence.params[i]
            for v in param.values:
                param(v)
                self._play_loop(sequence, i+1, wait)
        else:
            sequence.upload()
            sequence.play()
            if wait and self.pulse._backend in ['Keysight', 'Keysight_QS']:
                sequence.uploader.wait_until_AWG_idle()

    def play(self, sequence, wait=False):
        self._play_loop(sequence, 0, wait=wait)

    def run(self, name, sequence, *params, silent=False, sweeps=[], close_sequence=True):
        runner = self._configuration['runner']
        self.last_sequence = sequence
        if runner == 'qcodes':
            from pulse_lib.tests.utils.qc_run import qc_run
            self._init_codes_data()
            ds = qc_run(name, *sweeps, sequence, *params, quiet=silent)

        elif runner == 'core_tools':
            self.init_coretools()
            scan_sweeps = []
            for sw in sweeps:
                scan_sweeps.append(sweep(*sw))
            ds = Scan(*scan_sweeps, sequence, *params, name=name, silent=silent).run()
        else:
            print(f'no implementation for {runner}')
        if close_sequence:
            try:
                sequence.close()
            except Exception:
                pass
        return ds

    def set_mock_data(self,
                      data: Dict[str, List[Union[float, np.ndarray]]],
                      repeat=None):
        if not repeat:
            repeat = 1
        for ch_name, values in data.items():
            l = []
            for value in values:
                if isinstance(value, Number):
                    l.append([value])
                else:
                    l.append(value)
            ch_data = np.tile(np.concatenate(l), repeat)
            try:
                self._set_mock_data(ch_name, ch_data)
            except Exception:
                logger.error("Couldn't set mock data for {ch_name}", exc_info=True)

    def _set_mock_data(self, ch_name, ch_data, scaling=1.0):
        ch_data = np.require(ch_data, dtype=float)
        ch_data *= scaling
        backend = self._configuration['backend']
        if backend == 'Qblox':
            seq_def = self.pulse.uploader.q1instrument.readouts[ch_name]
            in_ranges = self.pulse.uploader.q1instrument.get_input_ranges(ch_name)
            sequencer = self.station[seq_def.module_name].sequencers[seq_def.seq_nr]
            ch_data /= in_ranges[seq_def.channels]
            sequencer.set_acquisition_mock_data(ch_data)
        elif backend in ['Keysight', 'Keysight_QS', 'Tektronix_5014']:
            dig_ch = self.pulse.digitizer_channels[ch_name]
            dig = self.pulse.digitizers[dig_ch.module_name]
            if dig_ch.iq_input:
                ch_re, ch_im = dig_ch.channel_numbers
                dig.set_data(ch_re, ch_data.real)
                dig.set_data(ch_im, ch_data.imag)
            else:
                dig.set_data(dig_ch.channel_number, ch_data)
        else:
            raise Exception(f'unknown backend {backend}')

    def plot_awgs(self, sequence, index=None, print_acquisitions=False,
                  analogue_out=False, savefig=False,
                  **kwargs):
        self.last_sequence = sequence
        job = sequence.upload(index)
        sequence.play(index)
        pulse = self.pulse
        if savefig:
            ion_ctx = pt.ioff()
        for awg in list(pulse.awg_devices.values()) + list(pulse.digitizers.values()):
            if hasattr(awg, 'plot'):
                pt.figure()
                render_kwargs = {}
                if analogue_out:
                    render_kwargs['analogue'] = True
                awg.plot(**render_kwargs)
                # awg.plot(discrete=True)
                pt.legend()
                pt.grid()
                pt.ylabel('amplitude [V]')
                pt.xlabel('time [ns]')
                pt.title(f'output {awg.name}')
                for (method, arguments) in kwargs.items():
                    getattr(pt, method)(*arguments)
                if savefig:
                    self._savefig()
        if savefig and ion_ctx.wasinteractive:
            pt.ion()

        if print_acquisitions:
            backend = self._configuration['backend']
            if backend == 'Keysight':
                print(sequence.hw_schedule.sequence_params)
            elif backend == 'Keysight_QS':
                print(sequence.hw_schedule.sequence_params)
                for dig in pulse.digitizers.values():
                    dig.describe()
            elif backend == 'Qblox':
                print('*** See .q1asm file for acquisition timing ***', flush=True)
            elif backend == 'Tektronix_5014':
                print('triggers:', job.digitizer_triggers)
            else:
                print('No acquisition info for backend ' + backend)

#    def plot_measurement(self, sequence, m_param):
#        # average n_rep
#        # time trace...
#        s_params = sequence.params
#        n_params = len(s_params)
#        data = m_param()
#        if n_params == 0:
#            print({name:values[0] for name,values in zip(m_param.names, data)})
#        elif n_params == 1:
#            for name in m_param.names:
#                pt.figure()
#                pt.legend()
#                pt.grid()
#                pt.ylabel('amplitude [V]')
#                pt.xlabel('time [ns]')
#                pt.title(f'name')
#                self._savefig()

    def plot_segments(self, segments, index=(0,), channels=None, awg_output=True,
                      savefig=False):
        # TODO: fix index if ndim > 1
        if savefig:
            ion_ctx = pt.ioff()
        for s in segments:
            pt.figure()
            pt.title(f'Segment index:{index}')
            s.plot(index, channels=channels, render_full=awg_output)
            if savefig:
                self._savefig()
        if savefig and ion_ctx.wasinteractive:
            pt.ion()

#    def plot_ds(self, ds):
#        runner = self._configuration['runner']
#        if runner == 'core_tools':
#            pass
#        elif runner == 'qcodes':
#            pass

    def _savefig(self):
        backend = self._configuration['backend']
        self.n_plots += 1
        pt.savefig(f'figure_{self.n_plots}-{backend}.png')
        pt.close()


init_logging()
context = Context()
