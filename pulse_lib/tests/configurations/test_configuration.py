import logging
import os
import math
from collections.abc import Sequence
import matplotlib.pyplot as pt
import numpy as np

import qcodes as qc
import qcodes.logger as logger
from qcodes.logger import start_all_logging
from qcodes.data.data_set import DataSet
from qcodes.data.io import DiskIO

from ruamel.yaml import YAML

from pulse_lib.tests.utils.qc_run import qc_run
from pulse_lib.base_pulse import pulselib
from pulse_lib.virtual_channel_constructors import IQ_channel_constructor
from pulse_lib.tests.hw_schedule_mock import HardwareScheduleMock
from pulse_lib.schedule.tektronix_schedule import TektronixSchedule

try:
    import core_tools as ct
    from core_tools.sweeps.sweeps import do0D
    _ct_imported = True
    _ct_configured = False
except:
    _ct_imported = False


def init_logging():
    start_all_logging()
    logger.get_console_handler().setLevel(logging.WARN)
    logger.get_file_handler().setLevel(logging.DEBUG)


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
                    if module.is_qcm_type:
                        awgs.append(module)
                    else:
                        digs.append(module)
        else:
            for name,component in station.components.items():
                if name.startswith('AWG'):
                    awgs.append(component)
                if name.startswith('Dig'):
                    digs.append(component)

        cfg = self._configuration
        backend = cfg['backend']
        if backend in ['Keysight', 'KeysightQS']:
#            for awg in awgs:
#                # anti-ringing filter
#                awg.set_digital_filter_mode(3)
            for dig in digs:
                # Set mode AVERAGE
                dig.set_acquisition_mode(1)

        self.awgs = awgs
        self.digitizers = digs

    def init_pulselib(self, n_gates=0, n_qubits=0, n_markers=0,
                      n_sensors=0, rf_sources=False,
                      virtual_gates=False, finish=True):
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
        for awg_name,gates in cfg['awg_channels'].items():
            for i,gate in enumerate(gates):
                if backend != 'Qblox':
                    i += 1
                gate_map[gate] = (awg_name,i)

        gates = []
        for i in range(n_gates):
            gate = f'P{i+1}'
            gates.append(gate)
            awg_name,channel = gate_map[gate]
            if awg_name not in pulse.awg_devices:
                pulse.add_awg(getattr(station, awg_name))
            pulse.define_channel(gate, awg_name, channel)
            pulse.add_channel_compensation_limit(gate, (-100, 50))
#        pulse.add_channel_attenuation(name, 0.2)
#        pulse.add_channel_delay(name, value)

        if virtual_gates:
            n_gates = len(gates)
            matrix = np.diag([0.9]*n_gates) + 0.1
            pulse.add_virtual_matrix(
                    name='virtual-gates',
                    real_gate_names=gates,
                    virtual_gate_names=['v'+gate for gate in gates],
                    matrix=matrix
                    )

        for i in range(n_markers):
            self._add_marker(f'M{i+1}')

        n_iq = math.ceil(n_qubits/2)
        for i in range(n_iq):
            I,Q = f'I{i+1}', f'Q{i+1}',
            awg,channel_I = gate_map[I]
            awg,channel_Q = gate_map[Q]
            if awg not in pulse.awg_devices:
                pulse.add_awg(station.components[awg])
            pulse.define_channel(I, awg, channel_I)
            pulse.define_channel(Q, awg, channel_Q)
            pulse.add_channel_delay(I, -40)
            pulse.add_channel_delay(Q, -40)
            sig_gen = station.components[f'sig_gen{i+1}']

            iq_channel_name = f'IQ{i+1}'
            if i == 0:
                iq_marker = 'M_IQ'
                self._add_marker(iq_marker, setup_ns=20, hold_ns=20)
                pulse.add_channel_delay(iq_marker, -40)
            else:
                iq_marker = ''
            pulse.define_iq_channel(iq_channel_name, i_name=I, q_name=Q,
                                    marker_name=iq_marker)
            pulse.set_iq_lo(iq_channel_name, sig_gen.frequency)

            sig_gen.frequency(2.400e9 + i*0.400e9)
            # LO freqs: 2.400, 2.800
            # qubit freqs: 2.450, 2.550, 2.650, 2.750
            for j in range(2):
                qubit = 2*i+j+1
                if qubit < n_qubits+1:
                    idle_frequency = 2.350e9 + qubit*0.100e9
                    pulse.define_qubit_channel(f"q{qubit}", iq_channel_name, idle_frequency)

        if n_sensors > 0 and backend in ['Keysight']:
            pulse.configure_digitizer = True
        for i in range(n_sensors):
            sensor = f'SD{i+1}'
            digitizer_name,channel = cfg['sensors'][sensor]
            if digitizer_name not in pulse.digitizers:
                pulse.add_digitizer(getattr(station, digitizer_name))
            pulse.define_digitizer_channel(sensor, digitizer_name, channel)

        if n_sensors > 0 and backend == 'Tektronix_5014':
            self._add_marker('M_M4i')
            pulse.add_digitizer_marker('Dig1', 'M_M4i')

        if rf_sources:
            for sensor,params in cfg['rf'].items():
                if sensor not in pulse.digitizer_channels:
                    continue
                if backend == 'Qblox':
                    pulse.digitizer_channels[sensor].iq_out = True
                    pulse.set_digitizer_frequency(sensor, params['frequency'])
                    pulse.set_digitizer_rf_source(sensor,
                                                  output=params['output'],
                                                  amplitude=params['amplitude'],
                                                  mode='pulsed',
                                                  startup_time_ns=params['startup_time'])
                else:
                    pulse.set_digitizer_rf_source(sensor,
                                                  output=params['output'],
                                                  mode='pulsed',
                                                  startup_time_ns=params['startup_time'])

        if backend == 'Tektronix_5014':
            # pulselib always wants a digitizer for Tektronix
            if 'Dig1' not in pulse.digitizers:
                pulse.add_digitizer(station.components['Dig1'])

        if finish:
            pulse.finish_init()

        return pulse

    def _add_marker(self, name, setup_ns=0, hold_ns=0):
        cfg = self._configuration
        pulse = self.pulse
        awg,channel = cfg['markers'][name]
        if isinstance(channel, Sequence):
            channel = tuple(channel)
        if awg not in pulse.awg_devices:
            pulse.add_awg(self.station.components[awg])
        pulse.define_marker(name, awg, channel, setup_ns=setup_ns, hold_ns=hold_ns)

    def add_hw_schedule(self, sequence):
        cfg = self._configuration
        schedule = cfg.get('schedule', None)
        if schedule == 'Mock':
            sequence.set_hw_schedule(HardwareScheduleMock())
        elif schedule == 'HVI2':
            hvi2_schedule = getattr(self, 'hvi2_schedule', None)
            if hvi2_schedule == None:
                from core_tools.HVI2.hvi2_schedule_loader import Hvi2ScheduleLoader
                self.hvi2_schedule = Hvi2ScheduleLoader(self.pulse, "SingleShot")
            sequence.set_hw_schedule(self.hvi2_schedule)
        elif schedule == 'TektronixM4i':
            sequence.set_hw_schedule(TektronixSchedule(self.pulse))

    def run(self, name, sequence, *params, silent=False):
        global _ct_configured
        runner = self._configuration['runner']
        if runner == 'qcodes':
            path = 'C:/measurements/test_pulselib'
            DataSet.default_io = DiskIO(path)
            return qc_run(name, sequence, *params, quiet=silent)

        elif runner == 'core_tools':
            if not _ct_imported:
                raise Exception('core_tools import failed')
            if not _ct_configured:
                ct.configure(os.path.join(self._dir, 'ct_config.yaml'))
                _ct_configured = True
            ct.set_sample_info(sample=self.configuration_name)
            return do0D(sequence, *params, name=name, silent=silent).run()

        else:
            print(f'no implementation for {runner}')

    def plot_awgs(self, sequence, index=None, print_acquisitions=False,
                  analogue_out=False,
                  **kwargs):
        job = sequence.upload(index)
        sequence.play(index)
        pulse = self.pulse
        pt.ioff()
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
                self._savefig()

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

    def plot_segments(self, segments, index=(0,), channels=None, awg_output=True):
        # TODO: fix index if ndim > 1
        pt.ioff()
        for s in segments:
            pt.figure()
            pt.title(f'Segment index:{index}')
            s.plot(index, channels=channels, render_full=awg_output)
            self._savefig()

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


#from core_tools.HVI2.hvi2_schedule_loader import Hvi2ScheduleLoader
#
#def cleanup_instruments():
#    try:
#        oldLoader.close_all()
#    except: pass
#    oldLoader = Hvi2ScheduleLoader

