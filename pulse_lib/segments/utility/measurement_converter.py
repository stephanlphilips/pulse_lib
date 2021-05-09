from dataclasses import dataclass, field
from typing import Tuple, List
import numpy as np
from pulse_lib.configuration.physical_channels import digitizer_channel, digitizer_channel_iq
from pulse_lib.segments.segment_measurements import measurement_acquisition, measurement_expression

from qcodes import MultiParameter


@dataclass
class setpoints_single:
    name : str
    label : str
    unit : str
    shape : Tuple[int] = field(default_factory=tuple)
    setpoints : Tuple[Tuple[float]] = field(default_factory=tuple)
    setpoint_names : Tuple[str] = field(default_factory=tuple)
    setpoint_labels : Tuple[str] = field(default_factory=tuple)
    setpoint_units : Tuple[str] = field(default_factory=tuple)

    def __post_init__(self):
        self.name = self.name.replace(' ', '_')

class setpoints_multi:
    '''
    Pass to MultiParameter using __dict__ attribute. Example:
        spm = setpoints_multi()
        param = MultiParameter(..., **spm.__dict__)
    '''
    def __init__(self, sps_list:List[setpoints_single]):
        self.names = tuple(sps.name for sps in sps_list)
        self.labels = tuple(sps.label for sps in sps_list)
        self.units = tuple(sps.unit for sps in sps_list)
        self.shapes = tuple(sps.shape for sps in sps_list)
        self.setpoints = tuple(sps.setpoints for sps in sps_list)
        self.setpoint_names = tuple(sps.setpoint_names for sps in sps_list)
        self.setpoint_labels = tuple(sps.setpoint_labels for sps in sps_list)
        self.setpoint_units = tuple(sps.setpoint_units for sps in sps_list)

class _MeasurementParameter(MultiParameter):
    def __init__(self, setpoints, getter):
        super().__init__('measurement', **setpoints.__dict__)
        self._getter = getter
        self.index = None
        self.dig = None
        self.mc = None

    def get_raw(self):
        data = self.dig.measure.get_data()
        print(data)
        self.mc.set_data(data, self.index)
        

        return self._getter()

    def setUpParam(self, mc, dig):
        '''
        set up the measurment parameter

        Args:
            mc (measurement_converter) : measurement convertor objct that generated this parmeter
            dig (dig) : digitzer used in the measurement
        '''
        self.mc = mc
        self.dig = dig

    def setIndex(self, idx):
        '''
        set index that is currenly playing

        Args:
            idx (tuple) : current index
        '''
        self.index = idx


class measurement_converter:
    def __init__(self, description, n_rep):
        self._description = description
        self.n_rep = n_rep
        self._channel_raw = {}
        self._raw = []
        self._states = []
        self._accepted = None
        self.sp_raw = []
        self.sp_states = []
        self.sp_selectors = []
        self.sp_values = []
        self.generate_setpoints_raw()
        self.generate_setpoints()


    def generate_setpoints_raw(self):
        digitizer_channels = self._description.digitizer_channels
        shape_raw = (self.n_rep,)
        for m in self._description.measurements:
            if isinstance(m, measurement_acquisition):
                channel_name = m.acquisition_channel
                channel = digitizer_channels[channel_name]
                channel_suffixes = ['_I', '_Q'] if isinstance(channel, digitizer_channel_iq) else ['']

                for suffix in channel_suffixes:
                    name = f'RAW_{m.name}{suffix}'
                    label = f'RAW {m.name}{suffix} ({channel_name}{suffix}:{m.index})'
                    sp_raw = setpoints_single(name, label, 'mV', shape_raw,
                                              ((np.arange(shape_raw[0]),),),
                                              ('repetition',), ('repetition',), ('',))
                    self.sp_raw.append(sp_raw)

    def generate_setpoints(self):
        shape_raw = (self.n_rep,)
        for m in self._description.measurements:
            if isinstance(m, measurement_acquisition) and not m.has_threshold:
                # do not add to result
                continue

            name = f'State_{m.name}'
            label = f'State {m.name}'
            sp_state = setpoints_single(name, label, '', shape_raw,
                                        ((np.arange(shape_raw[0]),),),
                                        ('repetition', ), ('repetition',), ('', ))
            self.sp_states.append(sp_state)

            if m.accept_if is not None:
                sp_result = setpoints_single(f'{m.name}_selected', f'{m.name} selected', '#')
                self.sp_selectors.append(sp_result)
            else:
                sp_result = setpoints_single(m.name, m.name, '%')
                self.sp_values.append(sp_result)

        self.sp_mask = setpoints_single('mask', 'mask', '') ####
        self.sp_total = setpoints_single('total_selected', 'total_selected', '#')


    def _set_channel_raw(self, data, index):
        digitizer_channels = self._description.digitizer_channels

        self._channel_raw = {}
        # set raw values
        for channel in digitizer_channels.values():
            acquisitions = self._description.acquisitions[channel.name][index]
            if isinstance(channel, digitizer_channel):
                ch_raw = data[channel.channel_number-1]
            elif isinstance(channel, digitizer_channel_iq):
                ch_raw = data[channel.channel_numbers[0]-1] + 1j * data[channel.channel_numbers[1]-1]
                if channel.phase is not None:
                    ch_raw = (ch_raw * np.exp(1j*channel.phase)).real
            else:
                raise NotImplementedError(f'Unknown channel type {type(channel)}')
            self._channel_raw[channel.name] = ch_raw.reshape((-1, len(acquisitions.data))).T


    def _set_data_raw(self, data):
        for m in self._description.measurements:
            if isinstance(m, measurement_acquisition):
                channel_name = m.acquisition_channel
                channel_raw = self._channel_raw[channel_name][m.index]
                if np.iscomplexobj(channel_raw):
                    self._raw.append(channel_raw.real)
                    self._raw.append(channel_raw.imag)
                else:
                    self._raw.append(channel_raw)

    def _set_states(self, data, index):
        # iterate through measurements and keep last named values in dictionary
        results = []
        selectors = []
        values_unfiltered = []
        last_result = {}
        accepted_mask = np.ones(self.n_rep, dtype=np.int)
        for m in self._description.measurements:
            if isinstance(m, measurement_acquisition):
                if not m.has_threshold:
                    # do not add to result
                    continue

                channel_name = m.acquisition_channel
                acq = self._description.acquisitions[channel_name][index].data[m.index]
                result = self._channel_raw[channel_name][m.index] > acq.threshold
                if acq.zero_on_high:
                    result = not result
                result = result.astype(int)
            elif isinstance(m, measurement_expression):
                result = m.expression.evaluate(last_result)
            else:
                raise NotImplementedError(f'Unknown measurement type {type(m)}')

            if m.accept_if is not None:
                accepted = result == m.accept_if
                accepted_mask *= accepted
                selectors.append(np.sum(accepted))
            else:
                values_unfiltered.append(result)
            last_result[m.name] = result
            results.append(result)

        total_selected = np.sum(accepted_mask)
        self.total_selected = total_selected
        self._states = results
        self._accepted = accepted_mask
        self._selectors = selectors
        self._values = [np.sum(result*accepted_mask)/total_selected for result in values_unfiltered]


    def set_data(self, data, index=(0,)):
        # todo add module_name to digitizer data

        self._set_channel_raw(data, index)
        self._set_data_raw(data)
        self._set_states(data, index)


    def raw(self):
        return _MeasurementParameter(setpoints_multi(self.sp_raw),
                                     lambda: self._raw)

    def states(self):
        return _MeasurementParameter(setpoints_multi(self.sp_states),
                                     lambda: self._states)

    def selectors(self):
        return _MeasurementParameter(setpoints_multi(self.sp_selectors),
                                     lambda: self._selectors)


    def values(self):
        return _MeasurementParameter(setpoints_multi(self.sp_values),
                                     lambda: self._values)

    def all_results(self):
        setpoints = setpoints_multi(self.sp_raw + self.sp_states + self.sp_selectors
                                    + self.sp_values + [self.sp_total])
        getter = lambda: self._raw + self._states + self._selectors + self._values + [self.total_selected]

        return _MeasurementParameter(setpoints, getter)


