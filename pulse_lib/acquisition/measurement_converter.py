from dataclasses import dataclass, field
from typing import Tuple, List
import numpy as np
from pulse_lib.segments.segment_measurements import measurement_acquisition, measurement_expression

from qcodes import MultiParameter


@dataclass
class SetpointsSingle:
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

class SetpointsMulti:
    '''
    Pass to MultiParameter using __dict__ attribute. Example:
        spm = setpoints_multi()
        param = MultiParameter(..., **spm.__dict__)
    '''
    def __init__(self, sps_list:List[SetpointsSingle]):
        self.names = tuple(sps.name for sps in sps_list)
        self.labels = tuple(sps.label for sps in sps_list)
        self.units = tuple(sps.unit for sps in sps_list)
        self.shapes = tuple(sps.shape for sps in sps_list)
        self.setpoints = tuple(sps.setpoints for sps in sps_list)
        self.setpoint_names = tuple(sps.setpoint_names for sps in sps_list)
        self.setpoint_labels = tuple(sps.setpoint_labels for sps in sps_list)
        self.setpoint_units = tuple(sps.setpoint_units for sps in sps_list)


@dataclass
class DataSelection:
    raw: bool = False
    states: bool = False
    values: bool = False
    selectors: bool = False
    total_selected: bool = False
    accept_mask: bool = False
    iq_complex: bool =True

# TODO @@@ allow derived data

class MeasurementParameter(MultiParameter):
    def __init__(self, name, source, mc, data_filter):
        setpoints = mc.get_setpoints(data_filter)
        super().__init__(name, **setpoints.__dict__)
        self._source = source
        self._mc = mc
        self._data_filter = data_filter
        self._derived_params = {}

    def add_derived_param(self, name, func, label=None, unit='mV',
                          time_trace=False,
                          sample_rate=None,
                          setpoints=None, setpoint_units=None,
                          setpoint_labels=None, setpoint_names=None):
        '''
        Create a parameter that is derived from a trace (such as an
        average). Input of the function is the array of channels that
        would be returned from get_raw() without derived parameters.

        Args:
            name (str): name of the parameter
            func (Callable[[Dict[str,np.ndarray]], np.ndarray]): function
                calculating derived parameter
            label (Optional[str]): label for the parameter
            unit (str): unit for the parameter
            time_trace (bool): if True `func` returns a time trace with time dimension
            sample_rate (float): if not None specifies the sample rate of the time trace
            setpoints (Optional[np.ndarray]): setpoints
            setpoint_unitss (Optional[np.ndarray]): setpoint units
            setpoint_labels (Optional[np.ndarray]): setpoint labels
            setpoint_names (Optional[np.ndarray]): setpoint names
        '''
        if label is None:
            label = name

        # check the shape returned by the derived parameter
        dummy_data = {name:np.zeros(shape)
                      for name,shape in zip(self.names,self.shapes)}
        dp_shape = np.shape(func(dummy_data))
        n_dim = len(dp_shape)
        n_rep = self._mc.n_rep

        self._derived_params[name] = func
        self.names += (name,)
        self.shapes += (dp_shape,)
        self.units += (unit,)
        self.labels += (label,)

        if setpoints is None:
            # Determine setpoints from dp_shape
            n_samples = None
            if time_trace:
                if n_dim == 0 or n_dim > 2:
                    raise Exception('Expecting dimensions (time,) or (n_rep, time), '
                                    f'but got {dp_shape}')
                add_repetitions = n_dim == 2
                n_samples = dp_shape[-1]
            else:
                if n_dim > 1:
                    raise Exception('Expecting dimensions (,) or (n_rep,), '
                                    f'but got {dp_shape}')
                add_repetitions = n_dim == 1

            if add_repetitions and dp_shape[0] != n_rep:
                raise Exception('Dimension mismatch for n-repetitions. '
                                f'Expected {n_rep}, got {dp_shape[0]}')

            setpoints = tuple()
            sp_names = tuple()
            sp_units = tuple()
            if add_repetitions:
                sp_names += ('repetitions',)
                sp_units += ('',)
                setpoints += ((np.arange(n_rep),),)
            if time_trace:
                if sample_rate is None:
                    sample_rate = self._mc._sample_rate
                period = 1e9/sample_rate
                time = tuple(np.arange(n_samples) * period)
                sp_names += ('time',)
                sp_units +=  ('ns',)
                setpoints += ((time,)*n_rep,)

            sp_labels = sp_names
            self.setpoints = self.setpoints + (setpoints,)
            self.setpoint_labels = self.setpoint_labels + (sp_labels,)
            self.setpoint_names = self.setpoint_names + (sp_names,)
            self.setpoint_units = self.setpoint_units + (sp_units,)
        else:
            if setpoint_labels and setpoint_names and setpoint_units:
                self.setpoints = self.setpoints + (setpoints,)
                self.setpoint_labels = self.setpoint_labels + (setpoint_labels,)
                self.setpoint_names = self.setpoint_names + (setpoint_names,)
                self.setpoint_units = self.setpoint_units + (setpoint_units,)
            else:
                raise Exception('Setpoint names, units and labels are also required when specifying setpoints')


    def get_raw(self):
        data = self._source.get_channel_data()
        self._mc.set_channel_data(data)

        data = self._mc.get_measurement_data(self._data_filter)

        if len(self._derived_params) > 0:
            data_map = {name:values for name,values in zip(self.names, data)}
            for name,dp in self._derived_params.items():
                dp_data = dp(data_map)
                data.append(dp_data)
                data_map[name] = dp_data

        return data


class MeasurementConverter:
    def __init__(self, description, n_rep, sample_rate):
        self._description = description
        self.n_rep = n_rep
        self._sample_rate = sample_rate
        self._channel_raw = {}

        self._raw = []
        self._raw_split = []
        self._states = []
        self._selectors = []
        self._values = []
        self._total_selected = []
        self._accepted = []

        self.sp_raw = []
        self.sp_raw_split = []
        self.sp_states = []
        self.sp_selectors = []
        self.sp_values = []
        self.sp_total = []
        self.sp_mask = []
        self._generate_setpoints_raw()
        self._generate_setpoints()
        self._collect_names()

    # @@@ add raw average over periods.

    def _generate_setpoints_raw(self):
        n_rep = self.n_rep
        digitizer_channels = self._description.digitizer_channels
        for m in self._description.measurements:
            if isinstance(m, measurement_acquisition):
                sp_names = ('repetition',)
                sp_units = ('',)
                shape = (n_rep,)
                setpoints = ((np.arange(n_rep),),)
                if m.n_samples is not None:
                    period = 1e9/self._sample_rate
                    time = tuple(np.arange(m.n_samples) * period)
                    shape += (m.n_samples,)
                    sp_names += ('time',)
                    sp_units +=  ('ns',)
                    setpoints += ((time,)*n_rep,)
                channel_name = m.acquisition_channel
                channel = digitizer_channels[channel_name]

                name = f'{m.name}'
                label = f'{m.name} ({channel_name}:{m.index})'
                sp_raw = SetpointsSingle(name, label, 'mV', shape, setpoints,
                                         sp_names, sp_names, sp_units)
                self.sp_raw.append(sp_raw)

                if channel.iq_out:
                    for suffix in ['_I', '_Q']:
                        name = f'{m.name}{suffix}'
                        label = f'{m.name}{suffix} ({channel_name}{suffix}:{m.index})'
                        sp_raw = SetpointsSingle(name, label, 'mV', shape, setpoints,
                                                 sp_names, sp_names, sp_units)
                        self.sp_raw_split.append(sp_raw)
                else:
                    self.sp_raw_split.append(sp_raw)


    def _generate_setpoints(self):
        shape_raw = (self.n_rep,)
        for m in self._description.measurements:
            if isinstance(m, measurement_acquisition) and not m.has_threshold:
                # do not add to result
                continue

            name = f'{m.name}_state'
            label = name
            sp_state = SetpointsSingle(name, label, '', shape_raw,
                                       ((np.arange(shape_raw[0]),),),
                                       ('repetition', ), ('repetition',), ('', ))
            self.sp_states.append(sp_state)

            if m.accept_if is not None:
                name = f'{m.name}_selected'
                label = name
                sp_result = SetpointsSingle(name, label, '#')
                self.sp_selectors.append(sp_result)
            else:
                name = f'{m.name}_fraction'
                label = name
                sp_result = SetpointsSingle(name, label, '%')
                self.sp_values.append(sp_result)

        if len(self.sp_selectors) > 0:
            self.sp_mask.append(SetpointsSingle('mask', 'mask', '', shape_raw,
                                                ((np.arange(shape_raw[0]),),),
                                                ('repetition', ), ('repetition',), ('', )))
            self.sp_total.append(SetpointsSingle('total_selected', 'total_selected', '#'))

    def _collect_names(self):
        self._names = []
        for sp in (self.sp_raw + self.sp_states + self.sp_values
                   + self.sp_selectors + self.sp_total + self.sp_mask):
            self._names.append(sp.name)
        self._names_split = []
        for sp in (self.sp_raw_split + self.sp_states + self.sp_values
                   + self.sp_selectors + self.sp_total + self.sp_mask):
            self._names_split.append(sp.name)

    def _set_data_raw(self):
        digitizer_channels = self._description.digitizer_channels
        self._raw = []
        for m in self._description.measurements:
            if isinstance(m, measurement_acquisition):
                channel_name = m.acquisition_channel
                channel = digitizer_channels[channel_name]
                if m.n_samples is None:
                    channel_raw = self._channel_raw[channel_name][:,m.data_offset]
                else:
                    channel_raw = self._channel_raw[channel_name][:,m.data_offset:m.data_offset+m.n_samples]

                self._raw.append(channel_raw)
                if channel.iq_out:
                    self._raw_split.append(channel_raw.real)
                    self._raw_split.append(channel_raw.imag)
                else:
                    self._raw_split.append(channel_raw.real)

    def _set_states(self):
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
                result = self._channel_raw[channel_name][m.index] > m.threshold
                if m.zero_on_high:
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
        self._states = results
        if len(selectors) > 0:
            self._accepted = [accepted_mask]
            self._total_selected = [total_selected]
        self._selectors = selectors
        self._values = [np.sum(result*accepted_mask)/total_selected for result in values_unfiltered]


    def set_channel_data(self, data):
        self._channel_raw = data
        self._set_data_raw()
        self._set_states()

    def get_setpoints(self, selection):
        sp_list = []
        if selection.raw:
            if selection.iq_complex:
                sp_list += self.sp_raw
            else:
                sp_list += self.sp_raw_split
        if selection.states:
            sp_list += self.sp_states
        if selection.values:
            sp_list += self.sp_values
        if selection.selectors:
            sp_list += self.sp_selectors
        if selection.total_selected:
            sp_list += self.sp_total
        if selection.accept_mask:
            sp_list += self.sp_mask
        return SetpointsMulti(sp_list)

    def get_measurement_data(self, selection):
        data = []
        if selection.raw:
            if selection.iq_complex:
                data += self._raw
            else:
                data += self._raw_split
        if selection.states:
            data += self._states
        if selection.values:
            data += self._values
        if selection.selectors:
            data += self._selectors
        if selection.total_selected:
            data += self._total_selected
        if selection.accept_mask:
            data += self._accepted
        return data

    def get_all_measurements(self, iq_complex=True):
        selection = DataSelection(raw=True, states=True, values=True,
                                  selectors=True, total_selected=True,
                                  accept_mask=True, iq_complex=iq_complex)
        result = {}
        names = self._names if iq_complex else self._names_split
        for name,value in zip(names, self.get_measurement_data(selection)):
            result[name] = value
        return result


