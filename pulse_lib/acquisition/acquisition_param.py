from qcodes.instrument.parameter import MultiParameter
import numpy as np

class AcquisitionParam(MultiParameter):
    """
    A qcodes parameter that returns an array of channels.
    For each channel the data is returned in an 0, 1, 2, or 3-dimensional
    array. The dimensions of the channel data are [repetition, trigger, time],
    each dimensions being optional:
        * repetition is present when `n_rep` is specified and average_repetition=False
        * trigger is present when `n_triggers` is specified
        * time is present when t_measure and sample_rate are specified

    Args:
        acquisition_hardware (object): object implementing get_measurement_data()
        name (st]): the local name of the whole parameter. Should be a valid
            identifier, ie no spaces or special characters. If this parameter
            is part of an Instrument or Station, this is how it will be
            referenced from that parent, ie ``instrument.name`` or
            ``instrument.parameters[name]``
        names (List[str]): A name for each channel returned by the digitizer.
        labels (Optional[Tuple[str]]): A label for each channel.
        units (Optional[Tuple[str]]): The unit of measure for each channel
        n_rep (Optional[int]): number of repetitions of the measurement.
        n_triggers (Optional[int]): number of read-outs in 1 measurement.
        t_measure (Optional[float]): measurement time of single trigger in ns.
        sample_rate (Optional[int]): sample rate in Hz.
        average_repetitions (bool): Average over repetitions (removes repetition dimension)

    Note:
        `start_func` can be used to upload and play sequence.
        `start_func` is called in every get_raw() call.
    """
    def __init__(self, acquisition_hardware,
                 name, names, labels=None, units=None,
                 n_rep=None, n_triggers=None,
                 t_measure=None, sample_rate = None,
                 average_repetitions=False,
                 ):


        if labels is None:
            labels = list(names)

        if units is None:
            units = ['mV']*len(names)

        super().__init__(name=name,
             names=names, labels=labels, units=units,
             shapes = ((),)*len(names))

        self.num_ch = len(names)
        self.raw_names = list(names)
        self.acquisition_hardware = acquisition_hardware
        self.n_rep = n_rep

        self.n_triggers = n_triggers
        self.t_measure = t_measure
        self.sample_rate = sample_rate
        self.average_repetitions = average_repetitions

        add_repetitions = self.n_rep and not average_repetitions
        add_triggers = self.n_triggers is not None
        add_time = sample_rate is not None and t_measure is not None
        if add_time:
            self.t_sample = 1e9/self.sample_rate
            self.n_samples = round(self.t_measure / self.t_sample)
        else:
            self.n_samples = None

        shapes = tuple()
        setpoints = tuple()
        setpoint_names = tuple()
        setpoint_labels = tuple()
        setpoint_units = tuple()
        self.acq_shapes = []

        for i,name in enumerate(self.raw_names):
            shape, sp, sp_names, sp_labels, sp_units = \
                self._shape_and_setpoints(add_time, add_triggers, add_repetitions)
            shapes = shapes + (shape,)
            setpoints = setpoints + (sp,)
            setpoint_names = setpoint_names + (sp_names,)
            setpoint_labels = setpoint_labels + (sp_labels,)
            setpoint_units = setpoint_units + (sp_units,)
            acq_shape = (shape
                         if not (average_repetitions and self.n_rep)
                         else shape + (self.n_rep,))
            self.acq_shapes.append(acq_shape)
        self.shapes = shapes
        self.setpoints = setpoints
        self.setpoint_names = setpoint_names
        self.setpoint_labels = setpoint_labels
        self.setpoint_units = setpoint_units

        self.derived_params = {}

    def _shape_and_setpoints(self, add_time, add_triggers, add_repetitions):
        shape = ()
        setpoints = ()
        sp_names = ()
        sp_labels = ()
        sp_units = ()

        # build dimensions from last to first for correct setpoints
        if add_time:
            setp_time = tuple(np.arange(self.n_samples)*self.t_sample)
            shape = (self.n_samples,)
            setpoints = (setp_time,)
            sp_names = ('t',)
            sp_labels = ('time',)
            sp_units = ('ns',)

        if add_triggers:
            setp_trigger = tuple(range(1,self.n_triggers+1))
            shape = (self.n_triggers,) + shape
            setpoints =  (setp_trigger,) + ( (setpoints*self.n_triggers,) if setpoints else () )
            sp_names = ('trigger',) + sp_names
            sp_labels = ('trigger',) + sp_labels
            sp_units = ('',) + sp_units

        if add_repetitions:
            setp_rep = tuple(range(1,self.n_rep+1))
            shape = (self.n_rep,) + shape
            setpoints =  (setp_rep,) + ( (setpoints*self.n_rep,) if setpoints else () )
            sp_names = ('N',) + sp_names
            sp_labels = ('repetitions',) + sp_labels
            sp_units = ('',) + sp_units
        return shape, setpoints, sp_names, sp_labels, sp_units

    def add_derived_param(self, name, func, label=None, unit='mV',
                          reduce_time=False, reduce_triggers=False,
                          reduce_repetitions=False,
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
            reduce_time (bool): if True `func` reduce the time dimension
            reduce_triggers (bool): if True `func` reduce the trigger dimension
            reduce_repetitions (bool): if True `func` reduce the repetitions dimension
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

        self.derived_params[name] = func
        self.names = self.names + (name,)
        self.shapes = self.shapes + (dp_shape,)
        self.units.append(unit)
        self.labels.append(label)

        if setpoints is None:
            add_time = self.n_samples is not None and not reduce_time
            add_triggers = self.n_triggers is not None and not reduce_triggers
            add_repetitions = (self.n_rep is not None
                               and not self.average_repetitions
                               and not reduce_repetitions)

            shape, setpoints, sp_names, sp_labels, sp_units = \
                self._shape_and_setpoints(add_time, add_triggers, add_repetitions)

            if shape != dp_shape:
                raise Exception(f"Shapes don't match. "
                                f"func returns {dp_shape}, expected {shape}")
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
                raise Exception('Please also supply setpoint names/units/labels')

    def get_raw(self):
        data_raw = self.acquisition_hardware.get_channel_data()
        data = {}
        for i,name in enumerate(self.raw_names):
            data[name] = np.reshape(data_raw[name], self.acq_shapes[i])

        # average
        res_volt = data

        if self.average_repetitions:
            res_volt = np.mean(res_volt, axis=1)

        for name,dp in self.derived_params.items():
            data[name] = dp(data)

        return list(data.values())

