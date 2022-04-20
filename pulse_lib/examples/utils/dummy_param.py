import numpy as np

from qcodes import MultiParameter

class DummyParam(MultiParameter):
    def __init__(self, name, t_measure, n_repeat, ch=[0,1], sample_rate=1e9):
        self._t_measure = t_measure
        self._ch = ch
        self._sample_rate = sample_rate
        self._n_rep = n_repeat

        self._timetrace = tuple(np.arange(0, t_measure, 1e9/sample_rate))
        self._index = tuple(range(0, n_repeat))
        self._n_samples = len(self._timetrace)

        names = tuple()
        shapes = tuple()
        setpoints = tuple()
        setpoint_names = tuple()
        setpoint_units = tuple()
        for (i,channel) in enumerate(ch):
            names += (f't_trace{i+1}',)
            shapes += ((n_repeat,self._n_samples),)
            setpoints += ((self._index, (self._timetrace,)*self._n_rep),)
            setpoint_names += (('repetition','t'),)
            setpoint_units += (('#','ns'),)
        super().__init__(name=name, names=names, shapes=shapes,
                         units=('mV',)*len(ch),
                         setpoints=setpoints, setpoint_names=setpoint_names,
                         setpoint_labels=setpoint_names,
                         setpoint_units=setpoint_units)

    def snapshot_base(self, update=True, params_to_skip_update=None):
        snapshot = super().snapshot_base(update, params_to_skip_update)
        snapshot['t_measure'] = self._t_measure
        snapshot['sample_rate'] = self._sample_rate
        return snapshot

    def get_raw(self):
        return_data = list()
        for ch in self._ch:
            apex = int(self._n_samples / (ch+2))
            data = np.concatenate([np.linspace(0, 1, apex),
                                   np.linspace(1, 0, self._n_samples-apex)])
            result = np.zeros((self._n_rep, self._n_samples))
            result += data
            return_data.append(result)

        return return_data
