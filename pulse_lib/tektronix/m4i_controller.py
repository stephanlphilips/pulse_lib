from typing import Optional, List

import numpy as np
import pyspcm

from pulse_lib.uploader.digitizer_triggers import DigitizerTriggers


def notnone(*values):
    for value in values:
        if value is not None:
            return value
    return None


class M4iControl:
    def __init__(self, digitizer,
                 box_averages: Optional[float]  =None):
        self._dig = digitizer
        self._box_averages = box_averages
        self._digitizer_triggers = None

    def _cached_set(self, par_name, value):
        param = self._dig.parameters[par_name]
        if param.cache.valid and param.cache() == value:
            return
        else:
            param(value)

    def _cached_get(self, par_name):
        param = self._dig.parameters[par_name]
        return param.cache()

    def configure_acquisitions(self,
                               digitizer_triggers: DigitizerTriggers,
                               n_rep: Optional[int],
                               average_repetitions: bool = False):
        self._digitizer_triggers = digitizer_triggers
        n_triggers = len(digitizer_triggers.triggers)
        self._configure(sorted(digitizer_triggers.active_channels),
                        digitizer_triggers.t_measure,
                        n_triggers=n_triggers,
                        n_rep=n_rep,
                        data_sample_rate=digitizer_triggers.sample_rate,
                        average_repetitions=average_repetitions)

    def _configure(self,
                   channels: List[int],
                   t_measure: float,
                   n_triggers: Optional[int] = None,
                   n_rep: Optional[int] = None,
                   data_sample_rate: Optional[float] = None,
                   average_repetitions: bool = False,
                   sw_trigger: bool = False):

        if len(channels) == 0:
            # nothing to configure.
            return

        self._channels = channels
        if len(channels) == 3:
            self._enabled_channels = [0,1,2,3]
        else:
            self._enabled_channels = channels
        self._num_ch = len(self._enabled_channels)

        self._n_triggers = n_triggers
        self._n_rep = n_rep
        self._n_seg = notnone(self._n_rep, 1) * notnone(self._n_triggers, 1)
        self._data_sample_rate = data_sample_rate
        self._average_repetitions = average_repetitions
        self._sw_trigger = sw_trigger

        self._sample_rate = self._cached_get('sample_rate')
        if self._data_sample_rate and self._data_sample_rate > self._sample_rate:
            raise ValueError(f'down-sample rate ({self._data_sample_rate/1e6:5.2f} MHz) < '
                             f'digitizer sample rate ({self._sample_rate/1e6:5.2f} MHz)')
        divisor = self._box_averages if self._box_averages is not None else 1
        self._eff_sample_rate = self._sample_rate / divisor
        self._samples_per_segment = round(self._eff_sample_rate * t_measure * 1e-9)
        if self._samples_per_segment == 0:
            raise ValueError(f'invalid settings: sample_rate:{self._sample_rate/1e6:5.2f} MHz t_measure:{t_measure} ns')

        self._cached_set('enable_channels', sum(2**ch for ch in self._enabled_channels))
        if sw_trigger:
            self._dig.trigger_or_mask(pyspcm.SPC_TMASK_SOFTWARE)
        else:
            self._dig.trigger_or_mask(pyspcm.SPC_TMASK_EXT0)
        if self._box_averages:
            self._dig.box_averages(self._box_averages)
        self._dig.setup_multi_recording(self._samples_per_segment,
                                        n_triggers=self._n_seg,
                                        boxcar_average=self._box_averages is not None)

    def get_data(self):
        if self._sw_trigger:
            self._dig.start_triggered()

        m4i_seg_size = self._dig.segment_size()
        memsize = self._dig.data_memory_size()
        pretrigger = self._dig.pretrigger_memory_size()
        n_seg = memsize / m4i_seg_size

        if n_seg != self._n_seg:
            raise Exception(f'Acquisition failed: n_seg mismatch: {n_seg} != {self.n_seg}')

        acq_shape = (self._num_ch, )
        if self._n_rep:
            acq_shape += (self._n_rep, )
        if self._n_triggers:
            acq_shape += (self._n_triggers, )
        acq_shape += (m4i_seg_size, )

        data_raw = self._dig.get_data()
        # print(f'data: {len(data_raw)} {memsize}, seg:{m4i_seg_size}, pre:{pretrigger}, n_seg:{n_seg}')

        # reshape and remove pretrigger
        data = np.reshape(data_raw, acq_shape)
        data = data[..., pretrigger:pretrigger+self._samples_per_segment]

        # filter channels
        if len(self._channels) == 3:
            selection = list(self._channels)
            data = data[selection]

        # aggregate samples (down-sampling / time average)
        if self._data_sample_rate is None:
            data = np.mean(data, axis=-1)
        else:
            step = self._eff_sample_rate / self._data_sample_rate
            boundaries = np.floor(np.arange(0, self._samples_per_segment, step)+0.5).astype(int)
            if self._samples_per_segment - boundaries[-1] > 0.8*step:
                boundaries = np.concatenate([boundaries, [self._samples_per_segment]])
            res = np.empty(data.shape[:-1] + (len(boundaries)-1,))
            for i in range(len(boundaries)-1):
                res[..., i] = np.mean(data[..., boundaries[i]:boundaries[i+1]], axis=-1)
            data = res

        # filter acquisitions: the number of acquisitions per channel can differ
        if self._digitizer_triggers:
            res = []
            for i,ch in enumerate(self._channels):
                sel = self._digitizer_triggers.get_channel_indices(ch)
                if len(sel):
                    ch_data = data[i]
                    if self._n_rep:
                        ch_data = ch_data[:, sel]
                        if self._average_repetitions:
                            ch_data = np.mean(ch_data, axis=0)
                    else:
                        ch_data = ch_data[sel]
                    res.append(ch_data.flatten())
        else:
            res = [d.flatten() for d in data]

        return res

    def actual_acquisition_points(self, duration, sample_rate):
        dig_sample_rate = self._cached_get('sample_rate')
        divisor = self._box_averages if self._box_averages is not None else 1
        eff_sample_rate = dig_sample_rate / divisor
        n_dig_samples = round(eff_sample_rate * duration * 1e-9)
        step = eff_sample_rate / sample_rate
        n_samples = int(n_dig_samples / step + 0.2)
        return n_samples, 1e9/eff_sample_rate
