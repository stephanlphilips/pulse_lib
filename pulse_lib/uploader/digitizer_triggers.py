from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Union, Optional


@dataclass
class DigitizerTriggers:
    active_channels: List[int]
    triggered_channels: Dict[float, List[int]]
    t_measure: Union[float, Dict[int, float]]
    sample_rate: Optional[float]

    @property
    def triggers(self):
        return self.triggered_channels.keys()

    def get_channel_indices(self, ch_num):
        sel = []
        for j, channels in enumerate(self.triggered_channels.values()):
            if ch_num in channels:
                sel.append(j)
        return sel

class DigitizerTriggerBuilder:
    def __init__(self,
                 default_t_measure: float,
                 sample_rate: Optional[float],
                 t_measure_per_channel: bool = False):
        self._default_t_measure = default_t_measure
        self._sample_rate = sample_rate
        self._t_measure_per_channel = t_measure_per_channel
        self._triggers: Dict[float, List[int]] = defaultdict(list)
        self._t_measure: Union[None, float, Dict[int, int]] = (
                dict() if self._t_measure_per_channel else None
                )

    def _get_t_measure(self, ch_num: int):
        if self._t_measure_per_channel:
            return self._t_measure.get(ch_num, None)
        else:
            return self._t_measure

    def _set_t_measure(self, ch_num: int,
                       t_measure: Optional[float]):
        cur_t_measure = self._get_t_measure(ch_num)
        if cur_t_measure is not None:
            if t_measure != cur_t_measure:
                raise Exception(
                        't_measure must be same for all triggers, '
                        f'channel:{ch_num}, '
                        f'{t_measure}!={cur_t_measure}')
        else:
            if self._t_measure_per_channel:
                self._t_measure[ch_num] = t_measure
            else:
                self._t_measure = t_measure

    def add_acquisition(self, ch_num: Union[int, List[int]],
                        t: float,
                        t_measure: Optional[float] = None):
        if isinstance(ch_num, int):
            ch_num = [ch_num]
        for ch in ch_num:
            if t_measure is not None:
                self._set_t_measure(ch, t_measure)
            self._triggers[t].append(ch)

    def get_result(self):
        triggers = dict(sorted(self._triggers.items()))
        times = list(self._triggers.keys())
        if self._t_measure_per_channel:
            for ch_num in triggers:
                if self._t_measure.get(ch_num, None) is None:
                    self._t_measure[ch_num] = self._default_t_measure
            t_measure_max = max(self._t_measure)
        else:
            if self._t_measure is None:
                self._t_measure = self._default_t_measure
            t_measure_max = self._t_measure
        for i in range(1, len(times)):
            if times[i] - times[i-1] < t_measure_max: # TODO add margin ?
                raise Exception(
                        f'Trigger {i} is too short after previous trigger. '
                        f'Trigger {i}:{times[i]}, trigger {i-1}:{times[i-1]}, '
                        f't_measure: {t_measure_max}')
        channels = set()
        for ch in triggers.values():
            channels |= set(ch)
        return DigitizerTriggers(sorted(channels), triggers, self._t_measure, self._sample_rate)

