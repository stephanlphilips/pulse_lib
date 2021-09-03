
import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as pt

from qcodes.instrument.base import Instrument


class MemoryManager:
    def __init__(self):
        self._free_slots = [i for i in range(50)]
        self._used_slots = {}

    def allocate(self, size):
        slot = self._free_slots.pop(0)
        self._used_slots[slot] = size
        logging.info(f'allocated {slot}: {size}')
        return slot

    def free(self, slot):
        size = self._used_slots.pop(slot)
        self._free_slots.append(slot)
        logging.info(f'freed {slot}: {size}')

@dataclass
class WaveformReference:
    wave_number: int
    size: int
    memory_manager: MemoryManager
    waveform: List

    def release(self):
        self.memory_manager.free(self.wave_number)
        self.waveform = None


# mock for M3202A / SD_AWG_Async
class MockM3202A(Instrument):

    def __init__(self, name, chassis, slot):
        super().__init__(name)
        self._slot_number = slot
        self._chassis_numnber = chassis
        self.memory_manager = MemoryManager()
        self.channel_data = {}
        self.channel_prescaler = {}
        self.amplitudes = {}
        for i in range(4):
            self.channel_data[i+1] = []
            self.channel_prescaler[i+1] = []

        self.chassis = chassis
        self.slot = slot

    def slot_number(self):
        return self._slot_number

    def chassis_number(self):
        return self._chassis_numnber

    def upload_waveform(self, wave) -> WaveformReference:
        size = len(wave)
        slot = self.memory_manager.allocate(size)
        logging.info(f'{self.name}.upload_waveform({slot}, {size})')
        return WaveformReference(slot, size, self.memory_manager, wave)

    def set_channel_amplitude(self, amplitude, channel):
        logging.info(f'{self.name}.set_channel_amplitude({amplitude}, {channel})')
        self.amplitudes[channel] = amplitude

    def set_channel_offset(self, offset, channel):
        logging.info(f'{self.name}.set_channel_offset({offset}, {channel})')

    def awg_flush(self, channel):
        logging.info(f'{self.name}.awg_flush({channel})')
        self.channel_data[channel] = []

    def awg_stop(self, channel):
        logging.info(f'{self.name}.awg_stop({channel})')

    def awg_queue_waveform(self, channel, waveform_ref, trigger_mode, start_delay, cycles, prescaler):
        logging.info(f'{self.name}.awg_queue_waveform({channel}, {waveform_ref.wave_number}, {trigger_mode}, {start_delay}, {cycles}, {prescaler})')
        self.channel_data[channel].append(waveform_ref.waveform * self.amplitudes[channel])
        self.channel_prescaler[channel].append(prescaler)

    def awg_is_running(self, channel):
        return False

    def get_data(self, channel):
        return self.channel_data[channel]

    def get_data_prescaler(self, channel):
        return self.channel_data[channel], self.channel_prescaler[channel]

    @staticmethod
    def convert_sample_rate_to_prescaler(sample_rate):
        """
        Args:
            sample_rate (float) : sample rate
        Returns:
            prescaler (int) : prescaler set to the awg.
        """
        # 0 = 1000e6, 1 = 200e6, 2 = 100e6, 3=66.7e6
        prescaler = int(200e6/sample_rate)

        return prescaler


    @staticmethod
    def convert_prescaler_to_sample_rate(prescaler):
        """
        Args:
            prescaler (int) : prescaler set to the awg.

        Returns:
            sample_rate (float) : effective sample rate the AWG will be running
        """
        # 0 = 1000e6, 1 = 200e6, 2 = 100e6, 3=66.7e6
        if prescaler == 0:
            return 1e9
        else:
            return 200e6/prescaler

    def plot(self, bias_T_rc_time=0):
        for channel in range(1,5):
            data, prescaler = self.get_data_prescaler(channel)
            print(f'{self.name}.{channel} data: {[(len(s),p) for s,p in zip(data,prescaler)]}')

            if len(data) == 0:
                continue

            wave_data = []
            corr_data = []
            corr_sum = 0
            t = []
            t0 = 0
            for d,p in zip(data, prescaler):
                sr = MockM3202A.convert_prescaler_to_sample_rate(p)
                if p == 0:
                    ts = np.arange(len(d))/sr + t0
                    t0 = ts[-1] + 1/sr
                    wd = d
                else:
                    ts = np.arange(len(d)+1)/sr + t0
                    ts = np.repeat(ts,2)[1:-1]
                    t0 = ts[-1]
                    wd = np.repeat(d,2)

                t.append(ts)
                wave_data.append(wd)
                if bias_T_rc_time:
                    corr = np.cumsum(wd) / sr / bias_T_rc_time
                    if p > 0:
                        corr /= 2
                    corr += corr_sum
                    corr_sum = corr[-1]
                    print(corr_sum)
                    corr_data.append(corr)

            wave = np.concatenate(wave_data)

            t = np.concatenate(t)*1e9
            pt.plot(t, wave, label=f'{self.name}-{channel}')
            if bias_T_rc_time:
                corr = np.concatenate(corr_data)
                pt.plot(t, wave+corr, ':', label=f'{self.name}-{channel} COR')


class MockM3202A_fpga(MockM3202A):
    '''
    Extension of M3202A with fpga programmed features:
        * markers via TriggerOut
        * local oscillators (TODO)
        * DC compensation (TODO)
    '''
    def __init__(self, name, chassis, slot, marker_amplitude=1500):
        super().__init__(name, chassis, slot)
        self.marker_table = []
        self._marker_amplitude = marker_amplitude

    def configure_marker_output(self, invert: bool = False):
        pass

    def load_marker_table(self, table:List[Tuple[int,int]]):
        '''
        Args:
            table: list with tuples (time on, time off)
        '''
        self.marker_table = table.copy()

    def plot_marker(self):
        if len(self.marker_table) > 0:
            t = []
            values = []
            print(self.marker_table)
            for m in self.marker_table:
                t += [m[0], m[0], m[1], m[1]]
                values += [0, self._marker_amplitude/1000, self._marker_amplitude/1000, 0]

            pt.plot(t, values, ':', label=f'{self.name}-T')

    def plot(self, bias_T_rc_time=0):
        super().plot(bias_T_rc_time=bias_T_rc_time)
        self.plot_marker()
