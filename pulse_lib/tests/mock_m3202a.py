import os
import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import xarray as xr
import scipy.signal as signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as pt

from qcodes.instrument.base import Instrument

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self):
        self._free_slots = [i for i in range(10_000)]
        self._used_slots = {}

    def allocate(self, size):
        slot = self._free_slots.pop(0)
        self._used_slots[slot] = size
        logger.info(f'allocated {slot}: {size}')
        return slot

    def free(self, slot):
        size = self._used_slots.pop(slot)
        self._free_slots.append(slot)
        logger.info(f'freed {slot}: {size}')

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
        self.digital_filter_mode = 1
        self.memory_manager = MemoryManager()
        self.channel_data = {}
        self.channel_prescaler = {}
        self.amplitudes = {}
        for i in range(4):
            self.channel_data[i+1] = []
            self.channel_prescaler[i+1] = []

        self.chassis = chassis
        self.slot = slot

    def get_idn(self):
        return dict(vendor='Pulselib', model=type(self).__name__, serial='', firmware='')

    def slot_number(self):
        return self._slot_number

    def chassis_number(self):
        return self._chassis_numnber

    def configure_marker_output(self, invert: bool = False):
        pass

    def set_waveform_limit(self, size):
        pass

    def upload_waveform(self, wave) -> WaveformReference:
        size = len(wave)
        # discretize samples
        data = (wave*2**15).astype(np.int16)
        data &= np.uint16(0xFFF8)  # 13 bit resolution
        data = data.astype(float)
        data /= 2**15
        slot = self.memory_manager.allocate(size)
        logger.info(f'{self.name}.upload_waveform({slot}, {size})')
        return WaveformReference(slot, size, self.memory_manager, data)

    def set_digital_filter_mode(self, mode):
        self.digital_filter_mode = mode

    def set_channel_amplitude(self, amplitude, channel):
        logger.info(f'{self.name}.set_channel_amplitude({amplitude}, {channel})')
        self.amplitudes[channel] = amplitude

    def set_channel_offset(self, offset, channel):
        logger.info(f'{self.name}.set_channel_offset({offset}, {channel})')

    def awg_flush(self, channel):
        logger.info(f'{self.name}.awg_flush({channel})')
        self.channel_data[channel] = []
        self.channel_prescaler[channel] = []

    def awg_stop(self, channel):
        logger.info(f'{self.name}.awg_stop({channel})')

    def awg_queue_waveform(self, channel, waveform_ref, trigger_mode, start_delay, cycles, prescaler):
        logger.info(f'{self.name}.awg_queue_waveform({channel}, {waveform_ref.wave_number}, {trigger_mode}, {start_delay}, {cycles}, {prescaler})')
        self.channel_data[channel].append(waveform_ref.waveform * self.amplitudes[channel])
        self.channel_prescaler[channel].append(prescaler)

    def awg_is_running(self, channel):
        return False

    def release_waveform_memory(self):
        pass

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

    def _upconvert_filtered(self, t, wave):
        fname = os.path.dirname(__file__) + f'/keysight_data/keysight_pulse_response_{self.digital_filter_mode}.hdf5'
        pulse_response = xr.open_dataset(fname)
        t_response = pulse_response.coords['t'].data
        response = pulse_response['y'].data / 0.77
        sr = round(1/(t_response[1]-t_response[0]))

        t = np.linspace(t[0], t[-1]+1e-9, len(t)*sr, endpoint=False)
        d = np.zeros(len(wave)*sr)
        d[::sr] = wave

        d = np.convolve(d, response)
        n_before = round(-t_response[0]*sr)
        n_after = round(t_response[-1]*sr)
        return t, d[n_before: -n_after]


    def plot(self, bias_T_rc_time=0, discrete=False, analogue=False, IQ=False, LO_f=None,
             analogue_shift=0.0):
        iq_data = {}
        for channel in range(1,5):
            data, prescaler = self.get_data_prescaler(channel)
            #print(f'{self.name}.{channel} data: {[(len(s),p) for s,p in zip(data,prescaler)]}')

            if len(data) == 0:
                continue

            wave_data = []
            biased_data = []
            t = []
            t0 = 0
            zi = [0]
            for d,p in zip(data, prescaler):
                sr = MockM3202A.convert_prescaler_to_sample_rate(p)
                if analogue:
                    n = round(1e9/sr)
                    ts = np.arange(len(d)*n)/1e9 + t0
                    t0 = ts[-1] + 1/sr
                    if n == 1:
                        wd = d
                    else:
                        wd = np.repeat(d,n)
                elif p == 0 and not discrete:
                    ts = np.arange(len(d))/sr + t0
                    t0 = ts[-1] + 1/sr
                    wd = d
                else:
                    ts = np.arange(len(d)+1)/sr + t0 -0.5e-9
                    ts = np.repeat(ts,2)[1:-1]
                    t0 = ts[-1]
                    wd = np.repeat(d,2)

                t.append(ts)
                wave_data.append(wd)
                if bias_T_rc_time:
                    alpha = bias_T_rc_time / (bias_T_rc_time + 1/sr)
                    a = [1.0, -alpha]
                    b = [alpha, -alpha]
                    biased,zi = signal.lfilter(b, a, d, zi=zi)
                    if p:
                        biased = np.repeat(biased,2)
                    biased_data.append(biased)

            wave = np.concatenate(wave_data)

            t = np.concatenate(t)*1e9
            if not analogue:
                pt.plot(t, wave, label=f'{self.name}-{channel}')
            else:
                t2 = np.concatenate([t-1.0, [t[-1]] ])
                t2 = np.repeat(t2,2)[1:-1]
                pt.plot(t2, np.repeat(wave,2), label=f'{self.name}-{channel} digital')
            if analogue:
                ta,da = self._upconvert_filtered(t, wave)
                if IQ:
                    iq_data[channel] = da
                pt.plot(ta+analogue_shift, da, label=f'{self.name}-{channel} analogue')
            if bias_T_rc_time:
                biased = np.concatenate(biased_data)
                pt.plot(t, biased, ':', label=f'{self.name}-{channel} bias-T')
        if IQ:
            pt.legend()
            pt.figure()
            for chI,chQ in [(1,2),(3,4)]:
                if chI not in iq_data or chQ not in iq_data:
                    continue
                IQ = iq_data[chI] + 1j*iq_data[chQ]
                LO = np.exp(1j*2*np.pi*LO_f*1e-9*ta)
                rf_out = LO*IQ
                rf = interp1d(ta, rf_out.real, 'quadratic')
                ta2 = np.concatenate([(ta[1:]+ta[:-1])/2, ta])
                ta2.sort()
                pt.plot(ta2, rf(ta2), label=f'{self.name}-{chI},{chQ} RF')
#            pt.legend()


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

    def set_lo_mode(self, ch, enable):
        pass

    def config_lo(self, awg_ch, osc_num, enable, frequency, amplitude):
        pass # TODO: add to plotting.

    def plot_marker(self):
        if len(self.marker_table) > 0:
            t = []
            values = []
            print(self.marker_table)
            last = 0
            ticks = 0
            for m in self.marker_table:
                # convert to on/off duration with truncation like in M3202A_fpga
                delta_on = (m[0] - last) // 5
                delta_off = (m[1] - m[0]) // 5
                last = m[1]
                on = (ticks + delta_on) * 5
                ticks += delta_on
                off = (ticks + delta_off) * 5
                ticks += delta_off

                t += [on, on, off, off]
                values += [0, self._marker_amplitude/1000, self._marker_amplitude/1000, 0]

            pt.plot(t, values, ':', label=f'{self.name}-T')

    def plot(self, bias_T_rc_time=0, discrete=False, analogue=False, IQ=False, LO_f=None,
             analogue_shift=0.0):
        super().plot(bias_T_rc_time=bias_T_rc_time, discrete=discrete, analogue=analogue,
                     IQ=IQ, LO_f=LO_f, analogue_shift=analogue_shift)
        self.plot_marker()
