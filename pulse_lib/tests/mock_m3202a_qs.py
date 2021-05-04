import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as pt

from qcodes.instrument.base import Instrument


'''
SequencerInstrument

SequencerChannel:
    * BB / IQ
    * frequency
    * Phase I/Q, Amplitude I/Q
    * BB prescaler (video mode)
    * Waveform Memory
    * Sequence

MarkerChannel
'''

# for QuantumSequencerInstrument
#class ChannelPair:
#    _12 = 12
#    _34 = 34
#
#class ChannelMode:
#    BB = 1
#    IQ = 2

def not_none(value, default):
    return default if value is None else value

@dataclass
class InstructionBase:
    address: int
    wait_after: float
    jump_address: Optional[int] = None

@dataclass
class AwgInstruction(InstructionBase):
    wave_number: Optional[int] = None
    index_register: Optional[int] = None


@dataclass
class Waveform:
    offset: int
    duration: int
    amplitude: float
    am_envelope: Union[np.ndarray, float]
    frequency: float
    pm_envelope: Union[np.ndarray, float]
    prephase: Optional[float]
    postphase: Optional[float]
    start_frequency: float
    end_frequency: float
    append_zero: bool

    def render(self, starttime, phase): # @@@ use LO i.s.o. phase
        if not self.amplitude:
            return np.zeros(0)
        amplitude = self.amplitude
        t = starttime + self.offset + np.arange(self.duration)
        phase = phase + not_none(self.prephase, 0) + not_none(self.pm_envelope, 0)
        modulated_wave = 0.001 * amplitude * np.sin(2*np.pi*self.frequency*1e-9*t + phase)
        print(f' {t[0]:6.0f}, {phase:5.2f}, {self.frequency*1e-6:6.1f} MHz {self.duration} ns')
        if self.offset == 0:
            return modulated_wave
        else:
            return np.concatenate([np.zeros(self.offset), modulated_wave])

    @property
    def phase_shift(self):
        freq_diff = 0
        if self.frequency != self.end_frequency:
            freq_diff = self.frequency - self.end_frequency
        return (not_none(self.prephase, 0)
                + not_none(self.postphase, 0)
                + 2*np.pi*self.duration*freq_diff)

    def describe(self):
        print(f'{self.offset} {self.duration} a:{self.amplitude} f:{self.frequency} '
              f'pre:{self.prephase*180/np.pi:5.1f} post:{self.postphase*180/np.pi:5.1f}')

class SequencerChannel:
    def __init__(self, instrument, number):
        self._instrument = instrument
        self._number = number
        self._frequency = 0
        self._phaseI = 0
        self._phaseQ = 90
        self._waveforms = [None]*64
        self._schedule = []
        self._components = 'IQ'

    def set_baseband(self, is_baseband):
        if is_baseband:
            isI = self._number % 2
            self._components = 'QI'[isI]
            self.configure_oscillators(frequency=0,
                                       phaseI=90 if isI else 0,
                                       phaseQ=0 if isI else 90)
        else:
            self._components = 'IQ'

    def configure_oscillators(self, frequency, phaseI=0, phaseQ=90):
        self._frequency = frequency
        self._phaseI = phaseI
        self._phaseQ = phaseQ

    def upload_waveform(self, number, offset, duration,
                        amplitude, am_envelope=None, frequency=None, pm_envelope=None,
                        prephase=None, postphase=None, restore_frequency=True, append_zero=False):
        if not restore_frequency:
            # note: requires also a start_frequency for the next wave
            raise NotImplementedError()
        self._waveforms[number] = Waveform(offset, duration, amplitude, am_envelope,
                       frequency, pm_envelope, prephase, postphase,
                       self._frequency,
                       self._frequency if frequency and restore_frequency else frequency,
                       append_zero)

    def flush_waveforms(self):
        self._waveforms = [None]*64

    def load_schedule(self, schedule:List[AwgInstruction]):
        self._schedule = schedule

    def _plot(self, phase, label):
        if len(self._schedule) == 0:
            return
        starttime = 0
        phase = phase
        wave = np.zeros(0)
        print(f'{label}:{len(self._schedule)}')
        for inst in self._schedule:
            duration = inst.wait_after
            if inst.wave_number is not None:
                waveform = self._waveforms[inst.wave_number]
                data = waveform.render(starttime, phase)
                if duration == 0:
                    duration = len(data)
                wave = np.concatenate([wave, data, np.zeros(int(duration)-len(data))])
                phase += waveform.phase_shift
            else:
                wave = np.concatenate([wave, np.zeros(int(duration))])
            starttime += duration
        pt.plot(np.arange(len(wave)), wave, label=label)

    def plot(self):
#        pt.figure(self._number)
        print(f'seq {self._number}')
        if self._components=='IQ':
            self._plot(self._phaseI, label=f'{self._instrument.name}-{self._number}.I')
            self._plot(self._phaseQ, label=f'{self._instrument.name}-{self._number}.Q')
        elif self._components=='I':
            self._plot(self._phaseI, label=f'{self._instrument.name}-{self._number}')
        elif self._components=='Q':
            self._plot(self._phaseQ, label=f'{self._instrument.name}-{self._number}')

    def describe(self):
        print(f'seq {self._number} schedule')
        for inst in self._schedule:
            print(inst)
        if len(self._waveforms) > 0:
            print('waveforms')
        for wvf in self._waveforms:
            if wvf is not None:
                wvf.describe()

class MockM3202A_QS(Instrument):
    '''
    Quantum Sequencer version of M3202A mock
    '''
    def __init__(self, name, chassis, slot, marker_amplitude=1000):
        super().__init__(name)
        self._slot_number = slot
        self._chassis_numnber = chassis

        self.chassis = chassis
        self.slot = slot

        self._sequencers = {}
        for i in range(1,9):
            self._sequencers[i] = SequencerChannel(self, i)

        self.marker_table = []
        self._marker_amplitude = marker_amplitude

    def slot_number(self):
        return self._slot_number

    def chassis_number(self):
        return self._chassis_numnber

    def get_sequencer(self, number):
        return self._sequencers[number]

    def plot_marker(self):
        if len(self.marker_table) > 0:
            t = []
            values = []
            print(self.marker_table)
            for m in self.marker_table:
                t += [m[0], m[0], m[1], m[1]]
                values += [0, self._marker_amplitude/1000, self._marker_amplitude/1000, 0]

            pt.plot(t, values, ':', label=f'{self.name}-T')

    def plot(self):
        for seq in self._sequencers.values():
            seq.plot()
        self.plot_marker()


    def describe(self):
        for i,seq in self._sequencers.items():
            seq.describe()
        print('Markers:', self.marker_table)


    def config_fpga_trigger(self, source, direction, polarity):
        pass

    def load_marker_table(self, table:List[Tuple[int,int]]):
        '''
        Args:
            table: list with tuples (time on, time off)
        '''
        self.marker_table = table.copy()

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
