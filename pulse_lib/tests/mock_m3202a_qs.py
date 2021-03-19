import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

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
13 bit GAWG: 40 kSa = 40 us. -> prescaler for video mode fast sweep and slow step
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
    amplitude: np.ndarray
    prephase: Optional[float]
    postphase: Optional[float]
    append_zero: bool

    def render(self, f, starttime, phase):
        amplitude = self.amplitude
        if self.postphase is not None:
            amplitude = np.concatenate((amplitude, [0.0, 0.0]))
        elif self.append_zero:
            amplitude = np.concatenate((amplitude, [0.0]))
        t = np.arange(len(amplitude)) + starttime
        return np.sin(2*np.pi*f*1e-9*t + phase + self.prephase) * amplitude * 0.001

    @property
    def phase_shift(self):
        return not_none(self.prephase, 0) + not_none(self.postphase, 0)

    def describe(self):
        print(f'{self.prephase*180/np.pi:5.1f} {self.postphase*180/np.pi:5.1f} {len(self.amplitude)} {self.amplitude}')

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

    def upload_waveform(self, number, amplitude, prephase=None, postphase=None, append_zero=False):
        self._waveforms[number] = Waveform(amplitude, prephase, postphase, append_zero)

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
                data = waveform.render(self._frequency, starttime, phase)
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
