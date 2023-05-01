import numpy as np
from dataclasses import dataclass
from typing import Union

def get_modulation(envelope_generator, duration):
    if envelope_generator is None:
        am_envelope = 1.0
        pm_envelope = 0.0
    else:
        am_envelope = envelope_generator.get_AM_envelope(duration, 1.0)
        pm_envelope = envelope_generator.get_PM_envelope(duration, 1.0)
    return am_envelope, pm_envelope


@dataclass
class SineWaveform:
    duration: int
    frequency: float = None
    phase: float = 0
    amod: Union[None, float, np.ndarray] = None
    phmod: Union[None, float, np.ndarray] = None
    offset: int = 0

    def __eq__(self, other):
        res = (self.duration == other.duration
                and self.frequency == other.frequency
                and self.phase == other.phase
                and np.all(self.amod == other.amod)
                and np.all(self.phmod == other.phmod)
                )
        return res

    def render(self, sample_rate=1e9):
        total_phase = self.phase + self.phmod
        n = int(self.duration)
        t = np.arange(n)
        data = self.amod * np.sin(2*np.pi*self.frequency/sample_rate*t + total_phase)
        if self.offset:
            result = np.zeros(n + self.offset)
            result[self.offset:] = data
        else:
            result = data
        return result

    def render_iq(self, sample_rate=1e9):
        total_phase = self.phase + self.phmod
        n = int(self.duration)
        t = np.arange(n)
        cycles = 2*np.pi*self.frequency/sample_rate*t + total_phase
        I,Q = (self.amod * np.cos(cycles), self.amod * np.sin(cycles))
        if not self.offset:
            return I, Q
        else:
            resultI = np.zeros(n + self.offset)
            resultI[self.offset:] = I
            resultQ = np.zeros(n + self.offset)
            resultQ[self.offset:] = Q
            return resultI, resultQ


