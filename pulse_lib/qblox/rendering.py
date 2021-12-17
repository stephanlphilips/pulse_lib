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
        t = np.arange(int(self.duration))
        return self.amod * np.sin(2*np.pi*self.frequency/sample_rate*t + total_phase)

    def render_iq(self, sample_rate=1e9):
        total_phase = self.phase + self.phmod
        t = np.arange(int(self.duration))
        cycles = 2*np.pi*self.frequency/sample_rate*t + total_phase
        return (self.amod * np.cos(cycles),
                self.amod * np.sin(cycles))

def render_custom_pulse(custom_pulse, scaling, sample_rate=1e9):
    duration = custom_pulse.stop - custom_pulse.start
    data = custom_pulse.func(duration, sample_rate,
                             custom_pulse.amplitude, **custom_pulse.kwargs)
    return data * scaling

