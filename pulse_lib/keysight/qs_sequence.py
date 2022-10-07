from dataclasses import dataclass, field
from typing import Union, Optional, List

import numpy as np

def iround(value):
    return int(value + 0.5)

@dataclass
class Waveform:
    amplitude: float = 0
    am_envelope: Union[None, float, np.ndarray] = None
    frequency: float = None
    pm_envelope: Union[None, float, np.ndarray] = None
    prephase: float = 0
    postphase: float = 0
    duration: int = 0
    offset: int = 0
    restore_frequency: bool = True

    def __eq__(self, other):
        return (self.amplitude == other.amplitude
                and np.all(self.am_envelope == other.am_envelope)
                and self.frequency == other.frequency
                and np.all(self.pm_envelope == other.pm_envelope)
                and self.prephase == other.prephase
                and self.duration == other.duration
                and self.restore_frequency == other.restore_frequency
                and self.offset == other.offset
                )

@dataclass
class SequenceEntry:
    time_after: int = 0
    waveform_index: Optional[int] = None

@dataclass
class SequenceConditionalEntry:
    time_after: int = 0
    cr: int = 0
    waveform_indices: List[int] = field(default_factory=list)

@dataclass
class DigitizerSequenceEntry:
    time_after: float = 0
    t_measure: Optional[float] = None
    threshold: Optional[float] = None
    measurement_id: Optional[int] = None
    name: Optional[str] = None
    pxi_trigger: Optional[int] = None
    n_cycles : int = 1


# TODO:
#  instruction is 'raw' waveform like segment
#  flush segment when t >= t_end (aligned)

class IQSequenceBuilder:
    def __init__(self, name, t_start, lo_freq):
        self.name = name
        self.time = iround(t_start)
        self.lo_freq = lo_freq
        self.end_pulse = self.time
        self.last_instruction = None
        self.sequence = []
        self.waveforms = []

    def pulse(self, t_pulse, iq_pulse):
        # TODO: split long pulses in start + stop pulse (Rabi)
        # TODO: Frequency chirp with prescaler: pass as FM i.s.o. PM, or Chirp?

        offset = self._get_wvf_offset(t_pulse)
        waveform_index, duration = self._render_waveform(iq_pulse, self.lo_freq, offset)
        t_end = t_pulse + duration
        self._append_instruction(t_pulse, t_end, waveform_index)

    def shift_phase(self, t, phase_shift):
        # TODO @@@ merge pulse and shift when on same clock cycle

        waveform_index, duration = self._render_phase_shift(phase_shift)
        t_end = t + duration
        self._append_instruction(t, t_end, waveform_index)

    def conditional_pulses(self, t_instr, segment_start, pulses, order,
                           condition_register):

        self._wait_till(t_instr)
        # start of aligned waveform
        t_instr = self.time

        entry = SequenceConditionalEntry(cr=condition_register)
        self.sequence.append(entry)
        self.last_instruction = entry

        t_end = t_instr
        wvf_indices = []
        for pulse in pulses:
            if pulse is None:
                # a do nothing pulse
                index, duration = self._render_phase_shift(0)
                pulse_end = t_instr + duration
            elif pulse.mw_pulse is not None:
                mw_entry = pulse.mw_pulse
                t_pulse = segment_start + mw_entry.start
                wvf_offset = iround(t_pulse - t_instr)
                index, duration = self._render_waveform(mw_entry, self.lo_freq, wvf_offset,
                                                        prephase=pulse.prephase,
                                                        postphase=pulse.postphase)
                pulse_end = t_pulse + duration
            else:
                index, duration = self._render_phase_shift(pulse.prephase)
                pulse_end = t_instr + duration
            wvf_indices.append(index)
            t_end = max(t_end, pulse_end)

        self._set_pulse_end(t_end)

        for ibranch in order:
            entry.waveform_indices.append(wvf_indices[ibranch])

    def close(self):
        # set wait time of last instruction
        if self.last_instruction is not None and self.end_pulse > self.time:
            t_wait = self.end_pulse - self.time
            t = (iround(t_wait) // 5 + 1) * 5
            self.last_instruction.time_after = t
            self.last_instruction = None

    def _append_instruction(self, t_start, t_end, waveform_index):
        self._wait_till(t_start)
        entry = SequenceEntry()
        entry.waveform_index = waveform_index
        self.sequence.append(entry)
        self.last_instruction = entry
        self._set_pulse_end(t_end)

    def _set_pulse_end(self, t_end):
        self.end_pulse = max(self.end_pulse, iround(t_end))

    def _wait_till(self, t_start):
        t_instruction = (iround(t_start) // 5) * 5
        if t_instruction < self.end_pulse:
            raise Exception(f'Overlapping pulses at {t_start} of {self.name}')

        t_wait = t_instruction - self.time
        if self.last_instruction is not None and t_wait < 5:
            raise Exception(f'wait < 5 {self.name} at {t_start}')
        elif self.last_instruction is None and t_wait < 0:
            raise Exception(f'wait < 0 {self.name} at {t_start}')

        # set wait time of last instruction
        if self.last_instruction is not None:
            # Max wait time for instruction with waveform ~ 32 ms
            max_wait = 5 * (2**16-1)
            t = min(max_wait, t_wait)
            self.last_instruction.time_after = t
            self.last_instruction = None
            t_wait -= t

        # insert additional instructions till t_start
        while t_wait > 0:
            entry = SequenceEntry()
            self.sequence.append(entry)
            # Max wait time for instruction without waveform ~ 335 ms
            max_wait = 5 * (2**26-1)
            t = min(max_wait, t_wait)
            entry.time_after = t
            t_wait -= t

        self.time = t_instruction


    def _render_waveform(self, mw_pulse_data, lo_freq:float, offset:float, prephase=0, postphase=0) -> Waveform:
        # always render at 1e9 Sa/s
        duration = iround(mw_pulse_data.stop) - iround(mw_pulse_data.start)
        if mw_pulse_data.envelope is None:
            amp_envelope = 1.0
            pm_envelope = 0.0
            add_pm = False
        else:
            amp_envelope = mw_pulse_data.envelope.get_AM_envelope(duration, 1.0)
            pm_envelope = mw_pulse_data.envelope.get_PM_envelope(duration, 1.0)
            add_pm = not np.all(pm_envelope == 0)

        frequency = mw_pulse_data.frequency - lo_freq
        if abs(frequency) > 450e6:
            raise Exception(f'Waveform frequency {frequency/1e6:5.1f} MHz out of range')

        waveform = Waveform(mw_pulse_data.amplitude, amp_envelope,
                            frequency, pm_envelope,
                            mw_pulse_data.start_phase + prephase,
                            -mw_pulse_data.start_phase + postphase,
                            duration, offset)

        extra = 0
        # post_phase and pm_envelope add 2 samples, but last sample is restore of NCO
        # frequency and can be overwritten by next pulse without consequences.
        if -mw_pulse_data.start_phase + postphase != 0 or add_pm:
            extra = 1

        index = self._get_waveform_index(waveform)
        return index, duration + extra


    def _render_phase_shift(self, phase_shift) -> Waveform:
        waveform = Waveform(prephase=phase_shift, duration=2)
        index = self._get_waveform_index(waveform)
        return index, 1

    def _get_waveform_index(self, waveform:Waveform):
        try:
            index = self.waveforms.index(waveform)
        except ValueError:
            index = len(self.waveforms)
            self.waveforms.append(waveform)
        return index

    def _get_wvf_offset(self, t_pulse):
        return iround(t_pulse) % 5


class AcquisitionSequenceBuilder:
    def __init__(self, name, t_start):
        self.name = name
        self.time = t_start
        self.end_pulse = self.time
        self.last_instruction = None
        self.sequence = []

    def acquire(self, t_start, t_integrate, n_repeat=1,
                threshold=0, pxi_trigger=None):
        self._wait_till(t_start)
        entry = DigitizerSequenceEntry()
        self.sequence.append(entry)
        self.last_instruction = entry

        entry.n_cycles = n_repeat
        entry.t_measure = t_integrate
        entry.measurement_id = len(self.sequence)
        entry.threshold = threshold
        entry.pxi_trigger = pxi_trigger
        self._set_pulse_end(t_start + t_integrate * n_repeat)

    def close(self):
        # set wait time of last instruction
        if self.last_instruction is not None and self.end_pulse > self.time:
            t_wait = self.end_pulse - self.time
            t = (iround(t_wait) // 5 + 1) * 5
            self.last_instruction.time_after = t
            self.last_instruction = None

    def _set_pulse_end(self, t_end):
        self.end_pulse = max(self.end_pulse, iround(t_end))

    def _wait_till(self, t_start):
        t_instruction = (iround(t_start) // 10) * 10
        if t_instruction < self.end_pulse:
            raise Exception(f'Overlapping pulses at {t_start} of {self.name}')

        t_wait = t_instruction - self.time
        if self.last_instruction is not None and t_wait < 10:
            raise Exception(f'wait < 10 {self.name} at {t_start}')
        elif self.last_instruction is None and t_wait < 0:
            raise Exception(f'wait < 0 {self.name} at {t_start}')

        # set wait time of last instruction
        if self.last_instruction is not None:
            # Max wait time for instruction with waveform ~ 167 ms
            max_wait =  10*(2**24-1)
            t = min(max_wait, t_wait)
            self.last_instruction.time_after = t
            self.last_instruction = None
            t_wait -= t

        # insert additional instructions till t_start
        while t_wait > 0:
            entry = DigitizerSequenceEntry()
            self.sequence.append(entry)
            # Max wait time for instruction without waveform ~ 167 ms
            max_wait =  10*(2**24-1)
            t = min(max_wait, t_wait)
            entry.time_after = t
            t_wait -= t

        self.time = t_instruction
