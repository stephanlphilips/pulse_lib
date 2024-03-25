from dataclasses import dataclass, field
from typing import Union, Optional, List
import math
import numpy as np

def iround(value):
    return math.floor(value + 0.5)

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

    @property
    def extra_samples(self):
        # post_phase and pm_envelope add 2 samples, but last sample is restore of NCO
        # frequency and can be overwritten by next pulse without consequences.
        if self.postphase != 0:
            return 1
        try:
            # if pm_envelope is not an array, then this obviously fails.
            return 1 if self.pm_envelope[-1] != 0.0 else 0
        except:
            return 0

    @property
    def instruction_duration(self):
        return self.offset + self.duration + self.extra_samples

    def __eq__(self, other):
        return (self.amplitude == other.amplitude
                and np.all(self.am_envelope == other.am_envelope)
                and self.frequency == other.frequency
                and np.all(self.pm_envelope == other.pm_envelope)
                and self.prephase == other.prephase
                and self.postphase == other.postphase
                and self.duration == other.duration
                and self.offset == other.offset
                and self.restore_frequency == other.restore_frequency
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
class Instruction:
    t_start: int
    ''' start time of instruction. Multiple on 5 ns. '''
    waveform: Optional[Waveform] = None
    phase_shift: Optional[float] = None

    @property
    def t_pulse_end(self):
        ''' end time of instruction, exclusive. Multiple on 5 ns. '''
        if self.waveform:
            wvf = self.waveform
            return self.t_start + wvf.duration + wvf.offset
        else:
            return self.t_start

    @property
    def t_instruction_end(self):
        ''' end time of instruction, exclusive. Multiple on 5 ns. '''
        if self.waveform:
            wvf = self.waveform
            duration = wvf.duration + wvf.offset + wvf.extra_samples
            duration = (duration+4) // 5 * 5
        else:
            duration = 5
        return self.t_start + duration


@dataclass
class DigitizerSequenceEntry:
    time_after: float = 0
    t_measure: Optional[float] = None
    threshold: Optional[float] = None
    measurement_id: Optional[int] = None
    name: Optional[str] = None
    pxi_trigger: Optional[int] = None
    n_cycles : int = 1


class IQSequenceBuilder:
    def __init__(self, name, t_start, lo_freq, attenuation=1.0):
        self.name = name
        self.time = iround(t_start)
        self.end_pulse = self.time
        self.lo_freq = lo_freq
        self.attenuation = attenuation
        self.pending_instruction = None
        self.sequence = []
        self.waveforms = []

    def shift_phase(self, t, phase_shift):
        if abs(phase_shift) < 2*np.pi/2**18:
            # phase shift is too small for hardware
            return
        self._add_phase_shift(t, phase_shift)

    def pulse(self, t_pulse, iq_pulse):
        if (iq_pulse.envelope.AM_envelope_function is None
            and iq_pulse.envelope.PM_envelope_function is None
            and (iq_pulse.stop - iq_pulse.start > 200)):
            # split long pulses in start + stop pulse (Rabi)
            duration = iround(iq_pulse.stop - iq_pulse.start)
            start_wvf, stop_wvf = self._render_waveform_start_stop(iq_pulse)
            self._add_waveform(t_pulse, start_wvf)
            # align stop using duration. Make sure instruction offset is 0
            t_stop = iround(t_pulse) + duration
            duration_end = t_stop % 5
            stop_wvf.duration = duration_end
            t_stop_instruction = t_stop - duration_end
            self._add_waveform(t_stop_instruction, stop_wvf)
        else:
            waveform = self._render_waveform(iq_pulse)
            self._add_waveform(t_pulse, waveform)

    def chirp(self, t_pulse, chirp):
        # TODO: Frequency chirp with prescaler: pass as FM i.s.o. PM, or Chirp?
        waveform = self._render_chirp(chirp)
        self._add_waveform(t_pulse, waveform)

    def conditional_pulses(self, t_instr, segment_start, pulses, order,
                           condition_register):
        self._push_instruction()
        t_instr = iround(t_instr) // 5 * 5

        entry = SequenceConditionalEntry(cr=condition_register)

        t_end = t_instr
        wvf_indices = []
        for pulse in pulses:
            if pulse is None:
                # a do nothing pulse
                waveform = Waveform(duration=1)
                pulse_end = t_instr + 1
            elif pulse.mw_pulse is not None:
                mw_entry = pulse.mw_pulse
                t_pulse = segment_start + mw_entry.start
                wvf_offset = iround(t_pulse) - t_instr
                waveform = self._render_waveform(mw_entry, prephase=pulse.prephase, postphase=pulse.postphase)
                waveform.offset = wvf_offset
            else:
                waveform = Waveform(duration=2, prephase=pulse.prephase)

            index = self._get_waveform_index(waveform)
            wvf_indices.append(index)
            pulse_end = t_instr + waveform.instruction_duration
            t_end = max(t_end, pulse_end)

        for ibranch in order:
            entry.waveform_indices.append(wvf_indices[ibranch])

        self._append_sequence_entry(t_instr, entry)
        self._set_pulse_end(t_end)

    def close(self):
        self._push_instruction()
        # set wait time of last instruction
        last_entry = self._get_last_entry()
        if last_entry is not None and self.end_pulse > self.time:
            t_wait = self.end_pulse - self.time
            t = (iround(t_wait) // 5 + 1) * 5
            last_entry.time_after = t

    def _render_chirp(self, chirp):
        duration = iround(chirp.stop - chirp.start)
        frequency = chirp.start_frequency - self.lo_freq
        if abs(frequency) > 450e6:
            raise Exception(f'Chirp NCO frequency {frequency/1e6:5.1f} MHz is out of range')
        end_frequency = chirp.stop_frequency - self.lo_freq
        if abs(end_frequency) > 450e6:
            raise Exception(f'Chirp NCO frequency {end_frequency/1e6:5.1f} MHz is out of range')
        ph_gen = chirp.phase_mod_generator()
        return Waveform(
            chirp.amplitude / self.attenuation,
            1.0,
            frequency,
            ph_gen(duration, 1.0),
            0,
            0,
            duration)

    def _render_waveform(self, mw_pulse_data, prephase=0, postphase=0):
        # always render at 1e9 Sa/s
        duration = iround(mw_pulse_data.stop) - iround(mw_pulse_data.start)
        if mw_pulse_data.envelope is None:
            amp_envelope = 1.0
            pm_envelope = 0.0
        else:
            amp_envelope = mw_pulse_data.envelope.get_AM_envelope(duration, 1.0)
            pm_envelope = mw_pulse_data.envelope.get_PM_envelope(duration, 1.0)

        frequency = mw_pulse_data.frequency - self.lo_freq
        if abs(frequency) > 450e6:
            raise Exception(f'Waveform NCO frequency {frequency/1e6:5.1f} MHz is out of range')

        prephase += mw_pulse_data.phase_offset
        postphase -= mw_pulse_data.phase_offset
        return Waveform(
            mw_pulse_data.amplitude / self.attenuation,
            amp_envelope,
            frequency,
            pm_envelope,
            prephase,
            postphase,
            duration)

    def _render_waveform_start_stop(self, mw_pulse_data):
        frequency = mw_pulse_data.frequency - self.lo_freq
        if abs(frequency) > 450e6:
            raise Exception(f'Waveform NCO frequency {frequency/1e6:5.1f} MHz is out of range')

        prephase = mw_pulse_data.phase_offset
        postphase = -mw_pulse_data.phase_offset
        start_wvf = Waveform(
            mw_pulse_data.amplitude / self.attenuation,
            1.0,
            frequency,
            0.0,
            prephase=prephase,
            duration=2,
            restore_frequency=False)
        stop_wvf = Waveform(
            mw_pulse_data.amplitude / self.attenuation,
            1.0,
            frequency,
            0.0,
            postphase=postphase)
        return start_wvf, stop_wvf


    def _add_waveform(self, t, waveform):
        t = iround(t)
        if t < self.time:
            raise Exception(f'Oops! Pulses should be rendered in right order {t} < {self.time} ')
        offset = t % 5
        t_instruction = t - offset
        waveform.offset = offset

        if self.pending_instruction:
            pending = self.pending_instruction
            if t_instruction >= pending.t_instruction_end:
                self._push_instruction()
                self.pending_instruction = Instruction(t_instruction, waveform=waveform)
            elif pending.waveform:
                raise Exception(f'Overlapping MW pulses at {t_instruction}')
            elif pending.phase_shift is not None:
                waveform.prephase += pending.phase_shift
                pending.waveform = waveform
                pending.phase_shift = None
            else:
                raise Exception('Oops! instruction without waveform or pulse')
        else:
            # create new instruction with waveform
            self.pending_instruction = Instruction(t_instruction, waveform=waveform)

    def _add_phase_shift(self, t, phase_shift):
        t = iround(t)
        if t < self.time:
            raise Exception(f'Oops! Pulses should be rendered in right order {t} < {self.time} ')
        offset = t % 5
        t_instruction = t - offset

        if self.pending_instruction:
            pending = self.pending_instruction
            if t_instruction >= pending.t_instruction_end:
                self._push_instruction()
                self.pending_instruction = Instruction(t_instruction, phase_shift=phase_shift)
            elif pending.waveform:
                if t < pending.t_pulse_end:
                    raise Exception(f'Cannot shift phase during MW pulse at {t}')
                else:
                    pending.waveform.postphase += phase_shift
            elif pending.phase_shift is not None:
                pending.phase_shift += phase_shift
            else:
                raise Exception('Oops! instruction without waveform or pulse')
        else:
            # create new instruction with phase_shift
            self.pending_instruction = Instruction(t_instruction, phase_shift=phase_shift)

    def _push_instruction(self):
        pending = self.pending_instruction
        if not pending:
            return
        if pending.waveform:
            waveform = pending.waveform
        elif pending.phase_shift is not None:
            if pending.phase_shift != 0.0:
                waveform = Waveform(prephase=pending.phase_shift, duration=2)
            else:
                return
        else:
            raise Exception('Oops! instruction without waveform or pulse')
        entry = SequenceEntry()
        entry.waveform_index = self._get_waveform_index(waveform)
        self._append_sequence_entry(pending.t_start, entry)
        self._set_pulse_end(pending.t_instruction_end)
        self.pending_instruction = None

    def _set_pulse_end(self, t_end):
        self.end_pulse = max(self.end_pulse, iround(t_end))

    def _wait_till(self, t_start):
        t_instruction = (iround(t_start) // 5) * 5
        if t_instruction < self.end_pulse:
            raise Exception(f'Overlapping pulses at {t_start} of {self.name}')

        last_entry = self._get_last_entry()
        t_wait = t_instruction - self.time
        if t_wait < 0:
            raise Exception(f'wait < 0 {self.name} at {t_start}')
        if last_entry is not None and t_wait < 5:
            raise Exception(f'wait < 5 {self.name} at {t_start}')

        # set wait time of last instruction
        if last_entry is not None:
            # Max wait time for instruction with waveform ~ 327 us
            max_wait = 5 * (2**16-1)
            t = min(max_wait, t_wait)
            last_entry.time_after = t
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

    def _append_sequence_entry(self, t_start, entry):
        self._wait_till(t_start)
        self.sequence.append(entry)

    def _get_last_entry(self):
        if len(self.sequence):
            return self.sequence[-1]
        return None

    def _get_waveform_index(self, waveform:Waveform):
        try:
            index = self.waveforms.index(waveform)
        except ValueError:
            index = len(self.waveforms)
            self.waveforms.append(waveform)
        return index


class RFSequenceBuilder:
    def __init__(self, name, rf_source, offset):
        # note rf frequency can change. RF frequency should be set in play()
        self.name = name
        self._time = 0
        self.sequence = []
        self.waveforms = []
        self._amplitude = rf_source.amplitude / rf_source.attenuation
        self._start_offset = offset + rf_source.delay - rf_source.startup_time_ns
        self._stop_offset = offset + rf_source.delay - rf_source.prolongation_ns
        self._amplitude = rf_source.amplitude / rf_source.attenuation

    def enable(self, t_start, t_stop):
        if t_start < 0:
            raise Exception(f'start time cannot be negative ({t_start})')
        t_start = iround(t_start + self._start_offset) // 5 * 5
        t_stop = iround(t_stop + self._stop_offset + 4) // 5 * 5
        first_enable = len(self.sequence) == 0
        if first_enable:
            self._add_waveforms()
            if t_start > 0:
                self._add_entry(0, self._idle_index)
        if self._time >= t_start:
            # change time of last stop by increasing wait_after of last start
            last_instruction = self.sequence[-2]
            last_instruction.time_after += t_stop - self._time
            self._time = t_stop
        else:
            self._add_entry(t_start, self._active_index)
            self._add_entry(t_stop, self._idle_index)

    def _add_entry(self, t, index):
        if t > 0:
            time_after = t - self._time
            self.sequence[-1].time_after = time_after
        self.sequence.append(SequenceEntry(waveform_index=index))
        self._time = t

    def _add_waveforms(self):
        idle_wvf = Waveform(amplitude=0, duration=2)
        active_wvf = Waveform(amplitude=self._amplitude, duration=2)
        self.waveforms = [idle_wvf, active_wvf]
        self._idle_index = 0
        self._active_index = 1


class AcquisitionSequenceBuilder:
    def __init__(self, name):
        self.name = name
        self.time = 0
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
