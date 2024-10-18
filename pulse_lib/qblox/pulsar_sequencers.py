import logging
import math
from numbers import Number
from copy import copy
from typing import Callable
from dataclasses import dataclass
from contextlib import contextmanager

import numpy as np

from packaging.version import Version
from q1pulse import __version__ as q1pulse_version

if Version(q1pulse_version) < Version('0.9.0'):
    raise Exception('Upgrade q1pulse to version 0.9+')

from q1pulse.lang.conditions import CounterFlags


logger = logging.getLogger(__name__)


def iround(value):
    return math.floor(value + 0.5)


class PulsarConfig:
    ALIGNMENT = 4 # pulses must be aligned on 4 ns boundaries
    # 8000 samples memory, max 256 waves.
    EMIT_LENGTH1 = 100
    EMIT_LENGTH2 = 200
    EMIT_LENGTH3 = 300

    NS_SUB_DIVISION = 5

    @staticmethod
    def align(value):
        return math.floor(value / PulsarConfig.ALIGNMENT + 0.5) * PulsarConfig.ALIGNMENT

    @staticmethod
    def ceil(value):
        return math.ceil(value / PulsarConfig.ALIGNMENT - 1e-8) * PulsarConfig.ALIGNMENT

    @staticmethod
    def floor(value):
        return math.floor(value / PulsarConfig.ALIGNMENT + 1e-8) * PulsarConfig.ALIGNMENT

    @staticmethod
    def offset(value):
        return value % PulsarConfig.ALIGNMENT

    @staticmethod
    def hres_round(value):
        return math.floor(value * PulsarConfig.NS_SUB_DIVISION + 0.5) / PulsarConfig.NS_SUB_DIVISION


@dataclass
class LatchEvent:
    time: int
    reset: bool = False
    enable: bool = False


@dataclass
class MarkerEvent:
    time: int
    enabled_markers: int
    '''
    every bit represents a physical marker output.
    '''

class SequenceBuilderBase:
    verbose = False

    def __init__(self, name, sequencer):
        self.name = name
        self.seq = sequencer
        self.t_end = 0
        self.sinewaves = []
        self.t_next_marker = None
        self.imarker = 0
        self.max_output_voltage = sequencer.max_output_voltage

    def register_sinewave(self, waveform):
        try:
            index = self.sinewaves.index(waveform)
            waveid = f'sine{index}'
        except:
            index = len(self.sinewaves)
            self.sinewaves.append(waveform)
            waveid = f'sine{index}'
            data = waveform.render()
            self.seq.add_wave(waveid, data)
        return waveid

    def register_sinewave_iq(self, waveform):
        try:
            index = self.sinewaves.index(waveform)
            waveids = (f'iq{index}I', f'iq{index}Q')
        except:
            index = len(self.sinewaves)
            self.sinewaves.append(waveform)
            waveids = (f'iq{index}I', f'iq{index}Q')
            data = waveform.render_iq()
            self.seq.add_wave(waveids[0], data[0])
            self.seq.add_wave(waveids[1], data[1])
        return waveids

    def add_markers(self, markers):
        self.markers = markers
        self.imarker = -1
        self._set_next_marker()

    def _set_next_marker(self):
        self.imarker += 1
        if len(self.markers) > self.imarker:
            self.t_next_marker = self.markers[self.imarker].time
        else:
            self.t_next_marker = None

    def _insert_markers(self, t):
        while self.t_next_marker is not None and t >= self.t_next_marker:
            self._add_next_marker()

    def _add_next_marker(self):
        self.t_end = self.t_next_marker
        marker = self.markers[self.imarker]
        self._set_markers(marker.time, marker.enabled_markers)
        self._set_next_marker()

    def _set_markers(self, t, value):
        self.seq.set_markers(value, t_offset=t)

    # TODO @@@ improve error message. MW pulse (start, end), MW pulse (start, end)
    def _update_time(self, t, duration):
        if t < self.t_end:
            raise Exception(f'Overlapping pulses {t} < {self.t_end} ({self.name})')
        self.t_end = t + duration

    def _update_time_and_markers(self, t, duration):
        if t < self.t_end:
            raise Exception(f'Overlapping pulses {t} < {self.t_end} ({self.name})')
        self._insert_markers(t)
        self.t_end = t + duration

    def add_comment(self, comment):
        self.seq.add_comment(comment)

    def wait_till(self, t):
        self._update_time_and_markers(t, 0)
        self.seq.add_comment(f'wait {t}')
        self.seq.wait(t)

    def finalize(self):
        for i in range(self.imarker, len(self.markers)):
            marker = self.markers[i]
            self._set_markers(marker.time, marker.enabled_markers)
        if self.verbose:
            wavelengths = {name:len(wave.data) for name, wave in self.seq._waves._waves.items()}
            logger.info(f"waveforms: {wavelengths} ({sum(wavelengths.values())})")


# Range is -1.0 ... +1.0 with 16 bits => 2 / 2**16
_lsb_step = 3e-5


class VoltageSequenceBuilder(SequenceBuilderBase):
    '''
    Adds ramp, block, sine and custom pulses to q1pulse sequence.
    Markers are inserted in between pulses.
    Bias-T voltage compensation is added to block and ramp.
    '''
    def __init__(self, name, sequencer, rc_time=None):
        super().__init__(name, sequencer)
        self.rc_time = rc_time
        self.v_compensation = 0.0
        if rc_time is not None:
            self.compensation_factor = 1 / (1e9 * rc_time)
        else:
            self.compensation_factor = 0.0
        self._offset = None

        self._hres = False
        self._waveforms = []
        self._waveform = np.zeros(1000)

        self._rendering = False
        self._t_wave_start = 0
        self._t_wave_end = 0
        self._v_start = None
        self._v_end = None
        self._constant_end = False

    @property
    def hres(self):
        return self._hres

    @hres.setter
    def hres(self, value):
        self._hres = value

    def set_offset(self, t, duration, v):
        if duration == 0:
            # short-cut for optimization
            t = PulsarConfig.align(t)
            v += self.v_compensation
            self._update_time_and_markers(t, 0)
            self._set_offset(t, v)
        else:
            # due to bias-T compensation the offset will not be constant, but a ramp.
            self._ramp(t, t+duration, v, v)

    def ramp(self, t_start, t_end, v_start, v_end):
        # NOTE:
        # pulsar_uploader and data_pulse should make sure that custom_pulse and sin are added
        # before the ramp. No custom pulse or sin should start during a ramp.
        # There can be multiple ramps during 1 custom pulse. data_pulse sorts ramps and pulses.
        # No pulses will be added with a start time earlier than the previous one.

        if self.verbose:
            logger.info(f"ramp {t_start}, {t_end}, {v_start}, {v_end}")

        if self._hres:
            t_start = PulsarConfig.hres_round(t_start)
            t_end = PulsarConfig.hres_round(t_end)
        else:
            t_start = iround(t_start)
            t_end = iround(t_end)

        duration = t_end - t_start

        if duration == 0:
            if PulsarConfig.offset(t_start) == 0:
                # Used to reset voltage at end of segment.
                # This instruction will otherwise be turned into a new short waveform.
                # Also use it to flush waveform at end of segment
                if self._rendering and self._t_wave_end - self._t_wave_start >= 8:
                    self._emit_waveform(self._t_wave_end)
                if not self._rendering:
                    self.set_offset(t_start, 0, v_end)
            return

        if abs(v_start) < _lsb_step and abs(v_end) < _lsb_step:
            # nothing to render
            return

        is_ramp = abs(v_end - v_start) > _lsb_step
        is_long = (t_end - max(t_start, self._t_wave_end)) > (100 if is_ramp else 40)

        if is_long:
            line_start = PulsarConfig.ceil(max(t_start, self._t_wave_end))
            line_end = PulsarConfig.floor(t_end)
            dvdt = (v_end - v_start) / (t_end - t_start)
            if line_start - t_start > 0:
                self._emit_if_gap(t_start)
                t_end_wave = line_start
                v_end_wave = v_start + dvdt * (t_end_wave - t_start)
                self._render_ramp(t_start, t_end_wave, v_start, v_end_wave)
                self._emit_waveform(t_end_wave)
                t_start = t_end_wave
                v_start = v_end_wave
            elif self._rendering:
                self._emit_waveform(line_start)

            t_end_ramp = line_end
            if t_end_ramp != t_end:
                v_end_ramp = v_end + dvdt * (t_end_ramp - t_end)
                self._ramp(t_start, t_end_ramp, v_start, v_end_ramp)
                self._render_ramp(t_end_ramp, t_end, v_end_ramp, v_end)
            else:
                v_end_ramp = v_end
                self._ramp(t_start, t_end_ramp, v_start, v_end_ramp)
        else:
            self._emit_waveform_part(t_start)
            self._render_ramp(t_start, t_end, v_start, v_end)

    def add_sin(self, t_start, t_end, amplitude, frequency, phase):
        if self.verbose:
            logger.info(f"sin {t_start}, {t_end}")
        if self._hres:
            t_start = PulsarConfig.hres_round(t_start)
            t_end = PulsarConfig.hres_round(t_end)
            i_start = math.floor(t_start + 1e-5)
            i_end = math.ceil(t_end - 1e-5)
            t_offset = i_start - t_start
        else:
            t_start = iround(t_start)
            t_end = iround(t_end)
            i_start = t_start
            i_end = t_end
            t_offset = 0

        n_pt = i_end - i_start
        # if self._hres:
        #     n_pt += 1
        t = t_offset + np.arange(n_pt)
        w = 2*np.pi*frequency*1e-9
        sine_data = amplitude * np.sin(w*t + phase)
        if self._hres:
            frac_start = i_start + 1 - t_start
            frac_end = 1 - i_end + t_end
            sine_data[0] = frac_start * sine_data[0]
            sine_data[-1] = frac_end * sine_data[-1]

            # sine_data[0] = frac_start * amplitude * np.sin(phase)
            # if frac_end < 0.999:
            #     sine_data[-2] = frac_end * sine_data[-2]
            #     sine_data[-1] = frac_end * (amplitude*np.sin(w*iround(t_end-t_start-1)+phase) - sine_data[-2])
            # else:
            #     pass

        self._emit_waveform_part(t_start)
        self._add_waveform_data(i_start, sine_data)

    def custom_pulse(self, t_start, t_end, amplitude, custom_pulse):
        t_start = iround(t_start)
        data = custom_pulse.render(sample_rate=1e9) * amplitude
        self._emit_waveform_part(t_start)
        self._add_waveform_data(t_start, data)

    def wait_till(self, t):
        if self.verbose:
            logger.debug(f"wait till {t}, wave end: {self._t_wave_end}")
        if self._rendering:
            self._emit_waveform(self._t_wave_end)
        super().wait_till(t)

    def _add_integral(self, integral):
        self.v_compensation += integral * self.compensation_factor

    def _set_offset(self, t, v):
        # Note: only add instruction when it is a significant step
        # offset is changed by ramp and set_offset calls.
        if self._offset is None or abs(self._offset - v) > _lsb_step:
            self.seq.set_offset(v, t_offset=t)
            self._offset = v

    def _ramp(self, t_start, t_end, v_start, v_end):
        t_start = PulsarConfig.align(t_start)
        t_end = PulsarConfig.align(t_end)

        duration = t_end - t_start

        v_start_comp = v_start + self.v_compensation
        self._add_integral(duration * (v_start + v_end)/2)
        v_end_comp = v_end + self.v_compensation

        # ramp only when > 2 bits on 16-bit signed resolution
        is_ramp = abs(v_start_comp - v_end_comp) > 2*_lsb_step
        if is_ramp:
            self._update_time_and_markers(t_start, duration)
            self.seq.ramp(duration, v_start_comp, v_end_comp, t_offset=t_start, v_after=None)
            self._offset = None
        else:
            # sequencer only updates offset, no continuing action: duration=0
            self._update_time_and_markers(t_start, 0)
            self._set_offset(t_start, v_start_comp)

    def _render_ramp(self, t_start, t_end, v_start, v_end):
        if self._hres:
            istart = math.floor(t_start + 1e-8)
            iend = math.ceil(t_end - 1e-8)
            n = iend - istart
            if n == 0:
                return
            dt_start = t_start - istart
            dt_end = iend - t_end
            dvdt = (v_end - v_start) / (t_end - t_start)
            frac_start = 1 - dt_start
            frac_end = 1 - dt_end
            data = np.linspace(v_start-dt_start*dvdt, v_end+dt_end*dvdt, n, endpoint=False)
            data[0] = frac_start*(v_start)
            data[-1] = frac_end*(v_end-dvdt)
            self._add_waveform_data(istart, data, v_start)
        else:
            n = t_end - t_start
            if n == 0:
                return
            data  = np.linspace(v_start, v_end, n, endpoint=False)
            self._add_waveform_data(t_start, data, v_start)

        if self._v_start is not None:
            # Note self_v_start is not None if the waveform does not start with sin or custom.
            self._equal_voltage = abs(self._v_start - v_end) < _lsb_step
            if abs(v_end - v_start) > 2*_lsb_step:
                t_constant = 0
            else:
                t_constant = t_end - t_start
            self._t_constant = t_constant

    def _add_waveform_data(self, t_start, data, v_start=None):
        n = len(data)
        if n == 0:
            return
        if self.verbose:
            logger.debug(f"add data {t_start} {len(data)}")
        if t_start < self._t_wave_start:
            raise Exception(f"Error in rendering cannot start in past {t_start} < {self._t_wave_start}")
        t_end = t_start + n
        if not self._rendering:
            self._rendering = True
            self._t_wave_start = PulsarConfig.floor(t_start)
            self._t_wave_end = t_end
            self._v_start = v_start
            self._waveform = np.zeros(8000)
        else:
            self._t_wave_end = max(self._t_wave_end, t_end)

        istart = t_start - self._t_wave_start
        if istart + n > 8000:
            raise Exception(f'Rendered waveform too big for Qblox module ({istart+n} > 8000)')

        waveform = self._waveform
        waveform[istart:istart+n] += data

        # Note: this will be overwritten in _render_ramp.
        iend = self._t_wave_end - self._t_wave_start - 1
        self._equal_voltage = abs(waveform[istart] - waveform[iend]) < _lsb_step
        self._t_constant = 0

    def _emit_if_gap(self, t_start):
        if self._rendering and t_start - self._t_wave_end >= 40:
            # there is a gap
            self._emit_waveform(PulsarConfig.ceil(self._t_wave_end))

    def _emit_waveform_part(self, t_start):
        if not self._rendering:
            return
        t_start = int(t_start)
        t_wave_end = self._t_wave_end
        t_gap = t_start - t_wave_end
        if t_gap >= 40:
            # there is a significant gap
            self._emit_waveform(PulsarConfig.ceil(t_wave_end))
            return
        t_start = PulsarConfig.floor(t_start)
        if t_gap >= 0:
            t_constant = max(self._t_constant, t_gap)
            aligned = t_wave_end % 4 == 0
            length = t_wave_end - self._t_wave_start
            if length > PulsarConfig.EMIT_LENGTH1 and self._equal_voltage and (
                    t_constant >= 4 or aligned):
                # equal voltages: could be repeatable waveform
                self._emit_waveform(t_start)
                return
            if length > PulsarConfig.EMIT_LENGTH2 and (
                    t_constant >= 4 or aligned):
                # it's aligned: could be repeatable waveform
                self._emit_waveform(t_start)
                return

        length = t_start - self._t_wave_start
        if length > PulsarConfig.EMIT_LENGTH3:
            # it's getting long...
            self._emit_waveform(t_start)

    def _emit_waveform(self, t_end):
        # TODO optimize: Chop off constant part and use set_offset
        #      This reduces memory usage and could result in more reuse of waveforms in certain cases.

        # TODO Chop off constant part (at end) with v = 0.0
        #      This reduces memory usage.
        waveform = self._waveform
        n = t_end - self._t_wave_start
        if self.verbose:
            logger.info(f"EMIT waveform {t_end}: {n}")
        if n <= 0:
            return
        self._play_waveform(self._t_wave_start, waveform[:n].copy())
        self._t_wave_start = PulsarConfig.ceil(t_end)

        if t_end < self._t_wave_end:
            # copy remainder
            remainder = self._t_wave_end - t_end
            new = np.zeros(8000)
            new[:remainder] = waveform[n:n+remainder]
            self._waveform = new
            self._v_start = None
        else:
            self._rendering = False

    def _play_waveform(self, t_start, waveform):
        # TODO Optimize: check if offset has changed due to ramp or set_offset

        self._update_time_and_markers(t_start, 0)
        self._set_offset(t_start, self.v_compensation)
        self._add_integral(np.sum(waveform))

        for index, data in enumerate(self._waveforms):
            if len(data) == len(waveform) and np.allclose(data, waveform):
                waveid = f'wave_{index}'
                break
        else:
            index = len(self._waveforms)
            waveid = f'wave_{index}'
            self._waveforms.append(waveform)
            self.seq.add_wave(waveid, waveform)
        self.seq.shaped_pulse(waveid, 1.0, t_offset=t_start)


class IQSequenceBuilder(SequenceBuilderBase):
    def __init__(self, name, sequencer, nco_frequency,
                 mixer_gain=None, mixer_phase_offset=None):
        super().__init__(name, sequencer)
        self.nco_frequency = nco_frequency
        self.add_comment(f'IQ: NCO={nco_frequency/1e6:7.2f} MHz')
        self._square_waves = []
        self._trigger_counters = {}
        self._uses_feedback = False
        self._in_conditional = False
        self._t_next_latch_event = None
        self._ilatch_event = 0
        self._latch_events = []

        if mixer_gain is not None:
            self.seq.mixer_gain_ratio = mixer_gain[1]/mixer_gain[0]
        if mixer_phase_offset is not None:
            self.seq.mixer_phase_offset_degree = mixer_phase_offset/np.pi*180

    def pulse(self, t_start, t_end, amplitude, waveform):
        self._check_set_nco_freq()
        # NOTE:
        # Pulses are aligned on 1 ns. This affects the duration of the pulse.
        # A time shift should not affect the phase, because the phase is tracked by the NCO.
        t_start = iround(t_start)
        t_end = iround(t_end)
        if t_end - t_start == 0:
            # Nothing to render
            return
        t_pulse = PulsarConfig.floor(t_start)
        self._update_time_and_markers(t_pulse, t_end-t_pulse)
        self.add_comment(f'MW pulse {waveform.frequency/1e6:6.2f} MHz {waveform.duration} ns')
        waveform = copy(waveform)
        waveform.frequency -= self.nco_frequency
        waveform.offset = t_start-t_pulse

        # Always align on outer boundaries and add offset.
        # NOTE: There must be some padding between pulses if they are not aligned.
        #       Anyways, it is always better to have some padding between pulses.

        if abs(waveform.frequency) > 1:
            if abs(waveform.frequency) > 400e6:
                raise Exception(f'Waveform frequency {waveform.frequency/1e6:5.1f} MHz out of range')
            w = 2 * np.pi * waveform.frequency * 1e-9
            if isinstance(waveform.phmod, Number) and isinstance(waveform.amod, Number):
                self.seq.set_frequency(waveform.frequency + self.nco_frequency, t_offset=t_pulse)
                phase = waveform.phase + waveform.phmod - (t_start-t_pulse) * w
                self._add_boxcar_pulse(amplitude * waveform.amod, phase, t_pulse, t_start, t_end)
                t_pulse_end = PulsarConfig.ceil(t_end)
                self.seq.set_frequency(self.nco_frequency, t_offset=t_pulse_end)
                # Adjust NCO after running with other frequency for longer time than the pulse.
                delta_phase = -w*(PulsarConfig.ceil(t_end)-t_end + t_start-t_pulse)
                self.shift_phase(PulsarConfig.ceil(t_end), delta_phase)
            else:
                wave_ids = self.register_sinewave_iq(waveform)
                self.seq.shaped_pulse(wave_ids[0], amplitude,
                                      wave_ids[1], amplitude,
                                      t_offset=t_pulse)
                # Adjust NCO after driving with other frequency
                delta_phase = w*waveform.duration
                self.shift_phase(PulsarConfig.ceil(t_end), delta_phase)

        elif not isinstance(waveform.phmod, Number):
            wave_ids = self.register_sinewave_iq(waveform)
            self.seq.shaped_pulse(wave_ids[0], amplitude,
                                  wave_ids[1], amplitude,
                                  t_offset=t_pulse)
        else:
            # frequency is less than 1 Hz make it 0.
            waveform.frequency = 0
            # phase is constant
            phase = waveform.phase + waveform.phmod
            if isinstance(waveform.amod, Number):
                self._add_boxcar_pulse(amplitude * waveform.amod, phase, t_pulse, t_start, t_end)
            else:
                # phase is accounted for in ampI, ampQ
                waveform.phase = np.pi/2 # pi/2, because waveform.render uses sin instead of cos.
                waveform.phmod = 0
                ampI = amplitude * np.cos(phase)
                ampQ = amplitude * np.sin(phase)
                # same wave for I and Q
                wave_id = self.register_sinewave(waveform)
                self.seq.shaped_pulse(wave_id, ampI, wave_id, ampQ, t_offset=t_pulse)

    def _add_boxcar_pulse(self, amplitude, phase, t_pulse, t_start, t_end):
        ampI = amplitude * np.cos(phase)
        ampQ = amplitude * np.sin(phase)
        # generate block pulse
        t_start_offset = t_start % 4
        t_end_offset = t_end % 4
        duration = t_end-t_start
        if duration < 200:
            # Create square waveform with offset
            wave_id = self._register_squarewave(t_start_offset, duration)
            self.seq.shaped_pulse(wave_id, ampI, wave_id, ampQ, t_offset=t_pulse)
        elif t_start_offset == 0 and t_end_offset == 0:
            # Create aligned block pulse with offset
            self.seq.block_pulse(duration, ampI, ampQ, t_offset=t_pulse)
        else:
            # Create square waveform for start and end, and use offset in between.
            if t_start_offset:
                wave_id_start = self._register_squarewave(t_start_offset, 4-t_start_offset)
                self.seq.shaped_pulse(wave_id_start, ampI, wave_id_start, ampQ, t_offset=t_pulse)
            block_duration = PulsarConfig.floor(t_end) - PulsarConfig.ceil(t_start)
            self.seq.block_pulse(block_duration, ampI, ampQ,
                                 t_offset=PulsarConfig.ceil(t_start))
            if t_end_offset:
                wave_id_end = self._register_squarewave(0, t_end_offset)
                self.seq.shaped_pulse(wave_id_end, ampI, wave_id_end, ampQ, t_offset=t_end-t_end_offset)

    def _register_squarewave(self, offset, duration):
        wave_id = f'square_{offset}_{duration}'
        if wave_id not in self._square_waves:
            data = np.zeros(offset+duration)
            data[offset:] = 1.0
            self._square_waves.append(wave_id)
            self.seq.add_wave(wave_id, data)
        return wave_id

    def shift_phase(self, t, phase):
        '''
        Arguments:
            phase (float): phase in rad.
        '''
        # The phase shift can be before or after MW pulse.
        # First try to align towards lower t
        t_phase = PulsarConfig.floor(t)
        if t_phase < self.t_end:
            # Align towards higher t.
            t_phase = PulsarConfig.ceil(t)
        self._update_time_and_markers(t_phase, 0.0)
        # normalize phase to -1.0 .. + 1.0 for Q1Pulse sequencer
        norm_phase = (phase/np.pi + 1) % 2 - 1
        if -1e-6 < norm_phase < 1e-6:
            # ignore very small phase shifts
            return
        self.seq.shift_phase(norm_phase, t_offset=t_phase)

    def chirp(self, t_start, t_end, amplitude, start_frequency, stop_frequency):
        # set NCO frequency if valid. Otherwise set 0.0 to enable modulation
        if abs(start_frequency) > 450e6 or abs(stop_frequency) > 450e6:
            raise Exception(f"{self.name}: Chirp frequencies {start_frequency/1e6:5.1f}, "
                            f"{stop_frequency/1e6:5.1f} MHz out of range")
        if start_frequency * stop_frequency <= 0.0:
            raise Exception(f"Qblox does not support chirp through 0.0 Hz. "
                            f"Chirp({start_frequency/1e6:5.1f}, {stop_frequency/1e6:5.1f} MHz)")
        if self._has_valid_nco_freq():
            self.seq.nco_frequency = self.nco_frequency
        else:
            self.seq.nco_frequency = 0.0
        t_start = PulsarConfig.align(t_start)
        t_end = PulsarConfig.align(t_end)
        self._update_time_and_markers(t_start, 0.0)
        self.seq.chirp(t_end-t_start, amplitude,
                       start_frequency, stop_frequency,
                       t_offset=t_start)
        # restore NCO frequency if it is valid.
        if self._has_valid_nco_freq():
            self.seq.set_frequency(self.nco_frequency, t_offset=t_end)

    def _has_valid_nco_freq(self):
        return self.nco_frequency is not None and abs(self.nco_frequency) <= 450e6

    def _check_set_nco_freq(self):
        if self.nco_frequency is None:
            raise Exception(f'{self.name}: Qubit resonance frequency has not been set')
        if abs(self.nco_frequency) > 450e6:
            raise Exception(f'{self.name}: NCO frequency {self.nco_frequency/1e6:5.1f} MHz out of range')
        self.seq.nco_frequency = self.nco_frequency

    def add_trigger_counter(self, trigger):
        self._uses_feedback = True
        self._trigger_counters[trigger.sequencer_name] = self.seq.add_trigger_counter(trigger)

    @contextmanager
    def conditional(self, channels, t_min, t_max):
        t = max(self.t_end+4, t_min, t_max-40)
        t = PulsarConfig.ceil(t)
        if t > t_max-8:
            raise Exception(f'Failed to schedule conditional pulse on {self.name}. '
                            f' t:{t} > {t_max-8}')
        # add markers before conditional
        self._update_time_and_markers(t, 0)
        self._in_conditional = True
        counters = [self._trigger_counters[ch] for ch in channels]
        flags = CounterFlags(self)
        with self.seq.conditional(counters, t_offset=t):
            yield flags
        self._in_conditional = False

    @contextmanager
    def condition(self, operator):
        t_condition_start = self.t_end
        self.seq.enter_condition(operator)
        yield
        self.seq.exit_condition(PulsarConfig.ceil(self.t_end))
        # reset end time
        self.t_end = t_condition_start

    def _insert_markers(self, t):
        '''
        Override of insert_marker taking care of conditional blocks and
        counter latching.
        '''
        if self._uses_feedback:
            if self._in_conditional:
                if self.t_next_marker is not None and t >= self.t_next_marker:
                    raise Exception(f'Cannot set marker in conditional segment {self.name}, t:{t}')
                if self._t_next_latch_event is not None and t >= self._t_next_latch_event:
                    raise Exception(f'Cannot enable latches in conditional segment {self.name}, t:{t}, '
                                    f'{self._latch_events[self._ilatch_event]}')
                return
            else:
                # add latch events, but not on same time as marker
                loop = True
                while loop:
                    loop = False
                    if (self._t_next_latch_event is not None
                        and self._t_next_latch_event < t
                        and (self.t_next_marker is None
                             or self._t_next_latch_event < self.t_next_marker)):
                        self._add_next_latch_event()
                        loop = True
                    elif self.t_next_marker is not None and t >= self.t_next_marker:
                        self._add_next_marker()
                        loop = True
        else:
            super()._insert_markers(t)

    def add_latch_events(self, latch_events, t_offset):
        self._latch_events = []
        for event in latch_events:
            event = copy(event)
            event.time = PulsarConfig.floor(event.time + t_offset)
            self._latch_events.append(event)
        self._ilatch_event = -1
        self._set_next_latch_event()

    def _set_next_latch_event(self):
        self._ilatch_event += 1
        if len(self._latch_events) > self._ilatch_event:
            self._t_next_latch_event = self._latch_events[self._ilatch_event].time
        else:
            self._t_next_latch_event = None

    def _add_next_latch_event(self):
        latch_event = self._latch_events[self._ilatch_event]
        if latch_event.time + 20 < self.t_end:
            raise Exception(f'Latch event {latch_event} on {self.name} scheduled too late t:{self.t_end}')
        # Increment time. There could already be a phase shift, awg offset or marker on t_end
        self.t_end += 4
        t = PulsarConfig.ceil(max(self.t_end, latch_event.time))
        logger.info(f'{latch_event} at t={self.t_end}')
        if latch_event.reset:
            self.seq.latch_reset(t_offset=t)
        else:
            if Version(q1pulse_version) < Version('0.11.5'):
                raise Exception('Upgrade q1pulse to version 0.11.5+ for correct feedback implementation')
            self.seq.latch_enable(latch_event.enable, t_offset=t)
        # Increment time with time used for latch instruction
        self.t_end = t+4
        self._set_next_latch_event()

    def finalize(self):
        if self._t_next_latch_event is not None:
            for latch_event in self._latch_events[self._ilatch_event:]:
                if latch_event.reset:
                    logger.info(f'latch reset at end {latch_event} ({self.t_end})')
                    if latch_event.time == np.inf:
                        t = PulsarConfig.ceil(self.t_end+4)
                    else:
                        t = latch_event.time
                    self.seq.latch_reset(t_offset=t)
                    self.t_end = max(self.t_end, t+4)
                else:
                    logger.info(f'Skipping latch event at end: {latch_event}')

        super().finalize()


@dataclass
class _SeqCommand:
    time: int
    func: Callable[..., any]
    args: list[any]
    kwargs: dict[str, any]


class AcquisitionSequenceBuilder(SequenceBuilderBase):
    def __init__(self, name, sequencer, n_repetitions, nco_frequency=None, rf_source=None):
        super().__init__(name, sequencer)
        self.n_repetitions = n_repetitions
        self.seq.nco_frequency = nco_frequency
        self.rf_source = rf_source
        self.n_triggers = 0
        self._integration_time = None
        self._data_scaling = None
        self._commands = []
        self._weights = []
        self.rf_source_mode = rf_source.mode if rf_source is not None else None
        self._pulse_end = -1
        self.offset_rf_ns = 0
        self._nco_prop_delay = 0
        self._nco_prop_delay_en = False
        if rf_source is not None:
            if isinstance(rf_source.output, str):
                raise Exception('Qblox RF source must be configured using module name and channel numbers')
            scaling = 1/(rf_source.attenuation * self.max_output_voltage*1000) # @@@@ Multiply with sqrt(2)
            self._rf_amplitude = rf_source.amplitude * scaling
            self._n_out_ch = 1 if isinstance(rf_source.output[1], int) else 2

    @property
    def integration_time(self):
        return self._integration_time

    @integration_time.setter
    def integration_time(self, value):
        if value is None:
            raise ValueError('Integration time cannot be None')
        if value > 2**24-4:
            raise ValueError(f'Integration time of {value} ns is too large. Max value: {2**24-4} ns')
        if self._integration_time is None:
            logger.info(f'{self.name}: integration time {value}')
            self._integration_time = value
            self.seq.integration_length_acq = value
        elif self._integration_time != value:
            raise Exception('Only 1 integration time (t_measure) per channel is supported')

    @property
    def nco_prop_delay(self):
        return self._nco_prop_delay

    @nco_prop_delay.setter
    def nco_prop_delay(self, delay):
        self._nco_prop_delay = delay

    def add_markers(self, markers):
        for marker in markers:
            t,value = marker
            # offset command sorting time to be before acquire
            self._add_command(t - 0.01,
                              self.seq.set_markers, value, t_offset=t)

    def acquire(self, t, t_integrate):
        if self.integration_time is not None and self.integration_time != t_integrate:
            # use weighed acquisition
            self.acquire_weighed(t, np.ones(int(t_integrate)))
        else:
            self._update_time(t, t_integrate)
            self.integration_time = t_integrate
            self.n_triggers += 1
            self._add_scaling(1/t_integrate, 1)
            if self.rf_source_mode in ['pulsed', 'shaped']:
                self._add_pulse(t, t_integrate)
            # enqueue: self.seq.acquire('default', 'increment', t_offset=t)
            self._add_command(t,
                              self.seq.acquire, 'default', 'increment', t_offset=t)

    def acquire_weighed(self, t, weight):
        t_integrate = len(weight)
        self._update_time(t, t_integrate)
        self.n_triggers += 1
        self._add_scaling(1/np.sum(np.abs(weight)), 1)
        if self.rf_source_mode in ['pulsed', 'shaped']:
            self._add_pulse(t, t_integrate)
        weight_id = self._add_weight(weight)
        # enqueue: self.seq.acquire_weighed('default', 'increment', weight_id, t_offset=t)
        self._add_command(t,
                          self.seq.acquire_weighed, 'default', 'increment', weight_id, t_offset=t)

    def repeated_acquire(self, t, t_integrate, n, t_period, f_sweep):
        duration = (n-1) * t_period + t_integrate
        self._update_time(t, duration)
        self.integration_time = t_integrate
        self.n_triggers += n
        self._add_scaling(1/t_integrate, n)
        if self.rf_source_mode in ['pulsed', 'shaped']:
            self._add_pulse(t, duration)
        if f_sweep is None:
            # enqueue: self.seq.repeated_acquire(n, t_period, 'default', 'increment', t_offset=t)
            self._add_command(t,
                              self.seq.repeated_acquire, n, t_period, 'default', 'increment', t_offset=t)
        else:
            # enable nco propagation delay for frequency changing
            self._nco_prop_delay_en = True
            # enqueue: self.seq.repeated_acquire(n, t_period, 'default', 'increment', t_offset=t)
            self._add_command(t,
                              self.seq.acquire_frequency_sweep, n, t_period, f_sweep[0], f_sweep[1],
                              'default', 'increment',
                              acq_delay=PulsarConfig.align(self._nco_prop_delay + 4),
                              t_offset=t)

    def reset_bin_counter(self, t):
        # enqueue: self.seq.reset_bin_counter('default')
        self._add_command(t,
                          self.seq.reset_bin_counter, 'default')

    def finalize(self):
        # note: no need to call super().finalize() because all markers have already been added.
        if not self.n_triggers:
            return
        if not self._integration_time:
            raise Exception(f'Measurement time not set for channel {self.name}')
        num_bins = self.n_triggers * self.n_repetitions
        self.seq.add_acquisition_bins('default', num_bins)

        if self.rf_source_mode == 'continuous':
            self._add_pulse(0, self.t_end)
        self._add_pulse_end()

        self._commands.sort(key=lambda cmd:cmd.time)
        for cmd in self._commands:
            cmd.func(*cmd.args, **cmd.kwargs)
        if self._nco_prop_delay_en:
            self.seq.nco_prop_delay = self.nco_prop_delay
        else:
            self.seq.nco_prop_delay = 0

    def get_data_scaling(self):
        if isinstance(self._data_scaling, (Number, type(None))):
            return self._data_scaling
        return np.array(self._data_scaling)

    def _add_scaling(self, scaling, n):
        if self._data_scaling is None:
            self._data_scaling = scaling
        elif isinstance(self._data_scaling, Number):
            if self._data_scaling == scaling: # @@@ rounding errors...
                pass
            else:
               self._data_scaling = [self._data_scaling] * self.n_triggers
               self._data_scaling[-n:] = [scaling] * n
        else:
           self._data_scaling += [scaling] * n

    def _add_command(self, t, func, *args, **kwargs):
        self._commands.append(_SeqCommand(t, func, args, kwargs))

    def _add_pulse(self, t, duration):
        t_start = t + self.offset_rf_ns
        t_end = t + duration
        t_start -= self.rf_source.startup_time_ns
        t_end += self.rf_source.prolongation_ns

        if t_start < 0:
            raise Exception('RF source has negative start time. Acquisition triggered too early. '
                            f'Acquisition start: {t}, RF source start: {t_start}')
        t_start = PulsarConfig.align(t_start)
        if t_start > self._pulse_end:
            self._add_pulse_end()
            amp0 = self._rf_amplitude
            # amplitude 1 should be 0.0. It's the Q-component used in IQ modulation.
            # only the I-component is used to set the amplitude.
            amp1 = 0.0 if self._n_out_ch == 2 else None
            # offset command sorting time to be before acquire
            self._add_command(t_start - 0.01,
                              self.seq.set_offset, amp0, amp1, t_offset=t_start)
        self._pulse_end = t_end

    def _add_pulse_end(self):
        if self._pulse_end > 0:
            t = self._pulse_end
            t = PulsarConfig.align(t)
            amp1 = 0.0 if self._n_out_ch == 2 else None
            # offset command sorting time to be before acquire
            self._add_command(t - 0.01,
                              self.seq.set_offset, 0.0, amp1, t_offset=t)
            self._pulse_end = -1

    def _add_weight(self, weight):
        for index, data in enumerate(self._weights):
            if len(data) == len(weight) and np.allclose(data, weight):
                weight_id = f'weight_{index}'
                break
        else:
            index = len(self._weights)
            weight_id = f'weight_{index}'
            self._weights.append(weight)
            self.seq.add_weight(weight_id, weight)
        return weight_id
