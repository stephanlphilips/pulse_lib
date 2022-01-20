import logging
from numbers import Number
from copy import copy
import numpy as np
from .rendering import render_custom_pulse

class SequenceBuilderBase:
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
        self.set_next_marker()

    def set_next_marker(self):
        self.imarker += 1
        if len(self.markers) > self.imarker:
            self.t_next_marker = self.markers[self.imarker][0]
        else:
            self.t_next_marker = None

    def insert_markers(self, t):
        while self.t_next_marker is not None and t > self.t_next_marker:
            marker = self.markers[self.imarker]
            self._set_markers(marker[0], marker[1])
            self.set_next_marker()

    def _set_markers(self, t, value):
        self.seq.set_markers(value, t_offset=t)

    def _update_time(self, t, duration):
        if t < self.t_end:
            raise Exception(f'Overlapping pulses {t} > {self.t_end} ({self.name})')
        self.insert_markers(t)
        self.t_end = t + duration

    def add_comment(self, comment):
        self.seq.add_comment(comment)

    def close(self):
        while self.t_next_marker is not None:
            marker = self.markers[self.imarker]
            self._set_markers(marker[0], marker[1])
            self._update_time(self.t_next_marker, 0.0)
            self.set_next_marker()


class VoltageSequenceBuilder(SequenceBuilderBase):
    def __init__(self, name, sequencer, rc_time=None):
        super().__init__(name, sequencer)
        self.rc_time = rc_time
        self.integral = 0.0
        self.custom_pulses = []
        if rc_time is not None:
            self.compensation_factor = 1 / (1e9 * rc_time)
        else:
            self.compensation_factor = 0.0

    def ramp(self, t, duration, v_start, v_end):
        self._update_time(t, duration)
        v_start_comp = self._compensate_bias_T(v_start)
        v_end_comp = self._compensate_bias_T(v_end)
        self.integral += duration * (v_start + v_end)/2
        self.seq.ramp(duration, v_start_comp, v_end_comp, t_offset=t, v_after=None)

    def set_offset(self, t, duration, v):
        # sequencer only updates offset, no continuing action: duration=0
        self._update_time(t, 0)
        v_comp = self._compensate_bias_T(v)
        self.integral += duration * v
        if self.rc_time and duration > 0.01 * 1e9 * self.rc_time:
            self._update_time(t, duration)
            v_end = self._compensate_bias_T(v)
            self.seq.ramp(duration, v_comp, v_end, t_offset=t, v_after=None)
        else:
            self.seq.set_offset(v_comp, t_offset=t)

    def pulse(self, t, duration, amplitude, waveform):
        self._update_time(t, duration)
        # TODO @@@ add 2*np.pi*t*frequency*1e-9 to phase ??
        wave_id = self.register_sinewave(waveform)
        self.seq.shaped_pulse(wave_id, amplitude, t_offset=t)

    def custom_pulse(self, t, duration, amplitude, custom_pulse):
        self._update_time(t, duration)
        wave_id = self.register_custom_pulse(custom_pulse, amplitude)
        self.seq.shaped_pulse(wave_id, 1.0, t_offset=t)

    def register_custom_pulse(self, custom_pulse, scaling):
        data = render_custom_pulse(custom_pulse, scaling)
        for index,wave in enumerate(self.custom_pulses):
            if np.all(wave == data):
                return f'pulse{index}'
        index = len(self.custom_pulses)
        waveid = f'pulse{index}'
        self.custom_pulses.append(data)
        self.seq.add_wave(waveid, data)
        return waveid


    def _compensate_bias_T(self, v):
        return v + self.integral * self.compensation_factor


class IQSequenceBuilder(SequenceBuilderBase):
    def __init__(self, name, sequencer, nco_frequency):
        super().__init__(name, sequencer)
        self.nco_frequency = nco_frequency
        self.add_comment(f'IQ: NCO={self.nco_frequency/1e6:7.2f} MHz')

    def pulse(self, t, duration, amplitude, waveform):
        self._update_time(t, duration)
        self.add_comment(f'MW pulse {waveform.frequency/1e6:6.2f} MHz {waveform.duration} ns')
        waveform = copy(waveform)
        waveform.frequency -= self.nco_frequency

        if abs(waveform.frequency) > 1:
            # TODO @@@ Fix coherent pulses
            print(f'Warning: incorrect phase for pulse at {t} ns')
            wave_ids = self.register_sinewave_iq(waveform)
            self.seq.shaped_pulse(wave_ids[0], amplitude,
                                  wave_ids[1], amplitude,
                                  t_offset=t)
        elif not isinstance(waveform.phmod, Number):
            wave_ids = self.register_sinewave_iq(waveform)
            self.seq.shaped_pulse(wave_ids[0], amplitude,
                                  wave_ids[1], amplitude,
                                  t_offset=t)
        else:
            # frequency is less than 1 Hz make it 0.
            waveform.frequency = 0
            # phase is constant
            cycles = 2*np.pi*(waveform.phase + waveform.phmod)
            if isinstance(waveform.amod, Number):
                # TODO @@@ add option to use waveform for short pulses
                ampI = amplitude * waveform.amod * np.sin(cycles)
                ampQ = amplitude * waveform.amod * np.cos(cycles)
                # generate block pulse
                self.seq.block_pulse(duration, ampI, ampQ, t_offset=t)
            else:
                # phase is accounted for in ampI, ampQ
                waveform.phase = np.pi*0.5
                waveform.phmod = 0
                ampI = amplitude * np.sin(cycles)
                ampQ = amplitude * np.cos(cycles)
                # same wave for I and Q
                wave_id = self.register_sinewave(waveform)
                self.seq.shaped_pulse(wave_id, ampI, wave_id, ampQ, t_offset=t)

    def shift_phase(self, t, phase):
        self._update_time(t, 0.0)
        self.seq.shift_phase(phase, t_offset=t)


class AcquisitionSequenceBuilder(SequenceBuilderBase):
    def __init__(self, name, sequencer, n_repetitions):
        super().__init__(name, sequencer)
        self.n_repetitions = n_repetitions
        self.n_triggers = 0
        # allocate minim size later adjust for number of triggers
        self.seq.add_acquisition_bins('default', n_repetitions)

    @property
    def integration_time(self):
        return self.seq.integration_length_acq

    @integration_time.setter
    def integration_time(self, value):
        logging.info(f'{self.name}: integration time {value}')
        self.seq.integration_length_acq = value

    def acquire(self, t):
        self.n_triggers += 1
        self.seq.acquire('default', 'increment', t_offset=t)

    def repeated_acquire(self, t, n, t_period):
        self.n_triggers += n
        self.seq.repeated_acquire(n, t_period, 'default', 'increment', t_offset=t)

    def close(self):
        super().close()
        num_bins = self.n_triggers * self.n_repetitions
        self.seq.add_acquisition_bins('default', num_bins)



