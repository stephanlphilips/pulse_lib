"""
data class to make pulses.
"""
import logging
import numpy as np
import copy
from dataclasses import dataclass
from numbers import Number
from typing import Any, Dict, Callable, List

from pulse_lib.segments.utility.rounding import iround
from pulse_lib.segments.data_classes.data_generic import parent_data
from pulse_lib.segments.data_classes.data_IQ import envelope_generator

total_pulse_deltas = 0

def get_total_deltas():
    return total_pulse_deltas

@dataclass
class pulse_delta:
    time: float
    step: float = 0.0
    ramp: float = 0.0

    def __post_init__(self):
        global total_pulse_deltas
        total_pulse_deltas += 1

    def __add__(self, other):
        if isinstance(other, Number):
            return pulse_delta(self.time,
                               self.step + other,
                               self.ramp)
        elif isinstance(other, pulse_delta):
            return pulse_delta(self.time,
                               self.step + other.step,
                               self.ramp + other.ramp)
        else:
            raise Exception(f'Cannot add pulse_data to {type(other)}')

    def __iadd__(self, other):
        if isinstance(other, Number):
           self.step += other
        elif isinstance(other, pulse_delta):
           self.step += other.step
           self.ramp += other.ramp
        else:
            raise Exception(f'Cannot add pulse_data to {type(other)}')
        return self

    def __mul__(self, other):
        if isinstance(other, Number):
            return pulse_delta(self.time,
                               self.step * other,
                               self.ramp * other)
        else:
            raise Exception(f'Cannot multiply pulse_data with {type(other)}')

    def __imul__(self, other):
        if isinstance(other, Number):
           self.step *= other
           self.ramp *= other
        else:
            raise Exception(f'Cannot multiply pulse_data with {type(other)}')
        return self

    @property
    def is_near_zero(self):
        # near zero if |step| < 0.001 uV and |ramp| < 1e-9 mV/ns (= 1 mV/s)
        # note: max ramp: 2V/ns = 2000 mV/ns, min ramp: 1 mV/s = 1e-9 mV/ns. ~ 12 orders of magnitude.
        # Regular floats have 16 digits precision.
        return -1e-6 < self.step < 1e-6 and -1e-9 < self.ramp < 1e-9

@dataclass
class custom_pulse_element:
    start: float
    stop: float
    amplitude: float
    func: Callable[..., np.ndarray]
    kwargs: Dict[str,Any]
    scaling: int = 1.0

    def render(self, sample_rate):
        duration = self.stop - self.start
        data = self.func(duration, sample_rate, self.amplitude, **self.kwargs)
        return data*self.scaling

@dataclass
class rendered_element:
    start: int
    stop: int
    wvf: np.ndarray = None


def shift_start_stop(data:List[Any], delta) -> None:
    for element in data:
        element.start += delta
        element.stop += delta

def get_max_time(data:List[Any]) -> float:
    stop = 0
    for element in data:
        stop = max(stop, element.stop)
    return stop

def shift_time(data:List[Any], delta) -> None:
    for element in data:
        element.time += delta


@dataclass
class PhaseShift:
    time: float
    phase_shift: float
    channel_name: str

    @property
    def start(self):
        return self.time

    @property
    def stop(self):
        return self.time

    def __add__(self, other):
        if isinstance(other, PhaseShift):
            if other.channel_name != self.channel_name:
                raise Exception(f'Segment corruption: {other.channel_name} != {self.channel_name}')
            return PhaseShift(
                    self.time,
                    self.phase_shift + other.phase_shift,
                    self.channel_name)
        else:
            raise Exception(f'Cannot add PhaseShift to {type(other)}')

    @property
    def is_near_zero(self):
        # near zero if |shift| < 2*pi/2**31
        eps = 2*np.pi/2**32
        return -eps < self.phase_shift < eps

# Changed [v1.6.0] time,duration -> start,stop
@dataclass
class OffsetRamp:
    start: float
    stop: float # time of next OffsetRamp
    v_start: float
    v_stop: float

# keep till end: start = np.inf
# slicing:
    # consolidate all in slice on `end`, keep `inf`. delta_new = sum(p.delta for p in slice), ...
# rendering:
    # cumsum ramp and step.
    # if abs(ramp) < 1e-9: ramp=0


class pulse_data(parent_data):
    """
    class defining base (utility) operations for baseband and microwave pulses.
    """
    def __init__(self):
        super().__init__()
        self.pulse_deltas = list()
        self.MW_pulse_data = list()
        self.custom_pulse_data = list()
        self.phase_shifts = list()

        self.start_time = 0
        self._end_time = 0
        self._consolidated = False
        self._preprocessed = False
        self._phase_shifts_consolidated = False

    def add_delta(self, delta):
        if not delta.is_near_zero:
            self.pulse_deltas.append(delta)
            self._consolidated = False
        # always update end time
        self._update_end_time(delta.time)

    def _update_end_time(self, t):
        if t != np.inf and t > self._end_time:
            self._end_time = t

    def add_MW_data(self, MW_data_object):
        """
        add object that defines a microwave pulse.

        Args:
            MW_data_object (IQ_data_single) : description MW pulse (see pulse_lib.segments.data_classes.data_IQ)
        """
        self.MW_pulse_data.append(MW_data_object)
        self._update_end_time(MW_data_object.stop)

    def add_custom_pulse_data(self, custom_pulse:custom_pulse_element):
        self.custom_pulse_data.append(custom_pulse)
        self._update_end_time(custom_pulse.stop)

    def add_phase_shift(self, phase_shift:PhaseShift):
        self._phase_shifts_consolidated = False
        self.phase_shifts.append(phase_shift)
        self._update_end_time(phase_shift.time)

    @property
    def total_time(self):
        '''
        total time of the current segment

        Returns:
            total_time (float) : total time of the segment.
        '''
        return self._end_time

    def reset_time(self, time):
        '''
        Preform a reset time on the current segment.
        Args:
            time (float) : time where you want the reset. Of None, the totaltime of the segment will be taken.
            extend_only (bool) : will just extend the time in the segment and not reset it if set to true [do not use when composing wavoforms...].
        '''
        if time is None:
            time = self.total_time
        else:
            self._update_end_time(time)

        self.start_time = time

    def wait(self, time):
        """
        Wait after last point for x ns (note that this does not reset time)

        Args:
            time (double) : time in ns to wait
        """
        self._end_time += time

    def append(self, other):
        '''
        Append two segments to each other, where the other segment is places after the first segment.
        Args:
            other (pulse_data) : other pulse data object to be appended

        '''
        time = self.total_time

        other_MW_pulse_data = copy.deepcopy(other.MW_pulse_data)
        shift_start_stop(other_MW_pulse_data, time)
        other_custom_pulse_data = copy.deepcopy(other.custom_pulse_data)
        shift_start_stop(other_custom_pulse_data, time)

        other_phase_shifts = copy.deepcopy(other.phase_shifts)
        shift_time(other_phase_shifts, time)
        other_pulse_deltas = copy.deepcopy(other.pulse_deltas)
        shift_time(other_pulse_deltas, time)

        self.pulse_deltas += other_pulse_deltas
        self.MW_pulse_data += other_MW_pulse_data
        self.custom_pulse_data += other_custom_pulse_data
        self.phase_shifts += other_phase_shifts

        self._consolidated = False
        self._phase_shifts_consolidated = False
        self._update_end_time(time + other.total_time)

    def repeat(self, n):
        """
        repeat n times
        Args
            n (int) : number of times to repeat
        """
        time = self.total_time

        new_pulse_deltas = copy.copy(self.pulse_deltas)
        new_MW_pulse_data =  copy.copy(self.MW_pulse_data)
        new_custom_pulse_data =  copy.copy(self.custom_pulse_data)
        new_phase_shifts =  copy.copy(self.phase_shifts)

        for i in range(n):
            shifted_pulse_deltas = copy.deepcopy(self.pulse_deltas)
            shift_time(shifted_pulse_deltas, (i+1)*time)
            new_pulse_deltas += shifted_pulse_deltas

            shifted_MW_pulse_data = copy.deepcopy(self.MW_pulse_data)
            shift_start_stop(shifted_MW_pulse_data, (i+1)*time)
            new_MW_pulse_data +=  shifted_MW_pulse_data

            shifted_custom_pulse_data = copy.deepcopy(self.custom_pulse_data)
            shift_start_stop(shifted_custom_pulse_data, (i+1)*time)
            new_custom_pulse_data +=  shifted_custom_pulse_data

            shifted_phase_shifts = copy.deepcopy(self.phase_shifts)
            shift_time(shifted_phase_shifts, (i+1)*time)
            new_phase_shifts += shifted_phase_shifts

        self.pulse_deltas = new_pulse_deltas
        self.MW_pulse_data = new_MW_pulse_data
        self.custom_pulse_data = new_custom_pulse_data
        self.phase_shifts = new_phase_shifts

        self._consolidated = False
        self._phase_shifts_consolidated = False
        self._end_time = (n+1) * time

    def shift_MW_frequency(self, frequency):
        '''
        shift the frequency of a MW signal that is defined. This is needed for dealing with the upconverion of a IQ signal.

        Args:
            frequency (float) : frequency you want to shift
        '''
        for mw_pulse in self.MW_pulse_data:
            mw_pulse.frequency -= frequency

    def shift_MW_phases(self, phase_shift):
        '''
        Shift the phases of all the microwaves present in the MW data object

        Args:
            phase_shift (float) : amount of phase to shift in rad.
        '''
        if phase_shift == 0:
            return

        for mw_pulse in self.MW_pulse_data:
            mw_pulse.start_phase += phase_shift


    '''
    operators for the data object.
    '''
    def __copy__(self):
        # NOTE: copy is called in pulse_data_all, before adding virtual channels.
        self._consolidate()
        my_copy = pulse_data()
        my_copy.pulse_deltas = copy.deepcopy(self.pulse_deltas)
        my_copy.MW_pulse_data = copy.deepcopy(self.MW_pulse_data)
        my_copy.phase_shifts = copy.copy(self.phase_shifts)
        my_copy.custom_pulse_data = copy.deepcopy(self.custom_pulse_data)
        my_copy.start_time = copy.copy(self.start_time)
        my_copy._end_time = self._end_time
        my_copy._consolidated = self._consolidated
        my_copy._phase_shifts_consolidated = self._phase_shifts_consolidated

        return my_copy

    def __add__(self, other):
        '''
        define addition operator for pulse_data object
        '''
        new_data = pulse_data()
        new_data.start_time = copy.copy(self.start_time)

        if isinstance(other, pulse_data):
            new_data.pulse_deltas = self.pulse_deltas + other.pulse_deltas
            new_data.MW_pulse_data = self.MW_pulse_data + other.MW_pulse_data
            new_data.phase_shifts = self.phase_shifts + other.phase_shifts
            new_data.custom_pulse_data = self.custom_pulse_data + other.custom_pulse_data
            new_data._end_time = max(self._end_time, other._end_time)

        elif isinstance(other, Number):
            # copy, because only new elements added to list
            new_pulse = copy.copy(self.pulse_deltas)
            new_pulse.insert(0, pulse_delta(0, other, 0))
            new_pulse.append(pulse_delta(np.inf, -other, 0))
            new_data.pulse_deltas = new_pulse

            new_data.MW_pulse_data = copy.copy(self.MW_pulse_data)
            new_data.phase_shifts = copy.copy(self.phase_shifts)
            new_data.custom_pulse_data = copy.copy(self.custom_pulse_data)
            new_data._end_time = self._end_time

        else:
            raise TypeError(f'Cannot add pulse_data to {type(other)}')

        return new_data

    def __iadd__(self, other):
        '''
        define addition operator for pulse_data object
        '''
        if isinstance(other, pulse_data):
            self.pulse_deltas += other.pulse_deltas
            self.MW_pulse_data += other.MW_pulse_data
            self.phase_shifts += other.phase_shifts
            self.custom_pulse_data += other.custom_pulse_data
            self._end_time = max(self._end_time, other._end_time)
            self._phase_shifts_consolidated = False

        elif isinstance(other, Number):
            self.pulse_deltas.insert(0, pulse_delta(0, other, 0))
            self.pulse_deltas.append(pulse_delta(np.inf, -other, 0))

        else:
            raise TypeError(f'Cannot add pulse_data to {type(other)}')

        self._consolidated = False
        return self

    def __mul__(self, other):
        '''
        multiplication operator for segment_single
        '''
        # consolidate to reduce number of elements.
        # multiplication is applied during rendering a good moment to reduce number of elements.
        self._consolidate()
        new_data = pulse_data()

        if isinstance(other, Number):
            # deepcopy, because elements are modified
            new_data.pulse_deltas = copy.deepcopy(self.pulse_deltas)
            for delta in new_data.pulse_deltas:
                delta *= other

            new_data.MW_pulse_data = copy.deepcopy(self.MW_pulse_data)
            for mw_pulse in new_data.MW_pulse_data:
                mw_pulse.amplitude *=other

            new_data.custom_pulse_data = copy.deepcopy(self.custom_pulse_data)
            for custom_pulse in new_data.custom_pulse_data:
                custom_pulse.scaling *= other

            new_data.phase_shifts = copy.copy(self.phase_shifts)
            new_data._end_time = self._end_time
            new_data.start_time = self.start_time
            new_data._consolidated = self._consolidated
            new_data._phase_shifts_consolidated = self._phase_shifts_consolidated
        else:
            raise TypeError(f'Cannot multiply pulse_data with {type(other)}')

        return new_data

    def _consolidate(self):
        # merge deltas with same time.
        if self._consolidated:
            return
        if len(self.pulse_deltas) == 1:
            logging.error(f'Asjemenou {self.pulse_deltas}')
            raise Exception(f'Error in pulse data: {self.pulse_deltas}')

        if len(self.pulse_deltas) > 1:
            self.pulse_deltas.sort(key=lambda p:p.time)
            new_deltas = []
            last = self.pulse_deltas[0]
            for delta in self.pulse_deltas[1:]:
                if delta.time == last.time:
                    last = last + delta
                else:
                    if not last.is_near_zero:
                        new_deltas.append(last)
                    last = delta
            if not last.is_near_zero:
                new_deltas.append(last)

            self.pulse_deltas = new_deltas

        self._consolidated = True
        self._preprocessed = False

    def _pre_process(self):
        self._consolidate()
        if not self._preprocessed:
            n = len(self.pulse_deltas)
            if n == 0:
                times = np.zeros(0)
                intervals = np.zeros(0)
                amplitudes = np.zeros(0)
                amplitudes_end = np.zeros(0)
                ramps = np.zeros(0)
            else:
                times = np.zeros(n)
                intervals = np.zeros(n)
                steps = np.zeros(n)
                ramps = np.zeros(n)
                amplitudes = np.zeros(n)
                amplitudes_end = np.zeros(n)
                for i,delta in enumerate(self.pulse_deltas):
                    times[i] = delta.time
                    steps[i] = delta.step
                    ramps[i] = delta.ramp
                if times[-1] == np.inf:
                    times[-1] = self._end_time
                intervals[:-1] = times[1:] - times[:-1]
                ramps = np.cumsum(ramps)
                amplitudes[1:] = ramps[:-1] * intervals[:-1]
                amplitudes = np.cumsum(amplitudes) + np.cumsum(steps)
                amplitudes_end[:-1] = amplitudes[1:] - steps[1:]
    #            logging.debug(f'points: {list(zip(times, amplitudes))}')
            self._times = times
            self._intervals = intervals
            self._amplitudes = amplitudes
            self._amplitudes_end = amplitudes_end
            self._ramps = ramps
        self._preprocessed = True

    def integrate_waveform(self, sample_rate):
        '''
        takes a full integral of the currently scheduled waveform.
        Args:
            sample_rate (double) : rate at which the AWG will be run
        Returns:
            integrate (double) : the integrated value of the waveform (unit is mV/sec).
        '''
        self._pre_process()

        integrated_value = 0

        if len(self.pulse_deltas) > 0:
            integrated_value = 0.5*np.dot((self._amplitudes[:-1] + self._amplitudes_end[:-1]),
                                          self._intervals[:-1])

        for custom_pulse in self.custom_pulse_data:
            integrated_value += np.sum(custom_pulse.render(sample_rate))

        integrated_value *= 1e-9

        return integrated_value

    def _consolidate_phase_shifts(self):
        if self._phase_shifts_consolidated:
            return

        if len(self.phase_shifts) > 1:
            self.phase_shifts.sort(key=lambda p:p.time)
            new_shifts = []
            last = self.phase_shifts[0]
            for phase_shift in self.phase_shifts[1:]:
                if phase_shift.time == last.time:
                    last = last + phase_shift
                else:
                    if not last.is_near_zero:
                        new_shifts.append(last)
                    last = phase_shift
            if not last.is_near_zero:
                new_shifts.append(last)

            self.pulse_deltas = new_shifts

        self._phase_shifts_consolidated = True

    def get_data_elements(self):
        elements = []
        self._pre_process()
        self._consolidate_phase_shifts()
        for time, duration, v_start, v_stop in zip(self._times, self._intervals,
                                                   self._amplitudes, self._amplitudes_end):
            elements.append(OffsetRamp(time, time+duration, v_start, v_stop))
        elements += self.custom_pulse_data
        elements += self.MW_pulse_data
        elements += self.phase_shifts
        elements.sort(key=lambda p:(p.start,p.stop))
        return elements

    def _render(self, sample_rate, ref_channel_states, LO):
        '''
        make a full rendering of the waveform at a predetermined sample rate.
        '''
        self._pre_process()

        # express in Gs/s
        sample_rate = sample_rate*1e-9

        t_tot = self.total_time

        # get number of points that need to be rendered
        t_tot_pt = iround(t_tot * sample_rate) + 1

        wvf = np.zeros([int(t_tot_pt)])

        t_pt = iround(self._times * sample_rate)

        for i in range(len(t_pt)-1):
            pt0 = t_pt[i]
            pt1 = t_pt[i+1]
            if pt0 != pt1:
                if self._ramps[i] != 0:
                    wvf[pt0:pt1] = np.linspace(self._amplitudes[i], self._amplitudes_end[i], pt1-pt0+1)[:-1]
                else:
                    wvf[pt0:pt1] = self._amplitudes[i]

        # render MW pulses.
        # create list with phase shifts per ref_channel
        phase_shifts_channels = {}
        for ps in self.phase_shifts:
            ps_ch = phase_shifts_channels.setdefault(ps.channel_name, [])
            ps_ch.append(ps)

        for mw_pulse in self.MW_pulse_data:
            # start stop time of MW pulse

            start_pulse = mw_pulse.start
            stop_pulse = mw_pulse.stop

            # max amp, freq and phase.
            amp  =  mw_pulse.amplitude
            freq =  mw_pulse.frequency
            if LO:
                freq -= LO
            if abs(freq) > sample_rate*1e9/2:
                raise Exception(f'Frequency {freq*1e-6:5.1f} MHz is above Nyquist frequency ({sample_rate*1e3/2} MHz)')
            # TODO add check on configurable bandwidth.
            phase = mw_pulse.start_phase
            if ref_channel_states and mw_pulse.ref_channel in ref_channel_states.start_phase:
                ref_start_time = ref_channel_states.start_time
                ref_start_phase = ref_channel_states.start_phase[mw_pulse.ref_channel]
            else:
                ref_start_time = 0
                ref_start_phase = 0

            if mw_pulse.ref_channel in phase_shifts_channels:
                phase_shifts = [
                        ps.phase_shift
                        for ps in phase_shifts_channels[mw_pulse.ref_channel]
                        if ps.time <= start_pulse
                        ]
                phase_shift = sum(phase_shifts)
            else:
                phase_shift = 0

            # envelope data of the pulse
            if mw_pulse.envelope is None:
                mw_pulse.envelope = envelope_generator()

            amp_envelope = mw_pulse.envelope.get_AM_envelope((stop_pulse - start_pulse), sample_rate)
            phase_envelope = np.asarray(mw_pulse.envelope.get_PM_envelope((stop_pulse - start_pulse), sample_rate))

            #self.baseband_pulse_data[-1,0] convert to point numbers
            n_pt = int((stop_pulse - start_pulse) * sample_rate) if isinstance(amp_envelope, float) else len(amp_envelope)
            start_pt = iround(start_pulse * sample_rate)
            stop_pt = start_pt + n_pt

            # add the sin pulse
            total_phase = phase_shift + phase + phase_envelope + ref_start_phase
            t = start_pt+ref_start_time/sample_rate + np.arange(n_pt)
            wvf[start_pt:stop_pt] += amp*amp_envelope*np.sin(2*np.pi*freq/sample_rate*1e-9*t + total_phase)

        for custom_pulse in self.custom_pulse_data:
            data = custom_pulse.render(sample_rate*1e9)
            start_pt = iround(custom_pulse.start * sample_rate)
            stop_pt = start_pt + len(data)
            wvf[start_pt:stop_pt] += data

        # remove last value. t_tot_pt = t_tot + 1. Last value is always 0. It is only needed in the loop on the pulses.
        return wvf[:-1]

    def get_accumulated_phase(self):
        phase = 0
        for shift in self.phase_shifts:
            phase += shift.phase_shift
        # print(f'accumulated {phase} ({len(self.phase_shifts)})')
        return phase

    def _merge_elements(self, elements):
        if len(elements) < 1:
            return elements
        elements.sort(key=lambda e:e.start)
        result = []
        last = elements[0]
        for element in elements[1:]:
            if element.start < last.stop:
                nw_wvf = np.zeros(element.stop - last.start)
                nw_wvf[:len(last.wvf)] = last.wvf
                nw_wvf[-len(element.wvf):] += element.wvf
                last = rendered_element(last.start, element.stop, nw_wvf)
            else:
                result.append(last)
                last = element
        result.append(last)
        return result

    def render_MW_and_custom(self, sample_rate, ref_channel_states): # @@@ Is this method used anywhere?
        '''
        Render MW pulses and custom data in 'rendered_elements'.
        '''
        elements = []

        self._pre_process()

        # express in Gs/s
        sample_rate = sample_rate*1e-9

        # render MW pulses.
        # create list with phase shifts per ref_channel
        phase_shifts_channels = {}
        for ps in self.phase_shifts:
            ps_ch = phase_shifts_channels.setdefault(ps.channel_name, [])
            ps_ch.append(ps)

        for mw_pulse in self.MW_pulse_data:
            # start stop time of MW pulse

            start_pulse = mw_pulse.start
            stop_pulse = mw_pulse.stop

            # max amp, freq and phase.
            amp  =  mw_pulse.amplitude
            freq =  mw_pulse.frequency
            phase = mw_pulse.start_phase
            if ref_channel_states and mw_pulse.ref_channel in ref_channel_states.start_phase:
                ref_start_time = ref_channel_states.start_time
                ref_start_phase = ref_channel_states.start_phase[mw_pulse.ref_channel]
                phase_shift = 0
                if mw_pulse.ref_channel in phase_shifts_channels:
                    for ps in phase_shifts_channels[mw_pulse.ref_channel]:
                         if ps.time <= start_pulse:
                             phase_shift += ps.phase_shift
            else:
                ref_start_time = 0
                ref_start_phase = 0
                phase_shift = 0

            # envelope data of the pulse
            if mw_pulse.envelope is None:
                mw_pulse.envelope = envelope_generator()

            amp_envelope = mw_pulse.envelope.get_AM_envelope((stop_pulse - start_pulse), sample_rate)
            phase_envelope = mw_pulse.envelope.get_PM_envelope((stop_pulse - start_pulse), sample_rate)

            #self.baseband_pulse_data[-1,0] convert to point numbers
            n_pt = int((stop_pulse - start_pulse) * sample_rate) if isinstance(amp_envelope, float) else len(amp_envelope)
            start_pt = iround(start_pulse * sample_rate)
            stop_pt = start_pt + n_pt

            # add the sin pulse
            total_phase = phase_shift + phase + phase_envelope + ref_start_phase
            t = start_pt+ref_start_time/sample_rate + np.arange(n_pt)
            wvf = amp*amp_envelope*np.sin(2*np.pi*freq/sample_rate*1e-9*t + total_phase)
            elements.append(rendered_element(start_pt, stop_pt, wvf))

        for custom_pulse in self.custom_pulse_data:
            wvf = custom_pulse.render(sample_rate*1e9)
            start_pt = iround(custom_pulse.start * sample_rate)
            stop_pt = start_pt + len(wvf)
            elements.append(rendered_element(start_pt, stop_pt, wvf))

        return self._merge_elements(elements)

    def get_metadata(self, name):
        metadata = {}

        self._pre_process()

        # TODO: add custom pulses

        j = 0
        bb_d = {}
        for i in range(len(self._times)-1):
            start = self._times[i]
            stop = self._times[i+1]
            v_start = self._amplitudes[i]
            v_stop = self._amplitudes_end[i]
            if stop == np.inf:
                stop = self._end_time
            if stop - start < 1 or (v_start == 0 and v_stop == 0):
                continue
            bb_d[f'p{j}'] = {
                'start':start,
                'stop':stop,
                'v_start':v_start,
                'v_stop':v_stop
                }
            j += 1

        if bb_d:
            metadata[name+'_baseband'] = bb_d

        pulsedata = self.MW_pulse_data
        all_pulse = {}
        for (i,pulse) in enumerate(pulsedata):
            phase_shift = 0
            for ps in self.phase_shifts:
                 if ps.time <= pulse.start:
                     phase_shift += ps.phase_shift
            pd = {}
            pd['start'] = pulse.start
            pd['stop'] = pulse.stop
            pd['amplitude'] = pulse.amplitude
            pd['frequency'] = pulse.frequency
            pd['start_phase'] = pulse.start_phase + phase_shift
            envelope = pulse.envelope
            if envelope is None:
                envelope = envelope_generator()
            pd['AM_envelope'] = repr(envelope.AM_envelope_function)
            pd['PM_envelope'] = repr(envelope.PM_envelope_function)
            all_pulse[('p%i' %i)] = pd

        if all_pulse:
            metadata[name+'_pulses'] = all_pulse

        return metadata

