'''
Class that can be used to construct IQ pulses for qubit control applications.

Possible pulses include:o
* standard block pulse
* chirped pulse for adiabatic spin operation
* modulated pulsed.

As data format we will use a class to store
* type (std, chrip, mod)
* t0
* te
* freq1
* freq2 (opt)
* amp
* phase

TODO : change dicts to keep the data to an object!!
'''
import numpy as np
import copy

from pulse_lib.segments.segment_base import segment_base
from pulse_lib.segments.utility.data_handling_functions import loop_controller, update_dimension
from pulse_lib.segments.data_classes.data_pulse import pulse_data, PhaseShift
from pulse_lib.segments.data_classes.data_IQ import envelope_generator, IQ_data_single, Chirp
from pulse_lib.segments.data_classes.data_markers import marker_data
from pulse_lib.segments.data_classes.data_generic import data_container


class segment_IQ(segment_base):
    """
    Standard single segment for IQ purposes
    todo --> add global phase and time shift in the data class instead of this one (cleaner and more generic).
    """
    def __init__(self, name, qubit_channel, HVI_variable_data = None):
        '''
        Args:
            name : name of the IQ segment
            HVI_variable_data (segment_HVI_variables) : segment used to keep variables that can be used in HVI.

        Tip, make on of these segments for each qubit. Then you get a very clean implementation of reference frame changes!
        '''
        # @@@ Fix segment_type with rendering refactoring
        super().__init__(name, pulse_data(), HVI_variable_data) #, segment_type = 'IQ_virtual')
        self._qubit_channel = qubit_channel

    def __copy__(self):
        cpy = segment_IQ(self.name, self._qubit_channel, self._data_hvi_variable)
        return self._copy(cpy)

    @loop_controller
    def add_phase_shift(self, t, phase):
        self.data_tmp.add_phase_shift(PhaseShift(t + self.data_tmp.start_time, phase, self.name))
        return self.data_tmp

    @loop_controller
    def add_MW_pulse(self, t0, t1, amp, freq, phase = 0, AM = None, PM = None):
        '''
        Make a sine pulse (generic constructor)

        Args:
            t0(float) : start time in ns
            t1(float) : stop tiume in ns
            amp (float) : amplitude of the pulse.
            freq(float) : frequency
            phase (float) : phase of the microwave.
            AM ('str/tuple/function') : function describing an amplitude modulation (see examples in pulse_lib.segments.data_classes.data_IQ)
            PM ('str/tuple/function') : function describing an phase modulation (see examples in pulse_lib.segments.data_classes.data_IQ)
        '''
        MW_data = IQ_data_single(t0 + self.data_tmp.start_time,
                                 t1 + self.data_tmp.start_time,
                                 amp, freq,
                                 phase,
                                 envelope_generator(AM, PM),
                                 self.name)
        self.data_tmp.add_MW_data(MW_data)
        return self.data_tmp

    @loop_controller
    def add_chirp(self, t0, t1, f0, f1, amp):
        '''
        Add chirp to the segment.
        Args:
            t0(float) : start time in ns
            t1(float) : stop tiume in ns
            f0(float) : start frequency
            f1 (float) : stop frequency
            amp (float) : amplitude of the pulse.
        '''
        chirp = Chirp(t0 + self.data_tmp.start_time,
                      t1 + self.data_tmp.start_time,
                      amp, f0, f1, self.name)
        self.data_tmp.add_chirp(chirp)
        return self.data_tmp

    def get_IQ_data(self, out_channel_info):
        '''
        get I and Q data from the main element.
        Args:
            out_channel_info (IQ_render_info): render info like LO and I/Q
        Returns:
            data (np.ndarray<pulse_data>) : array with the pulsedata objects inside
        '''
        qubit_channel = self._qubit_channel
        LO = qubit_channel.iq_channel.LO

        phase_shift = 0
        if out_channel_info.IQ_comp == 'I':
            phase_shift += np.pi/2
            correction_gain = qubit_channel.correction_gain[0] if qubit_channel.correction_gain is not None else 1.0
        else:
            if qubit_channel.correction_phase is not None:
                phase_shift += qubit_channel.correction_phase
            correction_gain = qubit_channel.correction_gain[1] if qubit_channel.correction_gain is not None else 1.0
        if out_channel_info.image == '-':
            phase_shift += np.pi

        local_data = copy.copy(self.data).flatten()
        # downconvert the sigal saved in the data object, so later on, in the real MW source, it can be upconverted again.
        for i in range(len(local_data)):
            local_data[i] = self.data.flat[i] * correction_gain
            local_data[i].shift_MW_phases(phase_shift)
            local_data[i].shift_MW_frequency(LO)

        local_data = local_data.reshape(self.data.shape)

        return local_data

    def get_marker_data(self):
        '''
        generate markers for the PM of the IQ modulation
        '''
        my_marker_data = update_dimension(data_container(marker_data()), self.shape)
        my_marker_data = my_marker_data.flatten()

        # make a flat reference.
        local_data = self.data.flatten()

        for i in range(len(local_data)):
            for MW_pulse_info in local_data[i].MW_pulse_data:
                my_marker_data[i].add_marker(MW_pulse_info.start, MW_pulse_info.stop)
            for chirp in local_data[i].chirp_data:
                my_marker_data[i].add_marker(chirp.start, chirp.stop)

        my_marker_data = my_marker_data.reshape(self.shape)

        return my_marker_data

    def get_accumulated_phase(self, index):
        return self._get_data_all_at(index).get_accumulated_phase()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from scipy import signal

    def gaussian_sloped_envelope(delta_t, sample_rate = 1):
        """
        function that has blackman slopes at the start and at the end (first 8 and last 8 ns)

        Args:
            delta_t (double) : time in ns of the pulse.
            sample_rate (double) : sampling rate of the pulse (GS/s).

        Returns:
            evelope (np.ndarray) : array of the evelope.
        """

        n_points = int(delta_t*sample_rate + 0.9)
        envelope = np.ones([n_points], np.double)
        if delta_t < 20:
            envelope = signal.get_window('blackman', n_points*10)[::10]
        else:
            time_slope = (20 + delta_t)*sample_rate - int(delta_t*sample_rate)
            envelope_left_right = signal.get_window('blackman', int(time_slope*10))[::10]

            half_pt_gauss = int(time_slope/2)

            envelope[:half_pt_gauss] = envelope_left_right[:half_pt_gauss]
            envelope[half_pt_gauss:half_pt_gauss+n_points-int(time_slope)] = 1
            envelope[n_points-len(envelope_left_right[half_pt_gauss:]):] = envelope_left_right[half_pt_gauss:]

        return envelope

    s1 = segment_IQ("test")

    s1.add_MW_pulse(0,100,1,1e9,0, gaussian_sloped_envelope)
    s1.reset_time()
    # s1.add_chirp(1500,2500,0e7,1e7,1)
    s1.plot_segment(sample_rate = 1e10)
    plt.show()