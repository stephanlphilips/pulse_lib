 import numpy as np


 class pulselib:
    self.backend = 'keysight'

    def __init__(self):
         self.awg_channels = ['names']
         self.awg_channels_to_physical_locations = ['here_dict']
         self.marker_channels =['names']
         self.marger_channels_to_location = []
         self.delays = []
         self.convertion_matrix= []

         self.segments=segment_container(channels)

    def mk_segment(self, name):
        self.segement.append()

    def upload_data()
        
    def play()


class my_segment():
    self.name = ''
    self.channels = []
    self.
    def __init__(self, channels):
        self.channels = channels
    
    def add_pulse(array, channel):

    def reset_time():
        # aligns all time together -- the channel with the longest time will be chosen


class segment_container:
    self.segment = []
    
    def __init__(self):
        return

    def add_seggment(name, channels):
        if exists(name):
            raise ValueError("sement with the name : % \n alreadt exists"%name)
        self.segment.append(my_segment(channels))
        return self.get_segment(name)

    def get_segment(name):
        for i in self.segment:
            if i.name == name:
                return i
        raise ValueError("segment not found :(")

class channel_data_obj():
    #object containing the data for a specific channels
    #the idea is that all the data is parmeterised and will be constuceted whenever the function is called.

    self.my_data_array = np.empty()
    
    add_data

class block_pulses:
    # class to make block pulses

how to do pulses
-> sin?
-> pulses?
-> step_pulses

p = pulselin()

seg = pulselib.mk_segment('manip')
seg.p1.add_pulse(10,50, 20, prescaler= '1')
seg.p3.add_pulse(12,34, 40,)
seg.k2.add_pulse_advanced([pulse sequence])
seg.add_np(array, tstart_t_stop
seg.p5.add_sin(14,89, freq, phase, amp)

pulse
