import numpy as np
import copy

from pulse_lib.segments.segment_container import segment_container
from pulse_lib.sequencer import sequencer

from pulse_lib.virtual_channel_constructors import virtual_gates_constructor

from pulse_lib.keysight.M3202A_uploader import M3202A_Uploader

class pulselib:
    '''
    Global class that is an organisational element in making the pulses.
    The idea is that you first make individula segments,
    you can than later combine them into a sequence, this sequence will be uploaded
    '''
    def __init__(self, backend = "M3202A"):
        # awg channels and locations need to be input parameters.
        self.awg_devices = dict()
        self.awg_channels = []
        self.awg_markers = []
        self.virtual_channels = []
        self.IQ_channels = []

        self._backend = backend

        self.channel_delays = dict()
        self.channel_delays_computed = dict()
        self.channel_compenstation_limits = dict()
        self.channels_to_physical_locations = dict()
        self.AWG_to_dac_ratio = dict()

        self.delays = []
        self.convertion_matrix= []
        self.voltage_limits_correction = dict()

        self.segments_bin = None
        self.sequencer = None
        self.cpp_uploader = None

    @property
    def channels(self):
        channels = copy.copy(self.awg_channels)
        for i in self.virtual_channels:
            channels += i.virtual_gate_names
        return channels

    def add_awgs(self, name, awg):
        '''
        add a awg to the library
        Args:
            name (str) : name you want to give to a peculiar AWG
            awg (object) : qcodes object of the concerning AWG
        '''
        self.awg_devices[name] =awg
        if awg is not None and self.cpp_uploader is not None:
            self.cpp_uploader.add_awg_module(name, awg)

    def define_channel(self, channel_name, AWG_name, channel_number):
        '''
        define the channels and their location
        Args:
            channel_name (str) : name of a given channel on the AWG. This would usually the name of the gate that it is connected to.
            AWG_name (str) : name of the instrument (as given in add_awgs())
            channel_number (int) : channel number on the AWG
        '''
        self._check_uniqueness_of_channel_name(channel_name)

        self.awg_channels.append(channel_name)
        self.AWG_to_dac_ratio[channel_name] = 1

        # initialize basic properties of the channel
        self.channel_delays[channel_name] = 0
        self.channel_delays_computed[channel_name] = (0,0)
        self.channel_compenstation_limits[channel_name] = (0,0)
        self.channels_to_physical_locations[channel_name] = (AWG_name, channel_number)

    def define_marker(self, marker_name, AWG_name, channel_number):
        '''
        define the channels and their location
        Args:
            marker_name (str) : name of a given channel on the AWG. This would usually the name of the gate that it is connected to.
            AWG_name (str) : name of the instrument (as given in add_awgs())
            channel_number (int) : channel number on the AWG
        '''
        self.awg_markers.append(marker_name)
        self.define_channel(marker_name, AWG_name, channel_number)

    def add_channel_delay(self, channel, delay):
        '''
        Adds to a channel a delay.
        The delay is added by adding points in front of the first sequence/or
        just after the last sequence. The first value of the sequence will be
        taken as an extentsion point.

        Args:
            channel (str) : channel name as defined in self.define_channel().
            delay (int): delay of the current coax line (this may be a postive or negative number)

        Returns:
            0/Error
        '''
        if channel in self.awg_channels:
            self.channel_delays[channel] = delay
        else:
            raise ValueError("Channel delay error: Channel '{}' does not exist. Please provide valid input".format(channel))

        self.__process_channel_delays()
        return 0

    def add_channel_compenstation_limit(self, channel_name, limit):
        '''
        add voltage limitations per channnel that can be used to make sure that the intragral of the total voltages is 0.
        Args:
            channel (str) : channel name as defined in self.define_channel().
            limit (tuple<float,float>) : lower/upper limit for DC compensation, e.g. (-100,500)
        Returns:
            None
        '''
        if channel_name in self.awg_channels:
            self.channel_compenstation_limits[channel_name] = limit
        else:
            raise ValueError("Channel voltage compenstation error: Channel '{}' does not exist. Please provide valid input".format(channel_name))

    def finish_init(self):
        # function that finishes the initialisation
        # TODO rewrite, so this function is embedded in the other ones.

        if self._backend == "keysight":
            
            from pulse_lib.keysight.uploader import keysight_uploader
            from pulse_lib.keysight.uploader_core.uploader import keysight_upload_module

            self.cpp_uploader = keysight_upload_module()
            for name, awg in self.awg_devices.items():
                if awg is not None:
                    self.cpp_uploader.add_awg_module(name, awg)

            self.uploader = keysight_uploader(self.awg_devices, self.cpp_uploader, self.awg_channels,
                                              self.channels_to_physical_locations , self.channel_delays_computed,
                                              self.channel_compenstation_limits, self.AWG_to_dac_ratio)
        elif self._backend == "M3202A":
            self.uploader = M3202A_Uploader(self.awg_devices, self.awg_channels, self.channels_to_physical_locations,
                                            self.channel_delays_computed, self.channel_compenstation_limits, self.AWG_to_dac_ratio)

    def mk_segment(self):
        '''
        generate a new segment.
        Returns:
            segment (segment_container) : returns a container that contains all the previously defined gates.
        '''
        return segment_container(self.awg_channels, self.awg_markers, self.virtual_channels, self.IQ_channels)

    def mk_sequence(self,seq):
        '''
        seq: list of segment_container.
        '''
        seq_obj = sequencer(self.uploader, self.voltage_limits_correction)
        seq_obj.add_sequence(seq)
        seq_obj.metadata = {}
        for (i,pc) in enumerate(seq):
            md = pc.get_metadata()
            seq_obj.metadata[('pc%i'%i)] = md
        LOdict = {}
        for iq in self.IQ_channels:
            virt_maps = iq.virtual_channel_map
            for vm in virt_maps:
                name = vm.channel_name
                LOdict[name] = iq.LO
        seq_obj.metadata['LOs'] = LOdict
        return seq_obj

    def release_awg_memory(self, wait_idle=True):
        """
        Releases AWG waveform memory.
        Also flushes AWG queues.
        """
        if wait_idle:
            self.uploader.wait_until_AWG_idle()

        self.uploader.release_memory()

        for awg_name, channel_number in self.channels_to_physical_locations.values():
            awg = self.awg_devices[awg_name]
            awg.awg_flush(channel_number)


    def load_hardware(self, hardware):
        '''
        load virtual gates and attenuation via the harware class (used in qtt)

        Args:
            hardware (harware_parent) : harware class.
        '''
        for virtual_gate_set in hardware.virtual_gates:
            vgc = virtual_gates_constructor(self)
            vgc.load_via_harware(virtual_gate_set)

        # set output ratio's of the channels from the harware file.
        if self.AWG_to_dac_ratio.keys() == hardware.AWG_to_dac_conversion.keys():
            self.AWG_to_dac_ratio = hardware.AWG_to_dac_conversion
        else:
            hardware.AWG_to_dac_conversion = self.AWG_to_dac_ratio
            hardware.sync_data()

    def __process_channel_delays(self):
        '''
        Makes a variable that contains the amount of points that need to be put before and after when a upload is performed.
        '''
        self.channel_delays_computed = dict()

        for channel in self.channel_delays:
            self.channel_delays_computed[channel] = (self.__get_pre_delay(channel), self.__get_post_delay(channel))


    def __calculate_total_channel_delay(self):
        '''
        function for calculating how many ns time there is a delay in between the channels.
        Also support for negative delays...

        returns:
            tot_delay (the total delay)
            max_delay (hight amount of the delay)
        '''

        delays =  np.array( list(self.channel_delays.values()))
        tot_delay = np.max(delays) - np.min(delays)

        return tot_delay, np.max(delays)

    def __get_pre_delay(self, channel):
        '''
        get the of ns that a channel needs to be pushed forward/backward.
        returns
            pre-delay : number of points that need to be pushed in from of the segment
        '''
        tot_delay, max_delay = self.__calculate_total_channel_delay()
        max_pre_delay = tot_delay - max_delay
        delay = self.channel_delays[channel]
        return -(delay + max_pre_delay)

    def __get_post_delay(self, channel):
        '''
        get the of ns that a channel needs to be pushed forward/backward.
        returns
            post-delay: number of points that need to be pushed after the segment
        '''
        tot_delay, max_delay = self.__calculate_total_channel_delay()
        delay = self.channel_delays[channel]

        return -delay + max_delay

    def _check_uniqueness_of_channel_name(self, channel_name):
        if channel_name in self.awg_channels:
            raise ValueError("double declaration of the a channel/marker name ({}).".format(channel_name))


if __name__ == '__main__':
    from pulse_lib.virtual_channel_constructors import IQ_channel_constructor

    p = pulselib()


    class AWG(object):
        """docstring for AWG"""
        def __init__(self, name):
            self.name = name
            self.chassis = 0
            self.slot = 0
            self.type = "DEMO"

    AWG1 = AWG("AWG1")
    AWG2 = AWG("AWG2")
    AWG3 = AWG("AWG3")
    AWG4 = AWG("AWG4")

    # add to pulse_lib
    # p.add_awgs('AWG1',AWG1)
    # p.add_awgs('AWG2',AWG2)
    # p.add_awgs('AWG3',AWG3)
    # p.add_awgs('AWG4',AWG4)

    # define channels
    # p.define_channel('B0','AWG1', 1)
    p.define_channel('P1','AWG1', 2)
    p.define_channel('B1','AWG1', 3)
    p.define_channel('P2','AWG1', 4)
    p.define_channel('B2','AWG2', 1)
    p.define_channel('P3','AWG2', 2)
    p.define_channel('B3','AWG2', 3)
    p.define_channel('P4','AWG2', 4)
    p.define_channel('B4','AWG3', 1)
    p.define_channel('P5','AWG3', 2)
    p.define_channel('B5','AWG3', 3)
    p.define_channel('G1','AWG3', 4)
    p.define_channel('I_MW','AWG4',1)
    p.define_channel('Q_MW','AWG4',2)
    p.define_marker('M1','AWG4', 3)
    p.define_marker('M2','AWG4', 4)


    # format : channel name with delay in ns (can be posive/negative)
    p.add_channel_delay('I_MW',50)
    p.add_channel_delay('Q_MW',50)
    p.add_channel_delay('M1',20)
    p.add_channel_delay('M2',-25)

    # add limits on voltages for DC channel compenstation (if no limit is specified, no compensation is performed).
    # p.add_channel_compenstation_limit('B0', (-100, 500))

    try:
        from V2_software.drivers.virtual_gates.harware import hardware_example
        hw =  hardware_example("hw")
        p.load_hardware(hw)

    except:
        # set a virtual gate matrix (note that you are not limited to one matrix if you would which so)
        virtual_gate_set_1 = virtual_gates_constructor(p)
        virtual_gate_set_1.add_real_gates('P1','P2','P3','P4','P5','B0','B1','B2','B3','B4','B5')
        virtual_gate_set_1.add_virtual_gates('vP1','vP2','vP3','vP4','vP5','vB0','vB1','vB2','vB3','vB4','vB5')
        virtual_gate_set_1.add_virtual_gate_matrix(np.eye(11))

    #make virtual channels for IQ usage (also here, make one one of these object per MW source)
    IQ_chan_set_1 = IQ_channel_constructor(p)
    # set right association of the real channels with I/Q output.
    IQ_chan_set_1.add_IQ_chan("I_MW", "I")
    IQ_chan_set_1.add_IQ_chan("Q_MW", "Q")
    IQ_chan_set_1.add_marker("M1", -15, 15)
    IQ_chan_set_1.add_marker("M2", -15, 15)
    # set LO frequency of the MW source. This can be changed troughout the experiments, bit only newly created segments will hold the latest value.
    IQ_chan_set_1.set_LO(1e9)
    # name virtual channels to be used.
    IQ_chan_set_1.add_virtual_IQ_channel("MW_qubit_1")
    IQ_chan_set_1.add_virtual_IQ_channel("MW_qubit_2")

    print(p.channels)
    # p.finish_init()

    seg  = p.mk_segment()
    # seg2 = p.mk_segment()
    # seg3 = p.mk_segment()

    # seg.vP1.add_block(0,10,1)


    # # B0 is the barrier 0 channel
    # # adds a linear ramp from 10 to 20 ns with amplitude of 5 to 10.
    # seg.B0.add_pulse([[10.,0.],[10.,5.],[20.,10.],[20.,0.]])
    # # add a block pulse of 2V from 40 to 70 ns, to whaterver waveform is already there
    # seg.B0.add_block(40,70,2)
    # # just waits (e.g. you want to ake a segment 50 ns longer)
    # seg.B0.wait(50)
    # # resets time back to zero in segment. Al the commannds we run before will be put at a negative time.
    # seg.B0.reset_time()
    # # this pulse will be placed directly after the wait()
    # seg.B0.add_block(0,10,2)

