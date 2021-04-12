import numpy as np

from pulse_lib.segments.segment_container import segment_container
from pulse_lib.sequencer import sequencer
from pulse_lib.configuration.physical_channels import awg_channel, marker_channel
from pulse_lib.configuration.iq_channels import IQ_channel, qubit_channel

from pulse_lib.virtual_channel_constructors import virtual_gates_constructor
from pulse_lib.keysight.M3202A_uploader import M3202A_Uploader
from pulse_lib.keysight.qs_uploader import QsUploader

class pulselib:
    '''
    Global class that is an organisational element in making the pulses.
    The idea is that you first make individula segments,
    you can than later combine them into a sequence, this sequence will be uploaded
    '''
    def __init__(self, backend = "M3202A"):
        self.awg_devices = dict()
        self.digitizers = dict()
        self.awg_channels = dict()
        self.marker_channels = dict()
        self.digitizer_channels = dict()
        self.virtual_channels = []
        self.qubit_channels = dict() # TODO: add to name check
        self.IQ_channels = dict()

        self._backend = backend

    @property
    def channels(self):
        channels = []
        channels += self.awg_channels.keys()
        # Exclude marker_channels from channel list. channels property is used for 'sweepable' channels.
        # channels += self.marker_channels.keys()
        for i in self.virtual_channels:
            channels += i.virtual_gate_names
        return channels

    def add_awg(self, awg):
        '''
        add a awg to the library
        Args:
            awg (object) : qcodes AWG instrument
        '''
        self.awg_devices[awg.name] = awg

    def add_awgs(self, name, awg):
        '''
        add a awg to the library
        Args:
            name (str) : name you want to give to a peculiar AWG
            awg (object) : qcodes object of the concerning AWG
        '''
        if name != awg.name:
            raise Exception(f'name mismatch {name} != {awg.name}. Use awg.name or pulselib.add_awg(awg)')
        self.awg_devices[name] = awg

    def add_digitizer(self, digitizer):
        '''
        add a digitizer to the library
        Args:
            digitizer (object) : qcodes digitizer instrument
        '''
        self.digitizers[digitizer.name] = digitizer

    def define_channel(self, channel_name, AWG_name, channel_number):
        '''
        define the channels and their location
        Args:
            channel_name (str) : name of a given channel on the AWG. This would usually the name of the gate that it is connected to.
            AWG_name (str) : name of the instrument (as given in add_awgs())
            channel_number (int) : channel number on the AWG
        '''
        self._check_uniqueness_of_channel_name(channel_name)
        self.awg_channels[channel_name] = awg_channel(channel_name, AWG_name, channel_number)

    def define_marker(self, marker_name, AWG_name, channel_number, setup_ns=0, hold_ns=0,
                      amplitude=1000, invert=False):
        '''
        define the channels and their location
        Args:
            marker_name (str) : name of a given channel on the AWG. This would usually the name of the gate that it is connected to.
            AWG_name (str) : name of the instrument (as given in add_awgs())
            channel_number (int) : channel number on the AWG
            setup_ns (float): setup time for the device using the marker. marker raises `setup_ns` earlier.
            hold_ns (float): hold time for the device using the marker to ensure proper output. marker falls `hold_ns` later.
        '''
        self.marker_channels[marker_name] = marker_channel(marker_name, AWG_name, channel_number,
                                                           setup_ns, hold_ns, amplitude, invert)

    def add_channel_delay(self, channel, delay):
        '''
        Adds to a channel a delay.
        The delay is added by adding points in front of the first sequence/or
        just after the last sequence. The first value of the sequence will be
        taken as an extentsion point.

        Args:
            channel (str) : channel name as defined in self.define_channel().
            delay (int): delay of the current coax line (this may be a postive or negative number)
        '''
        if channel in self.awg_channels:
            self.awg_channels[channel].delay = delay
        else:
            raise ValueError(f"Channel delay error: Channel '{channel}' is not defined")


    def add_channel_compenstation_limit(self, channel_name, limit):
        # call the method with the correct name
        self.add_channel_compensation_limit(channel_name, limit)

    def add_channel_compensation_limit(self, channel_name, limit):
        '''
        add voltage limitations per channnel that can be used to make sure that the intregral of the total voltages is 0.
        Args:
            channel (str) : channel name as defined in self.define_channel().
            limit (tuple<float,float>) : lower/upper limit for DC compensation, e.g. (-100,500)
        '''
        if channel_name in self.awg_channels:
            self.awg_channels[channel_name].compensation_limits = limit
        else:
            raise ValueError(f"Channel compensation delay error: Channel '{channel_name}' is not defined")

    def add_channel_attenuation(self, channel_name, attenuation):
        '''
        Sets channel attenuation factor (AWG-to-DAC ratio).
        Args:
            channel_name (str) : channel name as defined in self.define_channel().
            attenuation (float) : attenuation factor
        '''
        if channel_name in self.awg_channels:
            self.awg_channels[channel_name].attenuation = attenuation
        else:
            raise ValueError(f"Channel '{channel_name}' is not defined")

    def define_IQ_channel(self, name):
        channel = IQ_channel(name)
        self.IQ_channel = self.IQ_channels[name] = channel
        return channel

    def define_qubit_channel(self, qubit_channel_name, IQ_channel_name, reference_frequency):
        iq_channel = self.IQ_channels[IQ_channel_name]
        qubit = qubit_channel(qubit_channel_name, reference_frequency)
        iq_channel.qubit_channels.append(qubit)
        self.qubit_channels[qubit_channel_name] = qubit

    def finish_init(self):
        # function that finishes the initialisation

        if self._backend == "keysight":
            raise Exception('Old keysight driver is not supported anymore. Use M3202A driver and backend="M3202A"')

        elif self._backend == "M3202A":
            self.uploader = M3202A_Uploader(self.awg_devices, self.awg_channels,
                                            self.marker_channels, self.qubit_channels)

        elif self._backend == "Keysight_QS":
            self.uploader = QsUploader(self.awg_devices, self.awg_channels, self.marker_channels,
                                       self.IQ_channels, self.digitizers, self.digitizer_channels)

    def mk_segment(self, name=None, sample_rate=None):
        '''
        generate a new segment.
        Returns:
            segment (segment_container) : returns a container that contains all the previously defined gates.
        '''
        return segment_container(self.awg_channels.keys(), self.marker_channels.keys(),
                                 self.virtual_channels, self.IQ_channels.values(),
                                 name=name, sample_rate=sample_rate)

    def mk_sequence(self,seq):
        '''
        seq: list of segment_container.
        '''
        seq_obj = sequencer(self.uploader)
        seq_obj.add_sequence(seq)
        seq_obj.metadata = {}
        for (i,pc) in enumerate(seq):
            md = pc.get_metadata()
            seq_obj.metadata[('pc%i'%i)] = md
        LOdict = {}
        for iq in self.IQ_channels.values():
            for vm in iq.qubit_channels:
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

        for channel in self.awg_channels:
            awg = self.awg_devices[channel.awg_name]
            awg.awg_flush(channel.channel_number)


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

        # copy all named channels from harware file to awg_channels
        for channel, attenuation in hardware.AWG_to_dac_conversion.items():
            self.awg_channels[channel].attenuation = attenuation

        sync = False
        for channel in self.awg_channels.values():
            if channel.name not in hardware.AWG_to_dac_conversion:
                hardware.AWG_to_dac_conversion[channel.name] = channel.attenuation
                sync = True
        if sync:
            hardware.sync_data()


    def _check_uniqueness_of_channel_name(self, channel_name):
        if channel_name in self.awg_channels or channel_name in self.marker_channels:
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
    p.define_channel('B0','AWG1', 1)
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
    p.define_marker('M1','AWG4', 3, setup_ns=15, hold_ns=15)
    p.define_marker('M2','AWG4', 4, setup_ns=15, hold_ns=15)


    # format : channel name with delay in ns (can be posive/negative)
    p.add_channel_delay('I_MW',50)
    p.add_channel_delay('Q_MW',50)
#    p.add_channel_delay('M1',20)
#    p.add_channel_delay('M2',-25)

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
    IQ_chan_set_1.add_marker("M1")
    IQ_chan_set_1.add_marker("M2")
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

