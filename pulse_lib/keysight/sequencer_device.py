import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class SequencerInfo:
    module_name: str
    sequencer_index: int
    channel_name: str # awg_channel or qubit_channel
    frequency: float
    phases: List[float]

# TODO @@@ split into generic Sequencer and implementation specific.
@dataclass
class SequencerDevice:
    awg: object
    name: str
    iq_channels: List[object] = field(default_factory=list) # iq channel objects
    sequencers: List[SequencerInfo] = field(default_factory=list)

    def __post_init__(self):
        self.iq_channels = [None]*2
        self.sequencers = [None]*8

    def add_iq_channel(self, IQ_channel, channel_numbers):
        if not channel_numbers in [[1,2], [2,1], [3,4], [4,3]]:
            raise Exception(f'sequencer requires channels 1/2 or 3/4. ({IQ_channel.channel_map})')
        index = 1 if 3 in channel_numbers else 0
        if self.iq_channels[index] is not None:
            raise Exception(f'sequencer can only have 1 IQ channel definition. ({IQ_channel.channel_map})')

        iq_pair = IQ_channel.IQ_out_channels
        phases = [self._get_phase(iq_out_channel) for iq_out_channel in iq_pair]

        self.iq_channels[index] = IQ_channel
        sequencer_numbers = [1,2,5,6] if index == 0 else [3,4,7,8]
        sequencers = []
        for i, qubit_channel in enumerate(IQ_channel.qubit_channels):
            if channel_numbers[1] > channel_numbers[0]:
                phases.reverse()
            f = qubit_channel.reference_frequency - IQ_channel.LO
            sequencer = SequencerInfo(self.name, sequencer_numbers[i],
                                      qubit_channel.channel_name, f, phases)
            self.sequencers[sequencer_numbers[i]-1] = sequencer
            sequencers.append(sequencer)
        return sequencers

    def _get_phase(self, iq_out_channel):
        I_or_Q = iq_out_channel.IQ_comp
        image = iq_out_channel.image
        phase_shift = 0
        if I_or_Q == 'I':
            phase_shift += np.pi/2
        if image == '-':
            phase_shift += np.pi
        return phase_shift

    def add_bb_channel(self, channel_number, channel_name):
        index = 1 if channel_number in [3,4] else 0
        if self.iq_channels[index] is not None:
            raise Exception(f'sequencer cannot combine IQ and BB channels on same output')
        if self.sequencers[channel_number] is not None:
            raise Exception(f'sequencer cannot have multiple BB channels on same output')
        phases = [90,0] if channel_number % 2 == 1 else [0,90]
        sequencer = SequencerInfo(self.name, channel_number, channel_name, 0, phases)
        self.sequencers[channel_number] = sequencer
        return sequencer

def add_sequencers(obj, AWGs, awg_channels, IQ_channels):
    obj.sequencer_devices = {}
    obj.sequencer_channels:Dict[str,SequencerInfo] = {}
    # collect output channels. They should not be rendered to full waveforms
    obj.sequencer_out_channels:List[str] = []

    for awg_name, awg in AWGs.items():
        if hasattr(awg, 'get_sequencer'):
            obj.sequencer_devices[awg_name] = SequencerDevice(awg, awg_name)

    for IQ_channel in IQ_channels.values():
        iq_pair = IQ_channel.IQ_out_channels
        if len(iq_pair) != 2:
            raise Exception(f'IQ-channel should have 2 awg channels '
                            f'({iq_pair})')
        iq_out_channels = [awg_channels[iq_channel_info.awg_channel_name] for iq_channel_info in iq_pair]
        awg_names = [awg_channel.awg_name for awg_channel in iq_out_channels]
        if not any(awg_name in obj.sequencer_devices for awg_name in awg_names):
            continue

        if awg_names[0] != awg_names[1]:
            raise Exception(f'IQ channels should be on 1 awg: {iq_pair}')

        obj.sequencer_out_channels += [iq_channel_info.awg_channel_name for iq_channel_info in iq_pair]

        seq_device = obj.sequencer_devices[awg_names[0]]
        channel_numbers = [awg_channel.channel_number for awg_channel in iq_out_channels]
        qubit_sequencers = seq_device.add_iq_channel(IQ_channel, channel_numbers)
        for iseq,seq in enumerate(qubit_sequencers):
            obj.sequencer_channels[seq.channel_name] = seq

    for awg_channel in awg_channels.values():
        if (awg_channel.awg_name in obj.sequencer_devices
            and awg_channel.name not in obj.sequencer_out_channels):
            seq_device = obj.sequencer_devices[awg_channel.awg_name]
            bb_seq = seq_device.add_bb_channel(awg_channel.channel_number, awg_channel.awg_name)

            obj.sequencer_channels[bb_seq.channel_name] = bb_seq
            obj.sequencer_out_channels += [bb_seq.channel_name]

    for dev in obj.sequencer_devices.values():
        awg = dev.awg
        for i,seq_info in enumerate(dev.sequencers):
            if seq_info is None:
                continue
            seq = awg.get_sequencer(i+1)
            if seq_info.frequency == 0:
                seq.set_baseband(True)
            else:
                seq.configure_oscillators(seq_info.frequency, seq_info.phases[0], seq_info.phases[1])


