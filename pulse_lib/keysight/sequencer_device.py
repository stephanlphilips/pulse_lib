from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np

@dataclass
class SequencerInfo:
    module_name: str
    sequencer_index: int
    channel_name: str # awg_channel or qubit_channel
    phases: List[float]
    gain_correction: List[float]


# TODO @@@ retrieve sequencer configuration from M3202A_QS object
@dataclass
class SequencerDevice:
    awg: object
    name: str
    iq_channels: List[object] = field(default_factory=list) # iq channel objects
    sequencers: List[SequencerInfo] = field(default_factory=list)

    def __post_init__(self):
        self.iq_channels = [None]*2
        self.sequencers = [None]*12

    def add_iq_channel(self, IQ_channel, channel_numbers):
        if not channel_numbers in [[1,2], [2,1], [3,4], [4,3]]:
            raise Exception(f'sequencer requires channels 1/2 or 3/4. ({IQ_channel.IQ_out_channels})')
        index = 1 if 3 in channel_numbers else 0
        if self.iq_channels[index] is not None:
            raise Exception(f'sequencer can only have 1 IQ channel definition. ({IQ_channel.IQ_out_channels})')
        self.iq_channels[index] = IQ_channel
        sequencer_numbers = [1,2,5,6,9,10] if index == 0 else [3,4,7,8,11,12]

        # names of output channels for modulators A and B
        out_channels = list(IQ_channel.IQ_out_channels)
        if channel_numbers[0] > channel_numbers[1]:
            out_channels.reverse()
        IQ_comps = ''.join(out_channel.IQ_comp for out_channel in out_channels)
        if IQ_comps not in ['IQ','QI']:
            raise Exception(f'Expected I and Q channel, but found {IQ_comps}')

        phases = [self._get_phase(out_channel) for out_channel in out_channels]

        sequencers = []
        for i, qubit_channel in enumerate(IQ_channel.qubit_channels):
            seq_num = sequencer_numbers[i]
            qubit_phases = phases.copy()
            if qubit_channel.correction_gain is None:
                gain_correction = [1.0, 1.0]
            else:
                if IQ_comps == 'IQ':
                    gain_correction = qubit_channel.correction_gain
                else:
                    gain_correction = list(reversed(qubit_channel.correction_gain))
            if qubit_channel.correction_phase is not None:
                if IQ_comps == 'IQ':
                    qubit_phases[1] += qubit_channel.correction_phase*180/np.pi
                else:
                    qubit_phases[0] += qubit_channel.correction_phase*180/np.pi

            #print(f'{qubit_channel.channel_name} {IQ_comps} {qubit_phases}')

            sequencer = SequencerInfo(self.name, seq_num,
                                      qubit_channel.channel_name, qubit_phases, gain_correction)
            self.sequencers[seq_num-1] = sequencer
            sequencers.append(sequencer)

             # @@@ sequencers could be mapped differently for memory size.
            seq = self.awg.get_sequencer(seq_num)

            # TODO @@@ cleanup M3202A_QS interface
            seq.configure_oscillators(0.0, qubit_phases[0], qubit_phases[1])
            max_gain = max(gain_correction)
            seq._gainA = gain_correction[0]/max_gain
            seq._gainB = gain_correction[1]/max_gain
            seq._init_lo()
        return sequencers

    def _get_phase(self, iq_out_channel):
        I_or_Q = iq_out_channel.IQ_comp
        image = iq_out_channel.image
        phase_shift = 0
        if I_or_Q == 'I':
            phase_shift += 90
        if image == '-':
            phase_shift += 180
        return phase_shift

#    def add_bb_channel(self, channel_number, channel_name):
#        index = 1 if channel_number in [3,4] else 0
#        if self.iq_channels[index] is not None:
#            raise Exception(f'sequencer cannot combine IQ and BB channels on same output')
#        if self.sequencers[channel_number] is not None:
#            raise Exception(f'sequencer cannot have multiple BB channels on same output')
#        phases = [90,0] if channel_number % 2 == 1 else [0,90]
#        sequencer = SequencerInfo(self.name, channel_number, channel_name, None, phases, [channel_number])
#        self.sequencers[channel_number] = sequencer
#        return sequencer

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
        iq_awg_channels = [awg_channels[iq_channel_info.awg_channel_name] for iq_channel_info in iq_pair]
        awg_names = [awg_channel.awg_name for awg_channel in iq_awg_channels]
        if not any(awg_name in obj.sequencer_devices for awg_name in awg_names):
            continue

        if awg_names[0] != awg_names[1]:
            raise Exception(f'IQ channels should be on 1 awg: {iq_pair}')

        obj.sequencer_out_channels += [awg_channel.name for awg_channel in iq_awg_channels]

        seq_device = obj.sequencer_devices[awg_names[0]]
        channel_numbers = [awg_channel.channel_number for awg_channel in iq_awg_channels]
        qubit_sequencers = seq_device.add_iq_channel(IQ_channel, channel_numbers)
        for iseq,seq in enumerate(qubit_sequencers):
            obj.sequencer_channels[seq.channel_name] = seq

# @@@ commented out, because base band channels do not yet work.
#    for awg_channel in awg_channels.values():
#        if (awg_channel.awg_name in obj.sequencer_devices
#            and awg_channel.name not in obj.sequencer_out_channels):
#            seq_device = obj.sequencer_devices[awg_channel.awg_name]
#            bb_seq = seq_device.add_bb_channel(awg_channel.channel_number, awg_channel.name)
#
#            obj.sequencer_channels[bb_seq.channel_name] = bb_seq
#            obj.sequencer_out_channels += [bb_seq.channel_name]

#    for dev in obj.sequencer_devices.values():
#        awg = dev.awg
#        for i,seq_info in enumerate(dev.sequencers):
#            if seq_info is None:
#                continue
#            seq = awg.get_sequencer(i+1)
#            if seq_info.frequency is None:
#                # sequencer output is A for channels 1 and 3 and B for 2 and 4
#                output = 'BA'[seq_info.channel_numbers[0] % 2]
#                seq.set_baseband(output)
#            else:
#                seq.configure_oscillators(seq_info.frequency, seq_info.phases[0], seq_info.phases[1])
#
#
