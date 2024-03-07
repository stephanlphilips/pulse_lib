from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class SequencerInfo:
    module_name: str
    sequencer_index: int
    channel_name: str # awg_channel or qubit_channel
    phases: List[float]
    gain_correction: List[float]
    sequencer_offset: int


# TODO retrieve sequencer configuration from M3202A_QS object
# TODO dynamic assignment of sequencers for optimal use of memory size.

class SequencerDevice:
    def __init__(self, awg):
        self.awg = awg
        self.unassigned_sequencers = {
            '1+2': [1,2,5,6,9,10],
            '3+4': [3,4,7,8,11,12],
            }

        self.sequencer_offset: int = 10
        if hasattr(self.awg, 'get_sequencer_offset'):
            self.sequencer_offset = self.awg.get_sequencer_offset()

    def _get_sequencer_group(self, channel_number: int):
        if channel_number in [1, 2]:
            return '1+2'
        if channel_number in [3, 4]:
            return '3+4'

    def _get_sequencer(self, channel_numbers: List[int]):
        group = self._get_sequencer_group(channel_numbers[0])
        for ch_num in channel_numbers[1:]:
            if self._get_sequencer_group(ch_num) != group:
                raise Exception(
                    f"Sequencer cannot be configured on channels {channel_numbers} of awg {self.awg.name}. "
                    "IQ pairs must be configured on channels 1+2 or 3+4."
                    )
        try:
            sequencer = self.unassigned_sequencers[group].pop(0)
        except IndexError:
            raise Exception(f"Not enough sequencers for channels {group} of awg {self.awg.name}") from None
        return sequencer

    def add_iq_channel(self, IQ_channel, channel_numbers):
        # channel names for output channels of NCOs A (ch 1 or 3) and B (ch 2 or 4)
        out_channels = list(IQ_channel.IQ_out_channels)
        if channel_numbers[0] > channel_numbers[1]:
            out_channels.reverse()
        IQ_comps = ''.join(out_channel.IQ_comp for out_channel in out_channels)
        if IQ_comps not in ['IQ','QI']:
            raise Exception(f'Expected I and Q channel, but found {IQ_comps}')

        phases = [self._get_phase(out_channel) for out_channel in out_channels]

        assigned_sequencers = {}
        for i, qubit_channel in enumerate(IQ_channel.qubit_channels):
            seq_num = self._get_sequencer(channel_numbers)
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

            sequencer = SequencerInfo(self.awg.name, seq_num,
                                      qubit_channel.channel_name, qubit_phases,
                                      gain_correction, self.sequencer_offset)
            assigned_sequencers[qubit_channel.channel_name] = sequencer

            max_gain = max(gain_correction)
            gainA = gain_correction[0]/max_gain
            gainB = gain_correction[1]/max_gain
            self._configure_ncos(seq_num, gainA, gainB, qubit_phases[0], qubit_phases[1])
        return assigned_sequencers

    def add_drive_channel(self, IQ_channel, channel_number):
        if channel_number % 2 == 1:
            # output on channel 1 or 3, i.e. NCO A
            gains = [1.0, 0.0]
        else:
            # output on channel 2 or 4, i.e. NCO B
            gains = [0.0, 1.0]
        phases = [90.0, 90.0]

        assigned_sequencers = {}
        for qubit_channel in IQ_channel.qubit_channels:
            seq_num = self._get_sequencer([channel_number])

            sequencer = SequencerInfo(self.awg.name, seq_num,
                                      qubit_channel.channel_name, phases,
                                      gains, self.sequencer_offset)
            assigned_sequencers[qubit_channel.channel_name] = sequencer

            self._configure_ncos(seq_num, gains[0], gains[1], phases[0], phases[1])
        return assigned_sequencers

    def _configure_ncos(self, seq_num, gainA, gainB, phaseA, phaseB):
        seq = self.awg.get_sequencer(seq_num)

        # TODO @@@ imporove M3202A_QS interface
        seq.configure_oscillators(0.0, phaseA, phaseB)
        seq._gainA = gainA
        seq._gainB = gainB
        seq._init_lo()

    def _get_phase(self, iq_out_channel):
        I_or_Q = iq_out_channel.IQ_comp
        image = iq_out_channel.image
        phase_shift = 0
        if I_or_Q == 'I':
            # NOTE: FPGA uses sine table. Add 90 degrees of I and 0 for Q.
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
