
import logging
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import matplotlib.pyplot as pt

from qcodes.instrument.base import Instrument


# mock for M3202A / SD_AWG_Async
class MockTektronix5014(Instrument):

    def __init__(self, name):
        super().__init__(name)
        self.visa_handle = VisaHandlerMock(self)
        self.settings = {}
        self.channel_settings = {ch:{} for ch in [1,2,3,4]}
        self.waveforms = {}
        self.waveforms_lengths = {}
        self.sequence = []
        self.state = 'Stopped'
        self.all_channels_off()

    def get_idn(self):
        return dict(vendor='Pulselib', model=type(self).__name__, serial='', firmware='')

    def trigger_source(self, value):
        self.settings['trigger_source'] = value

    def trigger_impedance(self, value):
        self.settings['trigger_impedance'] = value

    def trigger_level(self, value):
        self.settings['trigger_level'] = value

    def trigger_slope(self, value):
        self.settings['trigger_slope'] = value

    def clock_freq(self, value):
        self.settings['clock_freq'] = value

    def set(self, name, value):
        if name.startswith('ch'):
            channel = int(name[2])
            self.channel_settings[channel][name[4:]] = value
        else:
            self.settings[name] = value

    def delete_all_waveforms_from_list(self):
        self.waveforms = {}

    def write(self, cmd):
        if cmd.startswith('WLISt:WAVeform:DEL '):
            name = cmd.split('"')[1]
            if name in self.waveforms:
                del self.waveforms[name]
        elif cmd.startswith('WLISt:WAVeform:NEW '):
            name = cmd.split('"')[1]
            length = int(cmd.split(',')[1])
            self.waveforms_lengths[name] = length

    def sequence_length(self, length):
        self.sequence = [SequenceElement() for _ in range(length)]

    def set_sqel_waveform(self, wave_name, channel, element_no):
        self.sequence[element_no-1].wave_names[channel] = wave_name

    def set_sqel_goto_state(self, element_no, dest):
        self.sequence[element_no-1].goto = dest

    def set_sqel_trigger_wait(self, element_no):
        self.sequence[element_no-1].wait = True

    def set_sqel_loopcnt(self, n_repetitions, element_no):
        self.sequence[element_no-1].loop_cnt = n_repetitions

    def set_sqel_loopcnt_to_inf(self, element_no):
        self.sequence[element_no-1].loop_cnt = 10**6

    def run_mode(self, mode):
        self.settings['mode'] = mode

    def run(self):
        self.state = 'Running'

    def stop(self):
        self.state = 'Stopped'

    def force_trigger(self):
        self.state = 'Running'

    def all_channels_off(self):
        for ch in [1,2,3,4]:
            settings = self.channel_settings[ch]
            settings['state'] = 0

    def plot(self):
        for ch in [1,2,3,4]:
            settings = self.channel_settings[ch]
            print(self.name, ch, settings)

            if settings['state'] == 0:
                continue
            amp = settings.get('amp', 0.0)
            offset = settings.get('offset', 0.0)
            wave_raw = self.waveforms[self.sequence[0].wave_names[ch]]
            wave_data = (wave_raw & 0x3FFF) # ((wave_raw & 0x3FFF) << 2).astype(np.int16)
            wave = offset + amp * (wave_data/2**13 - 1)
            pt.plot(wave, label=f'ch{ch}')
            if settings.get('m1_high',False):
                marker_1 = (wave_raw & 0x4000) >> 14
                pt.plot(marker_1, ':', label=f'M{ch}.1')
            if settings.get('m2_high',False):
                marker_2 = (wave_raw & 0x8000) >> 15
                pt.plot(marker_2, ':', label=f'M{ch}.2')

class VisaHandlerMock:
    def __init__(self, parent):
        self.parent = parent

    def write_raw(self, msg):
        cmd_end = msg.index(ord(','))
        cmd = str(msg[:cmd_end])
        name = cmd.split('"')[1]
        data_length_len = int(chr(msg[cmd_end+2]))
        data_start = cmd_end+3+data_length_len
        # data_length = int(str(msg[cmd_end+3:data_start], encoding='utf-8'))
        data = np.frombuffer(msg[data_start:], dtype='<u2')
        self.parent.waveforms[name] = data

@dataclass
class SequenceElement:
    wave_names: Dict[int,str] = field(default_factory=dict)
    goto: int = 0
    wait: bool = False
    loop_cnt: int = 1

