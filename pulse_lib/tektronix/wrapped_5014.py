import time
from typing import List, Dict, Optional
import logging
import numpy as np
from dataclasses import dataclass

from pulse_lib.configuration.physical_channels import awg_channel, marker_channel
from qcodes.instrument_drivers.tektronix.AWG5014 import Tektronix_AWG5014

@dataclass
class ChannelData:
    name: str
    wvf: Optional[np.ndarray] = None
    m1: Optional[np.ndarray] = None
    m2: Optional[np.ndarray] = None

class Wrapped5014:
    def __init__(self,
                 awg: Tektronix_AWG5014,
                 awg_channels: Dict[str, awg_channel],
                 marker_channels: Dict[str, marker_channel]):
        '''
        '''
        self.__awg = awg
        self._awg_channels = awg_channels
        self._marker_channels = marker_channels

    def _get_channel_info(self, channel_name):
        if channel_name in self._awg_channels:
            channel_info = self._awg_channels[channel_name]
            if channel_info.awg_name == self.__awg.name:
                return channel_info
        if channel_name in self._marker_channels:
            channel_info = self._marker_channels[channel_name]
            if channel_info.module_name == self.__awg.name:
                return channel_info
        return None

    def upload_waveforms(self, sequences:Dict[str, np.ndarray]):
        logging.debug('upload waveforms...')

        upload_list:Dict[int, ChannelData] = dict()

        for channel_name, waveform_data in sequences.items():
            channel_info = self._get_channel_info(channel_name)
            if not channel_info:
                continue
            channel_number = channel_info.channel_number
            marker_number = None
            if isinstance(channel_number, tuple):
                marker_number = channel_number[1]
                channel_number = channel_number[0]

            if channel_number not in upload_list:
                l = len(waveform_data)
                channel_data = ChannelData(f'ch_{channel_number}')
                channel_data.wvf = np.zeros(l, dtype=np.uint16)
                channel_data.m1 = np.zeros(l, dtype=np.uint16)
                channel_data.m2 = np.zeros(l, dtype=np.uint16)
                upload_list[channel_number] = channel_data
            else:
                channel_data = upload_list[channel_number]

            if not marker_number:
                channel_data.wvf = waveform_data
            elif marker_number == 1:
                channel_data.m1 = waveform_data
            else:
                channel_data.m2 = waveform_data

        self.delete_sequence()
#        self.delete_waveforms()
        if len(upload_list) == 0:
            logging.info('no data for AWG')
            return

        channel_cfg = self.generate_cfg()
        self.pack_and_load_awg_file(upload_list, channel_cfg)

        logging.debug('set sequences')
        channels = list(upload_list.keys())
        sequences = [[data.name] for data in upload_list.values()]
        self._set_sequence(channels, sequences)


    def pack_and_load_awg_file(self, upload_list, channel_cfg):
        start = time.perf_counter()
        logging.debug('generate file')
        packed_waveforms = dict()
        for data in upload_list.values():
#            logging.debug(f'pack {data.name}: {len(data.wvf)} Sa')
            start = time.perf_counter()
#            logging.debug(f'{data.name}: {len(data.wvf)}, '
#                          f'{np.min(data.wvf)} - {np.max(data.wvf)}, '
#                          )
            package = self.__awg._pack_waveform(data.wvf, data.m1, data.m2)
            packed_waveforms[data.name] = package
#            logging.info(f'packed ({(time.perf_counter()-start)*1000:5.1f})ms')

        file_name = 'default.awg'
        self.__awg.visa_handle.write('MMEMory:CDIRectory "C:\\Users\\OEM\\Documents"')
        awg_file = self.__awg._generate_awg_file(packed_waveforms, np.array([]), [], [], [], [], channel_cfg)
        logging.debug(f'generated ({(time.perf_counter()-start)*1000:5.1f})ms')
        logging.debug('send file')
        self.__awg.send_awg_file(file_name, awg_file)
        start = time.perf_counter()
        logging.debug('load file')
        current_dir = self.__awg.visa_handle.query('MMEMory:CDIRectory?')
        current_dir = current_dir.replace('"', '')
        current_dir = current_dir.replace('\n', '\\')
        self.__awg.load_awg_file(f'{current_dir}{file_name}')
        logging.debug(f'loaded file ({(time.perf_counter()-start)*1000:5.1f})ms')


    def generate_cfg(self):
        awg = self.__awg
        amplitudes = [0]*4
        offsets = [0]*4
        marker_lows = [[0, 0] for i in range(4)]
        marker_highs = [[0, 0] for i in range(4)]

        for channel in self._awg_channels.values():
            if channel.awg_name == awg.name:
                if channel.amplitude > 4.5 or channel.amplitude < 0.02:
                    raise ValueError(f'amplitude ({channel.amplitude}) out of range [0.02, 4.5] V')
                amplitudes[channel.channel_number-1] = channel.amplitude

        for channel in self._marker_channels.values():
            if channel.module_name == awg.name:
                if channel.amplitude > 2.7 or channel.amplitude < -0.9:
                    raise ValueError(f'marker amplitude ({channel.amplitude}) out of range [-0.9, 1.7] V')
                if isinstance(channel.channel_number, tuple):
                    channel_number = channel.channel_number[0]
                    marker_number = channel.channel_number[1]
                    if channel.invert:
                        marker_lows[channel_number-1][marker_number-1] = channel.amplitude
                    else:
                        marker_highs[channel_number-1][marker_number-1] = channel.amplitude
                else:
                    amplitudes[channel.channel_number-1] = channel.amplitude

        # the return value of the parameter is different from what goes
        # into the .awg file, so we translate it
        filtertrans = {20e6: 1, 100e6: 3, 9.9e37: 10,
                       'INF': 10, 'INFinity': 10,
                       float('inf'): 10, None: None}
        filters = [filtertrans[awg.ch1_filter.get_latest()],
                   filtertrans[awg.ch2_filter.get_latest()],
                   filtertrans[awg.ch3_filter.get_latest()],
                   filtertrans[awg.ch4_filter.get_latest()]]

        channel_cfg = {}
        for i in range(4):
            ch = str(i+1)
            channel_cfg['ANALOG_METHOD_'+ch] = 1
#            channel_cfg['CHANNEL_STATE_'+ch] = 1
            channel_cfg['ANALOG_FILTER_'+ch] = filters[i]
            channel_cfg['ANALOG_AMPLITUDE_'+ch] = amplitudes[i] / 1000
            channel_cfg['ANALOG_OFFSET_'+ch] = offsets[i] / 1000
            channel_cfg['MARKER1_METHOD_'+ch] = 2
            channel_cfg['MARKER1_LOW_'+ch] = marker_lows[i][0] / 1000
            channel_cfg['MARKER1_HIGH_'+ch] = marker_highs[i][0] / 1000
            channel_cfg['MARKER2_METHOD_'+ch] = 2
            channel_cfg['MARKER2_LOW_'+ch] = marker_lows[i][1] / 1000
            channel_cfg['MARKER2_HIGH_'+ch] = marker_highs[i][1] / 1000

        return channel_cfg

    def _set_sequence(self, channels: List[int], sequence: List[List[str]]) -> None:
        """ Sets the sequence on the AWG using the user defined waveforms.

        Args:
            channels: A list with channel numbers that should output the waveform on the AWG.
            sequence: A list containing lists with the waveform names for each channel.
                      The outer list determines the number of rows the sequences has.
        """
        n_elements = 1
        self.__awg.sequence_length(n_elements)
        element_no = 1
        for channel in range(1,5):
            if channel in channels:
                ch_index = channels.index(channel)
                wave_name = sequence[ch_index][0]
                self.__awg.set_sqel_waveform(wave_name, channel, element_no)
            else:
                self.__awg.set_sqel_waveform("", channel, element_no)
        self.__awg.set_sqel_goto_state(element_no, 1)


    def delete_waveforms(self) -> None:
        """ Clears the user defined waveform list from the AWG."""
        self.__awg.delete_all_waveforms_from_list()

    def delete_sequence(self) -> None:
        """ Clears the sequence from the AWG."""
        self.__awg.sequence_length(0)

    def set_sample_rate(self, sample_rate):
        return self.__awg.clock_freq(int(sample_rate))

    def trigger(self):
        self.__awg.force_trigger()

