import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Any, Union
from collections.abc import Iterable

from pulse_lib.segments.conditional_segment import conditional_segment
from pulse_lib.segments.segment_base import segment_base
from pulse_lib.segments.segment_acquisition import segment_acquisition
from pulse_lib.segments.segment_IQ import segment_IQ
from pulse_lib.segments.segment_pulse import segment_pulse
from pulse_lib.segments.segment_markers import segment_marker

from pulse_lib.segments.utility.measurement_ref import MeasurementRef

err_acqs = None

class ConditionalAcquisition:
    class _AcquisitionData:
        def __init__(self, data):
            self.data = data

        def get_data(self):
            return self.data

    def __init__(self, seg_channels:List[segment_acquisition]):
        self.seg_channels = seg_channels

    def _get_data_all_at(self, index):
        ch_data = [
                ch._get_data_all_at(index).get_data()
                for ch in self.seg_channels]

        # acquisitions in conditional branch must all be equal
        for data in ch_data[1:]:
            if data != ch_data[0]:
                global err_acqs
                err_acqs = ch_data
                raise Exception('Non-sequenced channels must be equal for all branches.')

        return ConditionalAcquisition._AcquisitionData(ch_data[0])


class ConditionalMarker:
    class _MarkerData:
        def __init__(self):
            self.my_marker_data = []

    def __init__(self, seg_channels:List[segment_marker]):
        self.seg_channels = seg_channels

    def _get_data_all_at(self, index):
        # NOTE: marker are not conditional in hardware. Merge markers of all branches.
        # merge markers of all branches
        data = ConditionalMarker._MarkerData()
        for ch in self.seg_channels:
            data.my_marker_data += ch._get_data_all_at(index).my_marker_data

        return data


err_wvfs = None

class ConditionalWaveform:
    def __init__(self, seg_channels:List[segment_base]):
        self.seg_channels = seg_channels

    def get_segment(self, index=[0], sample_rate=1e9, ref_channel_states=None):
        '''
        Returns waveform for not-sequenced channel.
        All branches must be equals
        '''
        wvfs = [seg_ch.get_segment(index, sample_rate, ref_channel_states) for seg_ch in self.seg_channels]
        for wvf in wvfs[1:]:
            if not np.array_equal(wvf, wvfs[0]):
                global err_wvfs
                err_wvfs = wvfs
                raise Exception('Non-sequenced channels must be equal for all branches.')

        return wvfs[0]


def get_acquisition_names(conditional:conditional_segment):
    condition = conditional.condition
    refs = condition if isinstance(condition, Iterable) else [condition]
    acquisition_names = set()
    for ref in refs:
        acquisition_names.update(ref.keys)

    acquisition_names = list(acquisition_names)
    logging.info(f'acquisitions: {acquisition_names}')
    return acquisition_names


class QsConditionalChannel:
    def __init__(self, seg_channels:List[segment_base], conditional:conditional_segment):
        self.seg_channels = seg_channels
        self.conditional = conditional
        self.n_branches = len(seg_channels)
        condition = conditional.condition
        refs = condition if isinstance(condition, Iterable) else [condition]

        # Lookup acquistions for condition
        self.acquisition_names = self.get_acquisition_names(refs)

        self.order = self.get_branch_order(refs)

    def get_acquisition_names(self, refs:List[MeasurementRef]):
        acquisition_names = set()
        for ref in refs:
            acquisition_names.update(ref.keys)

        acquisition_names = list(acquisition_names)
        logging.info(f'acquisitions: {acquisition_names}')
        return acquisition_names

    def get_branch_order(self, refs):
        # Assumes max 4 branches

        # special case: 1 measurement, 2 acquisitions (and 2 options) => expand to 4 options
        # this is handled gracefully by this code:
        # 1 measurement: result contains only 0 and 1
        # 2 measurements: result contains 0,1,2,3

        # 0, 1, 2, 3 in binary representation on 2 acquisitions
        all_values = np.array([[0,1,0,1],[0,0,1,1]])
        values = {key:all_values[i] for i,key in enumerate(self.acquisition_names)}

        order = np.zeros(4, dtype=np.int)
        for ref in refs[::-1]:
            order = 2 * order + ref.evaluate(values)
        logging.info(f'reordered branches: {order}')
        return order


class QsConditionalMW(QsConditionalChannel):
    # sequencer: find common offset per sequencer, generate waveforms
    # when uploading, generate extra entries in index table for conditional waveforms
    # upload waveforms as usual. Store start/stop
    # generate index table entries for conditionals. Start at 248-251.

    # ASSUME: only 1 pulse per sequencer.
    # Phase shift can shift within segment between pulses.
    # Phase shift can be combined in pre-phase or post-phase

    # sequencer:
    # combine branch to pre, mw_pulse, post.
    # set MW pulse
    @dataclass
    class BranchPulse:
        mw_pulse: any = None
        prephase: float = 0.0
        postphase: float = 0.0

    @dataclass
    class ConditionalInstruction:
        start: float
        end: float
        pulses: List['QsConditionalMW.BranchPulse'] = field(default_factory=list)

    def __init__(self, seg_channels:List[segment_IQ],
                 conditional:conditional_segment, index):
        super().__init__(seg_channels, conditional)
        self.index = index
        self.conditional_instructions:List['QsConditionalMW.ConditionalInstruction'] = []
        self.combine_branches()

    def add_pulse(self, pulse, ibranch):
        start = pulse.start
        end = pulse.stop
        for instr in self.conditional_instructions:
            # pulse overlaps with instruction
            if start < instr.end and end > instr.start:
                instr.start = min(start, instr.start)
                instr.end = max(end, instr.end)
                if instr.pulses[ibranch] is not None:
                    raise Exception(f'overlapping pulses in conditional (branch:{ibranch})')
                instr.pulses[ibranch] = QsConditionalMW.BranchPulse(pulse)
                return

        # add new instruction
        instr = QsConditionalMW.ConditionalInstruction(start, end, [None]*self.n_branches)
        instr.pulses[ibranch] = QsConditionalMW.BranchPulse(pulse)
        self.conditional_instructions.append(instr)

    def add_phase(self, phase_shift, ibranch):
        for instr in self.conditional_instructions:
            if instr.end > phase_shift.time:
                logging.debug(f'Instr: {instr} Phase: {phase_shift}')
                pulse = instr.pulses[ibranch]
                # try to add phase shift to existing pulse
                if not pulse:
                    instr.pulses[ibranch] = QsConditionalMW.BranchPulse(prephase=phase_shift.phase_shift)
                elif not pulse.mw_pulse or pulse.mw_pulse.start > phase_shift.time:
                    pulse.prephase += phase_shift.phase_shift
                elif pulse.mw_pulse.stop <= phase_shift.time:
                    pulse.postphase += phase_shift.phase_shift
                else:
                    raise Exception(f'Phase overlaps with pulse. Branch:{ibranch} {phase_shift}, {pulse}')
                return

        # add new instruction
        start = phase_shift.time
        end = start + 2 # phase shift requires 2 ns
        instr = QsConditionalMW.ConditionalInstruction(start, end, [None]*self.n_branches)
        instr.pulses[ibranch] = QsConditionalMW.BranchPulse(prephase=phase_shift.phase_shift)
        self.conditional_instructions.append(instr)

    def combine_branches(self):
        # find time + duration of MW pulses
        for ibranch, branch in enumerate(self.seg_channels):
            pulse_data = branch._get_data_all_at(self.index).MW_pulse_data
            logging.debug(f'Adding MW pulses branch {ibranch} {pulse_data}')
            for pulse in pulse_data:
                self.add_pulse(pulse, ibranch)

        self.conditional_instructions.sort(key=lambda x:x.start)
        logging.debug(f'Conditional instructions: {self.conditional_instructions}')

        # add phase shifts to pulses, pre-phase of post-phase. Sum phase-shifts
        for ibranch, branch in enumerate(self.seg_channels):
            phase_data = branch._get_data_all_at(self.index).phase_shifts
            logging.debug(f'Adding phase shifts branch {ibranch} {phase_data}')
            for phase_shift in phase_data:
                if phase_shift.phase_shift != 0.0:
                    self.add_phase(phase_shift, ibranch)

        logging.debug(f'Conditional instructions: {self.conditional_instructions}')

        # check pulse overlaps.
        last_end = -1
        for instr in self.conditional_instructions:
            if instr.start < last_end:
                raise Exception(f'Overlapping conditional instructions')
            last_end = instr.end


def get_conditional_channel(conditional:conditional_segment, channel_name:str, index=None,
                            sequenced:bool=False):
    ## create Conditional channels
    seg_channels = [branch[channel_name] for branch in conditional.branches]

    if isinstance(seg_channels[0], segment_marker):
        return ConditionalMarker(seg_channels)
    if isinstance(seg_channels[0], segment_acquisition):
        return ConditionalAcquisition(seg_channels)

    if sequenced and isinstance(seg_channels[0], segment_IQ):
        return QsConditionalMW(seg_channels, conditional, index)

    if isinstance(seg_channels[0], segment_base):
        return ConditionalWaveform(seg_channels)

    raise Exception(f'Oops: {type(seg_channels[0])}')






