'''
For now just create a new segment for every added template.

Ideas for reuse:
    * subsequences
    * atoms
    * cache based on template IDs

'''
from dataclasses import dataclass, field
from typing import List


class builder_policy:
    TinySegments = 0
    BigSegments = 1

@dataclass
class conditional_segment:
    register: str
    n_options: int
    segments: List[any] = field(default_factory=list)

    def __post_init__(self):
        self.segments = [None]*self.n_options


class sequence_builder:

    def __init__(self, pulselib, policy=builder_policy.BigSegments):
        self._pulselib = pulselib
        self._policy = policy
        self._segments = []
        self._segment = None

    def _mk_segment(self):
        segment = self._pulselib.mk_segment()
        self._segments.append(segment)
        return segment

    def _get_segment(self):
        if self._segment is None:
            self._segment = self._mk_segment()
        return self._segment

    def add(self, template, reset=True, **kwargs):
        segment = self._get_segment()
        template.build(segment, reset=False, **kwargs) # @@@ internal reset_time should/could create new segment

        if reset:
            self.reset_time()

    def add_cond(self, register, *templates):
        ### for now we do not support replacement arguments with the templates
        kwargs = dict()

        if len(templates) not in [2,4]:
            raise Exception('conditional gates must have 2 or 4 options')
        self.reset_time()
        self._segment = None

        # create conditional segment
        cond_seg = conditional_segment(register, len(templates))
        for i,template in enumerate(templates):
            segment = cond_seg.segments[i]
            template.build(segment, reset=False, **kwargs)
        self._segments.append(cond_seg)

        # @@@ how to handle base band?
        #     Render all and compare? Compare before rendering?

        # @@@ handle conditional in mk_sequence => give them all equal length
        # @@@ handle conditional in qs_uploader


    def append(self, other):
        # close active segments in both sequences
        self._segment = None
        other._segment = None
        self._segments += other._segments

    def reset_time(self):
        if self._policy == builder_policy.TinySegments:
            # force start of new segment
            self._segment = None
        else:
            self._segment.reset_time()

    def wait(self, channels, t, amplitudes, reset_time=True):
        '''
        Adds a block to each of the specified channels.
        Args:
           t (float, loop_obj): duration of the block
           channels (List[str]): channels to apply the block to
           amplitudes (List[float, loop_obj]): amplitude per channel
           reset_time (bool): if True resets segment time after block
        '''
        segment = self._get_segment()
        for channel, amplitude in zip(channels, amplitudes):
            segment[channel].add_block(0, t, amplitude)

        if reset_time == True:
            self.reset_time()

    def add_block(self, channels, t, amplitudes, reset_time=True):
        '''
        Adds a block to each of the specified channels.
        Args:
           t (float, loop_obj): duration of the block
           channels (List[str]): channels to apply the block to
           amplitudes (List[float, loop_obj]): amplitude per channel
           reset_time (bool): if True resets segment time after block
        '''
        segment = self._get_segment()
        for channel, amplitude in zip(channels, amplitudes):
            segment[channel].add_block(0, t, amplitude)

        if reset_time == True:
            self.reset_time()

    def add_ramp(self, channels, t, start_amplitudes, stop_amplitudes,
                 reset_time=True):
        '''
        Adds a ramp to each of the specified channels.
        Args:
           t (float, loop_obj): duration of the block
           channels (List[str]): channels to apply the block to
           start_amplitudes (List[float, loop_obj]): start amplitude per channel
           stop_amplitudes (List[float, loop_obj]): stop amplitude per channel
           reset_time (bool): if True resets segment time after block
        '''
        segment = self._get_segment()
        for channel, start_amp, stop_amp in zip(channels, start_amplitudes, stop_amplitudes):
            segment.add_ramp_ss(0, t, start_amp, stop_amp)

        if reset_time == True:
            self.reset_time()

    def forge(self):
        return self._pulselib.mk_sequence(self._segments)
