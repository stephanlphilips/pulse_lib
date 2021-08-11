'''
For now just create a new segment for every added template.

Ideas for reuse:
    * subsequences
    * atoms
    * cache based on template IDs

'''
from .segments.conditional_segment import conditional_segment

class builder_policy:
    TinySegments = 0
    ''' Start a new segment when reset_time() is called. '''
    BigSegments = 1
    ''' Build 1 big segment '''
    RootSegments = 2
    '''
    Start a new segment when a template is added to the sequence builder.
    Nested templates do not start a new segment.
    '''


class sequence_builder:

    def __init__(self, pulselib, policy=builder_policy.RootSegments):
        self._pulselib = pulselib
        self._policy = policy
        self._segments = []
        self._segment = None
        self.n_rep = 2000
        self._nested_templates = 0

    def _mk_segment(self):
        segment = self._pulselib.mk_segment()
        self._segments.append(segment)
        return segment

    def _get_segment(self):
        if self._segment is None:
            self._segment = self._mk_segment()
        return self._segment

    def add(self, template, reset=True, **kwargs):
        self._nested_templates += 1
        template.build(self, reset=False, **kwargs)
        self._nested_templates -= 1

        if reset:
            self.reset_time()

    def add_segment(self, *segments):
        '''
        add raw segments to the sequence builder

        Args:
            *segments (segment_container) : segments that you want to add
        '''
        self._segments += segments


    ## mimic segment_container

    def __getitem__(self, index):
        segment = self._get_segment()
        return segment[index]

    def __getattr__(self, name):
        segment = self._get_segment()
        return segment[name]

    def add_measurement_expression(self, expression=None, name=None, accept_if=None):
        segment = self._get_segment()
        segment.add_measurement_expression(expression, accept_if=accept_if, name=name)

    def reset_time(self):
        if self._policy == builder_policy.TinySegments:
            # force start of new segment
            self._segment = None

        if self._policy == builder_policy.RootSegments and self._nested_templates == 0:
            # force start of new segment
            self._segment = None

        if self._segment is not None:
            self._segment.reset_time()

    ## sequence builder extra's

    def add_conditional(self, condition, templates):
        ### for now we do not support replacement arguments with conditionals
        kwargs = dict()

        if len(templates) not in [2,4]:
            raise Exception('conditional gates must have 2 or 4 options')
        self.reset_time()
        self._segment = None

        # create conditional segment
        n_branches = len(templates)
        branches = [self._pulselib.mk_segment() for i in range(n_branches)]
        cond_seg = conditional_segment(condition, branches)
        for i,template in enumerate(templates):
            if template is None:
                # empty
                continue
            segment = cond_seg.branches[i]
            # note: building on segment, not on sequence_builder !!
            template.build(segment, reset=False, **kwargs)
        print('adding', cond_seg)
        self._segments.append(cond_seg)

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

    def append(self, other):
        # close active segments in both sequences
        self._segment = None
        other._segment = None
        self._segments += other._segments

    def forge(self):
        return self._pulselib.mk_sequence(self._segments)
