
from pulse_lib.segments.data_classes.data_pulse import OffsetRamp, PhaseShift, custom_pulse_element
from pulse_lib.segments.data_classes.data_IQ import IQ_data_single
from pulse_lib.segments.conditional_segment import conditional_segment

def get_last_upload(pulse):
    jobs = pulse.uploader.jobs
    if len(jobs) == 0:
        print("No jobs...  :-(")
        return
    job = jobs[-1]
    return LastUpload(job)


class LastUpload:
    def __init__(self, job):
        self.job = job
        self.index = job.index
        self.segments = job.sequence
        self.segment_start = []
        print(f'job index:{job.index}')
        print(f'sequence: {len(self.segments)} segments')
        end = 0
        for segment in self.segments:
            duration = segment.get_total_time(self.index)
            start = end
            end += duration
            self.segment_start.append(start)
            print(f'  {start:8.1f} - {end:8.1f}, ({duration:8.1f}), shape:{segment.shape}')

    def describe(self, segment=None, channels=None):
        if segment is None:
            segments = self.segments
        else:
            segments = [self.segments[segment]]
        if channels is None:
            channels = list(self.segments[0].channels.keys())

        for channel in channels:
            print(f'Channel: {channel}')
            for i,seg in enumerate(segments):
                seg_offset = self.segment_start[i]
                if isinstance(seg, conditional_segment):
                    for j,branch in enumerate(seg.branches):
                        seg_ch = branch[channel]
                        data = seg_ch._get_data_all_at(self.index).get_data_elements()
                        if len(data):
                            print(f'Segment: {i} - Branch: {j}')
                            self._describe_segment(data, seg_offset)
                else:
                    seg_ch = seg[channel]
                    data = seg_ch._get_data_all_at(self.index).get_data_elements()
                    if len(data):
                        print(f'Segment: {i}')
                        self._describe_segment(data, seg_offset)

    def _describe_segment(self, data, seg_offset):
        for e in data:
            print(f'{e.start+seg_offset:8.1f} - {e.stop+seg_offset:8.1f} ', end='')
            if isinstance(e, OffsetRamp):
                print(f'ramp {e.v_start:6.2f}, {e.v_stop:6.2f} mV')
            elif isinstance(e, custom_pulse_element):
                print(f'custom {e.amplitude:6.2f} (*{e.scaling:5.3f}), {e.kwargs}')
            elif isinstance(e, PhaseShift):
                print(f'shift phase {e.phase_shift:+6.3f} rad ({e.channel_name})')
            elif isinstance(e, IQ_data_single):
                print(f'MW pulse {e.amplitude:6.2f} {e.frequency/1e6:6.1f} MHz {e.phase_offset:+6.3f} rad ({e.ref_channel})')

