
class HardwareSchedule:
    def compile(self, sequencer_hardware):
        pass

    def load(self):
        pass

    def is_loaded(self):
        return False

    def unload(self):
        pass

    def set_schedule_parameters(self, **kwargs):
        pass

    def start(self, waveform_duration, n_repetitions, sequence_params):
        pass
        # start without hvi.start(): start(), trigger(), stop()?

    def is_running(self):
        return False

    def close(self):
        pass

