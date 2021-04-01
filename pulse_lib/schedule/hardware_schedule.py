
class HardwareSchedule:
    def load(self):
        pass

    def unload(self):
        pass

    def set_schedule_parameters(self, **kwargs):
        pass

    def set_configuration(self, params, n_waveforms):
        pass

    def start(self, waveform_duration, n_repetitions, sequence_params):
        pass

    def is_running(self):
        return False
