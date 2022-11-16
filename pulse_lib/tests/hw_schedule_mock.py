import logging

class HardwareScheduleMock:
    def __init__(self):
        self.loaded = False
        self.sequence_params = None

    def compile(self, sequencer_hardware):
        pass

    def load(self):
        logging.info('load schedule')
        self.loaded = True

    def is_loaded(self):
        return self.loaded

    def unload(self):
        logging.info('unload schedule')
        self.loaded = False

    def set_schedule_parameters(self, **kwargs):
        logging.info(f'set parameters {kwargs}')

    def set_configuration(self, *args, **kwargs):
        logging.info(f'set configuration {args} {kwargs}')

    def start(self, waveform_duration, n_repetitions, sequence_params):
        self.sequence_params = sequence_params
        logging.info(f'start {n_repetitions}*{waveform_duration} {sequence_params}')

    def is_running(self):
        return False

    def close(self):
        logging.info(f'close()')

    def stop(self):
        logging.info(f'stop()')


