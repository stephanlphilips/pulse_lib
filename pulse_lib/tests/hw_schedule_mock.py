import logging

logger = logging.getLogger(__name__)

class HardwareScheduleMock:
    def __init__(self):
        self.loaded = False
        self.sequence_params = None

    def compile(self, sequencer_hardware):
        pass

    def load(self):
        logger.info('load schedule')
        self.loaded = True

    def is_loaded(self):
        return self.loaded

    def unload(self):
        logger.info('unload schedule')
        self.loaded = False

    def set_schedule_parameters(self, **kwargs):
        logger.info(f'set parameters {kwargs}')

    def set_configuration(self, *args, **kwargs):
        logger.info(f'set configuration {args} {kwargs}')

    def start(self, waveform_duration, n_repetitions, sequence_params):
        self.sequence_params = sequence_params
        logger.info(f'start {n_repetitions}*{waveform_duration} {sequence_params}')

    def is_running(self):
        return False

    def close(self):
        logger.info(f'close()')

    def stop(self):
        logger.info(f'stop()')


