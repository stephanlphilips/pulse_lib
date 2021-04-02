import logging
from typing import List

from .hardware_schedule import HardwareSchedule

class TektronixSchedule(HardwareSchedule):
    verbose = False

    def __init__(self, AWGs:List['Tektronix_AWG5014'], digitizer):
        self.awgs = AWGs
        self.digitizer = digitizer
        self.running = False
        self.schedule_parms = {}

    def set_schedule_parameters(self, **kwargs):
        for key,value in kwargs.items():
            self.schedule_parms[key] = value

    def set_configuration(self, params, n_waveforms):
        pass

    def load(self):
        pass

    def unload(self):
        pass

    def start(self, waveform_duration, n_repetitions, sequence_parameters):

        timeout_ms = self.digitizer.timeout.cache()
        duration_ms = waveform_duration * n_repetitions * 1e-6
        if duration_ms > timeout_ms:
            logging.warning(f'Duration of sequence ({duration_ms:5.1f} ms) > digitizer timeout ({timeout_ms} ms)')

        if not self.running:
            self.running = True
            for awg in self.awgs:
                element_no = 1
                awg.set_sqel_trigger_wait(element_no)
                awg.set_sqel_loopcnt(n_repetitions, element_no)
                awg.run_mode('SEQ')
                awg.run()

        logging.info('trigger')
        self.digitizer.start_triggered()
        for awg in self.awgs:
            awg.force_trigger()
        logging.info('started')

    def stop(self):
        if self.running:
            for awg in self.awgs:
                awg.stop()
                awg.all_channels_off()

        self.running = False

    def is_running(self):
        return self.running

    def close(self):
        self.stop()

