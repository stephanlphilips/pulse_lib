from qcodes import Parameter

from .physical_channels import digitizer_channel as DigitizerChannel


class RfParameters:
    def __init__(self, digitizer_channel: DigitizerChannel):
        self._frequency = RfFrequencyParameter(digitizer_channel)
        self._source_amplitude = RfAmplitudeParameter(digitizer_channel)
        self._phase = RfPhaseParameter(digitizer_channel)

    @property
    def frequency(self):
        return self._frequency

    @property
    def source_amplitude(self):
        return self._source_amplitude

    @property
    def phase(self):
        return self._phase


class RfFrequencyParameter(Parameter):
    def __init__(self, digitizer_channel: DigitizerChannel):
        super().__init__(
            name=f'{digitizer_channel.name}_frequency',
            label=f'{digitizer_channel.name} resonator frequency',
            unit='Hz')
        self.channel = digitizer_channel

    def get_raw(self):
        return self.channel.frequency

    def set_raw(self, value):
        self.channel.frequency = value


class RfAmplitudeParameter(Parameter):
    def __init__(self, digitizer_channel: DigitizerChannel):
        super().__init__(
            name=f'{digitizer_channel.name}_amplitude',
            label=f'{digitizer_channel.name} rf source amplitude',
            unit='mV')
        self.channel = digitizer_channel

    def get_raw(self):
        if self.channel.rf_source is None:
            return None
        return self.channel.rf_source.amplitude

    def set_raw(self, value):
        if self.channel.rf_source is None:
            raise Exception(f'No RF source configured for {self.channel.name}')
        self.channel.rf_source.amplitude = value


class RfPhaseParameter(Parameter):
    def __init__(self, digitizer_channel: DigitizerChannel):
        super().__init__(
            name=f'{digitizer_channel.name}_phase',
            label=f'{digitizer_channel.name} phase',
            unit='rad')
        self.channel = digitizer_channel

    def get_raw(self):
        return self.channel.phase

    def set_raw(self, value):
        self.channel.phase = value
