
from qcodes.instrument.base import Instrument

class MockM4i(Instrument):
    def __init__(self, name):
        super().__init__(name)
        self.timeout = Parameter(10000)

    def get_idn(self):
        return dict(vendor='Pulselib', model=type(self).__name__, serial='', firmware='')

    def start_triggered(self):
        pass

class Parameter:
    def __init__(self, value):
        self.value = value

    def set(self, value):
        self.value = value

    def cache(self):
        return self.value