import qcodes as qc

from pulse_lib.tests.mock_m3202a import MockM3202A_fpga
from pulse_lib.tests.mock_m3102a import MockM3102A

if not qc.Station.default:
    station = qc.Station()
else:
    station = qc.Station.default


def station_get_or_create(func):
    def wrapper(name, *args, **kwargs):
        try:
            return station[name]
        except:
            component = func(name, *args, **kwargs)
            station.add_component(component)
            return component
    return wrapper

@station_get_or_create
def add_awg(name, slot_nr):
    return MockM3202A_fpga(name, 0, slot_nr)

@station_get_or_create
def add_digitizer(name, slot_nr):
    return MockM3102A(name, 0, slot_nr)


#%%

_use_dummy=True

awg1 = add_awg('AWG1', 2)
awg2 = add_awg('AWG2', 3)
dig1 = add_digitizer('DIG1', 6)

