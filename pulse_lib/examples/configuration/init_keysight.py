import qcodes as qc

from pulse_lib.tests.mock_m3202a import MockM3202A_fpga

if not qc.Station.default:
    station = qc.Station()
else:
    station = qc.Station.default

_use_dummy=True

def add_awg(name, slot_nr):
    try:
        awg = station[name]
    except:
        awg = MockM3202A_fpga(name, 0, slot_nr)
        station.add_component(awg)
    return awg

awg1 = add_awg('AWG1', 2)
awg2 = add_awg('AWG2', 3)

