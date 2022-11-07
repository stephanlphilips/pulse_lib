import qcodes as qc

from pulse_lib.tests.mock_m3202a_qs import MockM3202A_QS
from pulse_lib.tests.mock_m3102a_qs import MockM3102A_QS

if not qc.Station.default:
    station = qc.Station()
else:
    station = qc.Station.default

_use_dummy=True

def add_awg(name, slot_nr):
    if name in station.components:
        awg = station[name]
    else:
        awg = MockM3202A_QS(name, 0, slot_nr)
        station.add_component(awg)
    return awg

def add_dig(name, slot_nr):
    try:
        dig = station[name]
    except:
        dig = MockM3102A_QS(name, 0, slot_nr)
        station.add_component(dig)
    return dig

awg1 = add_awg('AWG1', 2)
awg2 = add_awg('AWG2', 5)
dig1 = add_dig('Dig1', 7)
