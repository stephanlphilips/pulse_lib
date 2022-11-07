import qcodes as qc

_use_mock = True
_use_core_tools = True

if _use_mock:
    from pulse_lib.tests.mock_m3202a import MockM3202A_fpga
    from pulse_lib.tests.mock_m3102a import MockM3102A
elif _use_core_tools:
    from keysight_fpga.qcodes.M3202A_fpga import M3202A_fpga
    from core_tools.drivers.M3102A import SD_DIG, MODES
    from keysight_fpga.sd1.dig_iq import load_iq_image
else:
    from projects.keysight_fpga.M3202A_fpga import M3202A_fpga
    from projects.keysight_measurement.M3102A import SD_DIG, MODES
    from projects.keysight_fpga.dig_iq import load_iq_image


if not qc.Station.default:
    station = qc.Station()
else:
    station = qc.Station.default


def station_get_or_create(func):
    def wrapper(name, *args, **kwargs):
        if name in station.components:
            return station[name]
        else:
            component = func(name, *args, **kwargs)
            station.add_component(component)
            return component
    return wrapper

@station_get_or_create
def add_awg(name, slot_nr):
    if _use_mock:
        return MockM3202A_fpga(name, 0, slot_nr)
    else:
        return M3202A_fpga(name, 1, slot_nr)

@station_get_or_create
def add_digitizer(name, slot_nr):
    if _use_mock:
        return MockM3102A(name, 0, slot_nr)
    else:
        dig = SD_DIG(name, 1, slot_nr)
        load_iq_image(dig.SD_AIN)
        dig.set_acquisition_mode(MODES.AVERAGE)
        return dig


#%%

awg1 = add_awg('AWG1', 3)
awg2 = add_awg('AWG2', 7)
dig1 = add_digitizer('DIG1', 5)

