
from pulse_lib.tests.configurations.test_configuration import context
from pulse_lib.fast_scan.qblox_fast_scans import fast_scan1D_param, fast_scan2D_param

#%%
def test1():
    pulse = context.init_pulselib(n_gates=2, n_sensors=2, rf_sources=True)

    m_param1D = fast_scan1D_param(
            pulse,
            'P1', 100, 51, 2000,
            iq_complex=True)

    m_param2D = fast_scan2D_param(
            pulse,
            'P1', 100, 51,
            'P2', 20, 21,
            2000,
            iq_mode='Complex')

#    data1 = m_param1D()
    print(m_param1D.names)
#    print(data1)
#    data2 = m_param2D()
    print(m_param2D.names)
#    print(data2)
    print(flush=True)

    return context.run('fast_scan', m_param1D, m_param2D)

def test2():
    pulse = context.init_pulselib(n_gates=2, n_sensors=2, rf_sources=True)

    m_param1D = fast_scan1D_param(
            pulse,
            'P1', 100, 51, 2000,
            iq_complex=False)

    m_param2D = fast_scan2D_param(
            pulse,
            'P1', 100, 51,
            'P2', 20, 21,
            2000,
            iq_mode='I')

#    data1 = m_param1D()
    print(m_param1D.names)
#    print(data1)
#    data2 = m_param2D()
    print(m_param2D.names)
#    print(data2)
    print(flush=True)

    return context.run('fast_scan_IQ', m_param1D, m_param2D)

def test3():
    pulse = context.init_pulselib(n_gates=2, n_sensors=2, rf_sources=True)

    m_param1D = fast_scan1D_param(
            pulse,
            'P1', 100, 51, 2000,
            iq_complex=False)

    m_param2D = fast_scan2D_param(
            pulse,
            'P1', 100, 51,
            'P2', 20, 21,
            2000,
            iq_mode='I+Q')

#    data1 = m_param1D()
    print(m_param1D.names)
#    print(data1)
#    data2 = m_param2D()
    print(m_param2D.names)
#    print(data2)
    print(flush=True)

    return context.run('fast_scan_IQ', m_param1D, m_param2D)

def test4():
    pulse = context.init_pulselib(n_gates=2, n_sensors=2, rf_sources=True)

    m_param2D = fast_scan2D_param(
            pulse,
            'P1', 100, 6,
            'P2', 20, 5,
            20_000,
            iq_mode='amplitude+phase')


    return context.run('fast_scan_IQ', m_param2D)

if __name__ == '__main__':
    ds1 = test1()
    ds2 = test2()
    ds3 = test3()
    ds4 = test4()
