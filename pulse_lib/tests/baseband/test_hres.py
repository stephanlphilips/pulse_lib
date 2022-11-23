
from pulse_lib.tests.configurations.test_configuration import context
import pulse_lib.segments.utility.looping as lp

#%%
def test1():
    pulse = context.init_pulselib(n_gates=2)

    dt = lp.linspace(0.25, 0.75, 3, name='dt', axis=0)

    s = pulse.mk_segment(hres=True)

    for t in dt:
        s.wait(15)
        s.P1.add_ramp_ss(4, 8, 0, 80)
        s.P1.add_block(8, 10, 80)
        s.P1.add_ramp_ss(10, 14, 80, 0)
        s.P2.add_ramp_ss(4-t, 8-t, 0, 80)
        s.P2.add_block(8-t, 10+t, 80)
        s.P2.add_ramp_ss(10+t, 14+t, 80, 0)
        s.reset_time()

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = None
    context.add_hw_schedule(sequence)

    context.plot_awgs(sequence, ylim=(-0.0,0.100), xlim=(0, 50))

#    return context.run('hres1', sequence)

def test2():
    pulse = context.init_pulselib(n_gates=2)

    dt = lp.linspace(0.25, 0.75, 3, name='dt', axis=0)

    s = pulse.mk_segment(hres=True)
    t_ramp = 3

    for t in dt:
        s.P1.wait(4)
        s.P1.reset_time()
        s.P1.add_ramp_ss(0, t_ramp, 0, 80)
        s.P1.reset_time()
        s.P1.add_block(0, 8, 80)
        s.P1.reset_time()
        s.P1.add_ramp_ss(0, t_ramp, 80, 0)
        s.P1.wait(4)

        s.P2.wait(4+t)
        s.P2.reset_time()
        s.P2.add_ramp_ss(0, t_ramp, 0, 80)
        s.P2.reset_time()
        s.P2.add_block(0, 8, 80)
        s.P2.reset_time()
        s.P2.add_ramp_ss(0, t_ramp, 80, 0)

        s.reset_time()

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = None
    context.add_hw_schedule(sequence)

    context.plot_awgs(sequence, ylim=(-0.0,0.100), xlim=(0, 70))

#    return context.run('hres1', sequence)

def test3():
    pulse = context.init_pulselib(n_gates=3)

    dt = lp.linspace(0, 1.0, 6, name='dt', axis=0)

    s = pulse.mk_segment(hres=True)

    s.wait(10)
    s.P1.add_block(4, 8, 80)
    s.P2.add_block(4+dt, 8+dt, 80)
    s.P3.add_block(4-dt, 8-dt, 80)
    s.reset_time()
    s.wait(15)
    s.P1.add_ramp_ss(4, 8, 0, 80)
    s.P1.add_block(8, 10, 80)
    s.P1.add_ramp_ss(10, 14, 80, 0)
    s.P2.add_ramp_ss(4-dt, 8-dt, 0, 80)
    s.P2.add_block(8-dt, 10+dt, 80)
    s.P2.add_ramp_ss(10+dt, 14+dt, 80, 0)

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = None
    context.add_hw_schedule(sequence)

#    sequence.dt(0.2)
#    context.plot_awgs(sequence, ylim=(-0.100,0.100), xlim=(0, 26))
    for t in sequence.dt.values:
        sequence.dt(t)
        context.plot_awgs(sequence, ylim=(-0.100,0.100))

#    return context.run('hres2', sequence)

#%%
if __name__ == '__main__':
    ds1 = test1()
    ds2 = test2()
    ds3 = test3()
