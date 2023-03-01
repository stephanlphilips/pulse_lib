
from pulse_lib.tests.configurations.test_configuration import context
import pulse_lib.segments.utility.looping as lp

#%%
def add(segment, template):
    template.build(segment)
    segment.reset_time()

def add_simultaneous(pulse, segment, templates):
    for template in templates:
	    sub = pulse.mk_segment()
	    template.build(sub)
	    segment.add(sub)
    segment.reset_time()

class BlockTemplate:
    def __init__(self, gate, voltage):
        self.gate = gate
        self.voltage = voltage

    def build(self, segment, **kwargs):
        segment[self.gate].add_block(0, 100, self.voltage)

class MWTemplate:
    def __init__(self, qubit, f):
        self.qubit = qubit
        self.f = f

    def build(self, segment, **kwargs):
        segment[self.qubit].add_MW_pulse(5, 25, 70, self.f)
        segment.wait(5)

#%%

def test1():
    pulse = context.init_pulselib(n_gates=2, n_qubits=2)

    s = pulse.mk_segment()

    s.P1.add_block(10, 20, -10)
    s.wait(30, reset_time=True)
    add(s, BlockTemplate('P1', 90))
    add_simultaneous(pulse, s, [BlockTemplate('P1', 50), BlockTemplate('P2', -50)])
    add_simultaneous(pulse, s, [BlockTemplate('P1', 60), BlockTemplate('P2', -30), BlockTemplate('P1', 25)])
    add_simultaneous(pulse, s, [MWTemplate('q1', 2.45e9), MWTemplate('q2', 2.5e9)])
    add_simultaneous(pulse, s, [MWTemplate('q2', 2.5e9)])
    add_simultaneous(pulse, s, [MWTemplate('q1', 2.45e9)])

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 10
    context.add_hw_schedule(sequence)

    context.plot_segments([s])
    context.plot_awgs(sequence, ylim=(-0.100,0.100))


def test2():
    pulse = context.init_pulselib(n_gates=2, n_qubits=2, n_sensors=1, n_markers=1)

    n_cliff = lp.array([1,3,5], axis=0, name='n_clif')
    s = pulse.mk_segment()

    s.update_dim(n_cliff)

    for i,n in enumerate(n_cliff):
        si = s[i]
        si.P1.add_block(10, 20, -10*n)
        si.wait(30, reset_time=True)
        for _ in range(int(n)):
            add(si, BlockTemplate('P1', 10*i))
            add_simultaneous(pulse, si, [BlockTemplate('P1', 80-10*i), BlockTemplate('P2', -50)])

    sequence = pulse.mk_sequence([s])
    sequence.n_rep = 10
    context.add_hw_schedule(sequence)

    for i,n in enumerate(n_cliff):
        context.plot_segments([s], index=[i])
        context.plot_awgs(sequence, index=[i], ylim=(-0.100,0.100))


#%%
if __name__ == '__main__':
    ds1 = test1()
    ds2 = test2()

