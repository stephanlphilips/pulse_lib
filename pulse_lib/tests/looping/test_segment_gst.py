
from pulse_lib.tests.configurations.test_configuration import context
import pulse_lib.segments.utility.looping as lp

#%%
def test_slow(n_gates=10, n_seq=50):
    pulse = context.init_pulselib(n_gates=7, n_qubits=4, n_sensors=2, n_markers=1,
                                  virtual_gates=True)

    f_q1 = 2.450e9
    n_seq = lp.arange(1,n_seq+1,1, axis=0, name='i_sequence')
    n_rep = lp.arange(1,2,1, axis=1, name='n_repeat')

    s = pulse.mk_segment()

    context.segment = s

    s.update_dim(n_seq)
    s.update_dim(n_rep)

    for n,nr in enumerate(n_rep):
        for m,np in enumerate(n_seq):
            for _ in range(n_gates):
                seg = s[n][m]
#                s[n][m].vP1.add_ramp_ss(0, 100, -80, 80)
#                s[n][m].vP1.wait(10)
#                s[n][m].reset_time()
#                s[n][m].q1.add_MW_pulse(0, 10, 50.0, f_q1)
#                s[n][m].reset_time()
                seg.vP1.add_ramp_ss(0, 100, -80, 80)
                seg.vP1.wait(10)
                seg.reset_time()
                seg.q1.add_MW_pulse(0, 10, 50.0, f_q1)
                seg.reset_time()

    return pulse.mk_sequence([s])


def test_pretty_fast(n_gates=10, n_seq=50):
    pulse = context.init_pulselib(n_gates=7, n_qubits=4, n_sensors=2, n_markers=1,
                                  virtual_gates=True)

    f_q1 = 2.450e9
    n_seq = lp.arange(1,n_seq+1,1, axis=0, name='i_sequence')
    n_rep = lp.arange(1,2,1, axis=1, name='n_repeat')

    s = pulse.mk_segment()

    context.segment = s

    s.update_dim(n_seq)
    s.update_dim(n_rep)

    for m,np in enumerate(n_seq):
        seg = s[0][m]
        for _ in range(n_gates):
            seg.vP1.add_ramp_ss(0, 100, -80, 80)
            seg.vP1.wait(10)
            seg.reset_time()
            seg.q1.add_MW_pulse(0, 10, 50.0, f_q1)
            seg.reset_time()


    return pulse.mk_sequence([s])

def test_fast(n_gates=10, n_seq=50):
    pulse = context.init_pulselib(n_gates=7, n_qubits=4, n_sensors=2, n_markers=1,
                                  virtual_gates=True)

    f_q1 = 2.450e9
    n_seq = lp.arange(1,n_seq+1,1, axis=0, name='i_sequence')
    n_rep = lp.arange(1,2,1, axis=1, name='n_repeat')

    s = pulse.mk_segment()

    context.segment = s

    s.update_dim(n_seq)

    for m,np in enumerate(n_seq):
        seg = s[m]
        for _ in range(n_gates):
            seg.vP1.add_ramp_ss(0, 100, -80, 80)
            seg.vP1.wait(10)
            seg.reset_time()
            seg.q1.add_MW_pulse(0, 10, 50.0, f_q1)
            seg.reset_time()

    s.update_dim(n_rep)

    return pulse.mk_sequence([s])

#%%
if __name__ == '__main__':
    import time
    for _ in range(3):
        start = time.perf_counter()
        seq = test_slow()
        duration = time.perf_counter() - start
        print(f'duration 1: {duration*1000:5.0f} ms')
        time.sleep(duration/2)

    time.sleep(1)
    for _ in range(5):
        start = time.perf_counter()
        seq = test_pretty_fast()
        duration = time.perf_counter() - start
        print(f'duration 2: {duration*1000:5.0f} ms')
        time.sleep(duration/2)

    time.sleep(1)
    for _ in range(5):
        start = time.perf_counter()
        seq = test_fast()
        duration = time.perf_counter() - start
        print(f'duration 3: {duration*1000:5.0f} ms')
        time.sleep(duration/2)

'''
with HVI:
duration 1:  8799 ms
duration 1:  9025 ms
duration 1:  8847 ms
duration 2:  1229 ms
duration 2:  1176 ms
duration 2:  1272 ms
duration 2:  1193 ms
duration 2:  1249 ms
duration 3:   866 ms
duration 3:   749 ms
duration 3:   705 ms
duration 3:   795 ms
duration 3:   753 ms

without HVI:
duration 1:  1005 ms
duration 1:  1011 ms
duration 1:  1080 ms
duration 2:   578 ms
duration 2:   528 ms
duration 2:   535 ms
duration 2:   452 ms
duration 2:   529 ms
duration 3:   366 ms
duration 3:   346 ms
duration 3:   430 ms
duration 3:   384 ms
duration 3:   477 ms
'''