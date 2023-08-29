import pulse_lib.segments.utility.looping as lp
import numpy as np

def test_looping():
    max_shuttle_rounds = 4
    shuttle_step = 1
    shuttles = lp.arange(1, max_shuttle_rounds+1, shuttle_step, axis=0, name='shuttles', unit='#')

    ramp_time = 5 # ns
    total_wait = lp.linspace(100, 300, 3, axis=1, name='wait', unit='ns')

    wait_per_dot = ((total_wait-(2*ramp_time*shuttles))/(2*shuttles))
#    wait_per_dot = total_wait/(2*shuttles)-ramp_time
#    wait_per_dot = total_wait/shuttles
    wait_per_dot_int = np.floor(wait_per_dot)
    padding = total_wait - wait_per_dot_int*shuttles

    # NOTE: total time includes the ramp time. Total time is not constant!
    total_time = (2*ramp_time+2*wait_per_dot_int)*shuttles + padding

    avg_wait = total_time / shuttles

    offset = lp.linspace(1,6,5,axis=2, name='offset')
    avg_wait2 = (total_time + offset) / shuttles

    with np.printoptions(formatter={'float':lambda x:f'{x:6.1f}'}):
        print(shuttles)
        print(total_wait)
        print(shuttles*total_wait)
        print(wait_per_dot)
        print(wait_per_dot_int)
        print(padding)
        print(total_time)
        print(avg_wait)
        print((shuttles*total_wait)*(shuttles*total_wait))
        print('3-axis')
        print(avg_wait2)


if __name__ == '__main__':
    test_looping()
