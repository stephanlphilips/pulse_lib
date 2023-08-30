import numpy as np

def test_rounding():
#    max_shuttle_rounds = 6000
#    n_steps = 20
#
#    # option 1: relying on rounding and padding at the end to get total time.
#    shuttles = np.round(np.geomspace(1, max_shuttle_rounds, n_steps))
#
#    # optino 2: use nice series of values matching 1 ns wqit times
#    shuttles = np.array([1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 3600, 4500, 6000])

    max_shuttle_rounds = 20
    shuttle_step = 1
    shuttles = np.arange(1, max_shuttle_rounds+1, shuttle_step)

    ramp_time = 5 # ns
    total_wait = 200

    wait_per_dot = ((total_wait-(2*ramp_time*shuttles))/2*shuttles)
#    wait_per_dot = total_wait/(2*shuttles)-ramp_time
#    wait_per_dot = total_wait/shuttles
    wait_per_dot_int = np.floor(wait_per_dot)
    padding = total_wait - wait_per_dot_int*shuttles

    # NOTE: total time includes the ramp time. Total time is not constant!
    total_time = (2*ramp_time+2*wait_per_dot_int)*shuttles + padding

    with np.printoptions(formatter={'float':lambda x:f'{x:6.1f}'}):
        print(shuttles)
        print(wait_per_dot)
        print(wait_per_dot_int)
        print(padding)
        print(total_time)


if __name__ == '__main__':
    test_rounding()
