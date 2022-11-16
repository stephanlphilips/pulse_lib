from collections.abc import Sequence
import qcodes as qc
from qcodes.measure import Measure
from qcodes.loops import Loop
from qcodes.actions import Task

from pulse_lib.sequencer import sequencer

def upload_play(seq):
    seq.upload()
    seq.play()

def qc_run(name, *args, quiet=False):

    seq = None
    meas_params = []
    sweeps = []
    delays = {}
    for arg in args:
        if isinstance(arg, sequencer):
            seq = arg
        elif isinstance(arg, (qc.Parameter, qc.MultiParameter)):
            meas_params.append(arg)
        elif isinstance(arg, Sequence):
            if len(arg) not in [4,5]:
                raise ValueError(f'incorrect sweep {arg}')
            param = arg[0]
            if not isinstance(param, qc.Parameter):
                raise TypeError(f'Expected Parameter, got {type(param)}')
            sweeps.append(param[arg[1]:arg[2]:arg[3]])
            if len(arg) == 5:
                delay = arg[4]
                delays[len(sweeps)-1] = delay

    actions = []

    if seq is not None:
        for sp in seq.params:
            sweep = sp[sp.values]
            # np.random.shuffle(sweep._values)
            sweeps.append(sweep)
        play_task = Task(upload_play, seq)
        actions.append(play_task)

    loop = None
    for i,sweep in enumerate(sweeps):
        delay = delays.get(i, 0)
        if loop is None:
            loop = Loop(sweep, delay)
        else:
            loop = loop.loop(sweep, delay)

    actions += meas_params
    if loop is not None:
        m = loop.each(*actions)
    else:
        m = Measure(*actions)

    ds = m.run(loc_record={'name':name}, quiet=quiet)
    return ds
