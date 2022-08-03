.. title: Basic example

Basic example
=============



Configure pulse_lib
-------------------

Create a pulse_lib object with gates for voltage pulses and 2 qubit channels for MW pulses.

.. code-block:: python
	from pulse_lib.base_pulse import pulselib
	from pulse_lib.virtual_channel_constructors import IQ_channel_constructor

    pl = pulse_lib(backend='Qblox')
	pl.configure_digitizer = True

    # add AWGs and digitizers
    pl.add_awg(awg1)
    p1.add_digitizer(digitizer)

    # gates (voltage channels)
    pl.define_channel('P1', awg1, 1)
    pl.define_channel('P2', awg1, 2)

    # IQ output pair
    pl.define_channel('I1', awg1, 3)
    pl.define_channel('Q1', awg1, 4)
    IQ_pair_1 = IQ_channel_constructor(pl)
    IQ_pair_1.add_IQ_chan("I1", "I")
    IQ_pair_1.add_IQ_chan("Q1", "Q")

    # set frequency of the MW source for down-conversion
    IQ_pair_1.set_LO(mw_source.freq)

    # add qubit channels
    IQ_pair_1.add_virtual_IQ_channel("q1", q1_freq)
    IQ_pair_1.add_virtual_IQ_channel("q2", q2_freq)

    # acquisition channel
    pl.define_digitizer_channel('S1', digitizer, 1)

    pl.finish_init()


Create a sequence
-----------------

A sequence is made of one or more segments. Here we use a single segment.

.. code-block:: python

    seg = pl.mk_segment()

Initialize the qubit with a voltage pulse on the gates.
After the pulse the voltage returns to (0.0, 0.0)

Voltage pulses on multiple channels can be added with ``add_block`` and ``add_ramp`` on the segment.
The start and end time of a pulse are relative to a reference time in the segment.
The reference time is moved to the end of the last pulse with reset_time.

.. code-block:: python

    gates = ['P1','P2']
    # initialize qubit with 200 ns pulse of P1,P2 to v_init
    v_init = (46.0, 25.5)
    seg.add_block(0, 200, gates, v_init, reset_time=True)
    seg.wait(40, reset_time=True)

Generate a Ramsey pulse sequence with two MW pulses for qubit 1 with a wait time in between.
Channels can be accessed as an attribute of the segment (``seg.q1``), and as an index in the segment (``seg['q1']``).

.. code-block:: python

    seg.q1.add_MW_pulse(0, q1_X90_t, q1_X90_amplitude)
    seg.q1.wait(t_wait, reset_time=True)
    seg.q1.add_MW_pulse(0, q1_X90_t, q1_X90_amplitude)
    seg.wait(20, reset_time=True)

Move the qubit to the readout point and start measurement 140 ns after start of the block pulse.

.. code-block:: python

    v_readout = (46.0, 25.5)
    seg.add_ramp(0, 200, gates, (0.0, 0.0), v_readout, reset_time=True)
    seg.S1.acquire(140, t_measure=1500)
    seg.add_block(0, 1700, gates, v_readout, reset_time=True)

Execute the sequence
--------------------

The segments have to be compiled to a sequence. The sequence has to be uploaded to the AWGs before execution.
The acquired data can be retrieved with the acquisition parameter. This is a qcodes MultiParameter.

.. code-block:: python

    seq = pl.mk_sequence([seg])
	
	# NOTE: A hardware schedule must be set for Keysight and Tektronix. See below.
	# sequence.set_hw_schedule(hw_schedule)
	
    measurement_param = seq.get_measurement_param()

    # upload sequence data to AWG
    seq.upload()
    # play the sequence
    seq.play()

    # retrieve measurement data
    data = measurement_param()


Hardware schedule
-----------------

Pulselib needs a hardware schedule to properly configure the digitizer triggers 
and loops on the Tektronix and Keysight AWGs.

Keysight
////////
For Keysight you need HVI2 scripts with a license to use Pathwave TestSyncExecutive.
There is a set of scripts available in core-tools.

.. code-block:: python

	from core_tools.HVI2.hvi2_schedule_loader import Hvi2ScheduleLoader

    seq = pl.mk_sequence([seg])
	
	hw_schedule = Hvi2ScheduleLoader(pl, 'SingleShot', digitizer)
	sequence.set_hw_schedule(hw_schedule)


Tektronix + M4i hardware schedule
/////////////////////////////////

.. code-block:: python

	from pulse_lib.schedule.tektronix_schedule import TektronixSchedule

    seq = pl.mk_sequence([seg])
	
	hw_schedule = TektronixSchedule(pl)
	sequence.set_hw_schedule(hw_schedule)

Tektronix + ATS hardware schedule
/////////////////////////////////

.. code-block:: python

	from pulse_lib.schedule.tektronix_schedule import TektronixAtsSchedule

    seq = pl.mk_sequence([seg])
	
	hw_schedule = TektronixAtsSchedule(pl, acquisition_controller)
	sequence.set_hw_schedule(hw_schedule)


Tektronix + ATS hardware schedule
/////////////////////////////////

.. code-block:: python

	from pulse_lib.schedule.tektronix_schedule import TektronixUHFLISchedule

    seq = pl.mk_sequence([seg])
	
	hw_schedule = TektronixUHFLISchedule(pl, lockin, seq.n_reps)
	sequence.set_hw_schedule(hw_schedule)


