.. title: pulse_lib configuration

Configuration
=============

The configuration of pulse_lib is stored in a pulselib object.
This object is needed to create segments and sequences.

Create pulselib object with specific backend
---------------------------------------------

A pulselib object must created for a specific backend for the hardware.

.. code-block:: python

    from pulse_lib.base_pulse import pulselib

    pl = pulselib('Keysight')

    # Add channels ...
    # Configure channels ....

At the end, after adding and configuring all channels, ``pl.finish_init()``
must be called to complete the initialization.


Add AWGs and digitizers
-----------------------

.. code-block:: python

    pl.add_awg(awg1)
    pl.add_awg(awg2)
    pl.add_digitizer(digitizer)

Note: Qblox QRM should be added as a digitizer. See @@@

Gates (voltage channels)
------------------------

Gates (voltage channels) should be defined with a name for the channel and
a name and channel number of the AWG.

The channel delay specifies the amount of delay that should be added to the signal.
The channel attenuation is expressed in the fraction of the amplitude after attenuation.
The DC compensation limits are specified in AWG voltage before attenuation.

.. code-block:: python

    pl.add_channel("P1', awg1.name, 1)
    pl.add_channel_delay('P1', 17)

    # 0.01 = -20 dB attenuation
    pl.add_channel_attenuation('P1', 0.01)

    # Add limits on voltages for DC channel compensation.
    # When no limit is specified then no DC compensation is performed.
    # Limits are specified in AWG voltage before attenuation.
    pl.add_channel_compensation_limit('P1', (-200, 500))

    # Add compensation for bias-T with RC-time of 0.109 s
    pl.add_channel_bias_T_compensation('P1', 0.109)

Virtual matrix
--------------

``keep_square`` and @@@ are optional parameters that can only be set when ``real2virtual=True``. ???

.. code-block:: python

      pl.add_virtual_matrix(
        name='virtual-gates',
        real_gate_names=['B0', 'P1', 'B1', 'P2', 'B2'],
        virtual_gate_names=['vB0', 'vP1', 'vB1', 'vP2', 'vB2'],
        matrix=[
            [1.00, 0.25, 0.05, 0.00, 0.00],
            [0.14, 1.00, 0.12, 0.02, 0.00],
            [0.05, 0.16, 1.00, 0.11, 0.01],
            [0.02, 0.09, 0.18, 1.00, 0.16],
            [0.00, 0.01, 0.013 0.21, 1.00]
            ],
        real2virtual=True,
        keep_square=Tue,
        @@@)

      pl.add_virtual_matrix(
        name='detuning12',
        real_gate_names=['vP1', 'vP2'],
        virtual_gate_names=['e12', 'U12'],
        matrix=[
            [+0.5, +1.0],
            [-0.5, +1.0],
            ],
        real2virtual=False)

Qubit channels (MW channels)
----------------------------

.. code-block:: python

    from pulse_lib.virtual_channel_constructors import IQ_channel_constructor

    pl.add_channel("I1', awg2.name, 3)
    pl.add_channel("Q1', awg2.name, 4)
    pl.add_channel_delay('I1', -52)
    pl.add_channel_delay('Q1', -52)

    # define IQ output pair
    IQ_pair_1 = IQ_channel_constructor(pulse)
    IQ_pair_1.add_IQ_chan("I1", "I")
    IQ_pair_1.add_IQ_chan("Q1", "Q")

    # frequency of the MW source
    IQ_pair_1.set_LO(lo_freq)

    # add channel for qubit q1
    IQ_pair_1.add_virtual_IQ_channel("q1", 3.213e9)

IQ phase-gain compensation
Offset
add marker


Marker channels
---------------

delay


Digitizer channels
------------------

Note: Digitizers  triggers

.. code-block:: python

    pl.define_digitizer_channel('SD1', digitizer.name, 1)

IQ input pair

RF-source

