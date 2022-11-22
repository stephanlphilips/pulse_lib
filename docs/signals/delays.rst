.. title: Signal delay

Signal delays
=============

It takes time for a signal to get from the AWG to the target device and from the device to the digitizer.
Pulses with equal start time but on different channels should arrive at the same time on the target device.
This is not by default true, because signals travel along different paths with different travel times.
Pulse_lib can compensate for these difference in signal delays.

Signal delays in pulse and acquisition paths are caused by:

* propagation through cables, which is approximately 4 - 5 ns/m.
* propagation through vector signal generator, which takes 20 - 50 ns.
* propagation through RF up-convertor, down-converter.
* processing time in AWG
* processing time in digitizer

The delay can be specified per channel. It is specified as the delay added to the channel.

Channel delays add extra samples on the begin and end of the uploaded sequence.

Example:
 If the MW signal is delayed by the VSG and arrives 45 ns later than the signals of gates P1 and P2,
 then you can add 45 ns to the gates P1 and P2.

 .. code-block:: python

     pl.add_channel_delay('P1', 45)
     pl.add_channel_delay('P2', 45)

 Alternatively, you can subtract 45 ns for the I and Q channel.

 .. code-block:: python

     pl.add_channel_delay('I1', -45)
     pl.add_channel_delay('Q1', -45)


Example:
 .. figure:: /img/delay.png
    :scale: 100%

    Delays on P2 and SD1.

 .. code-block:: python

     pl.add_channel_delay('P1', 0)
     pl.add_channel_delay('P2', -10)
     pl.add_channel_delay('SD1', +20)
