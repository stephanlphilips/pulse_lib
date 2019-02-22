.. _chan_delay:

Channels delays
===============

Some signals that you send to you sample, take a different path then others. This causes a slightly different time of arrival (usually a few tens of ns).
It is possible to compenstate for these small offsets by pushing waveforms slightly forward or backward.

Let's consider a simple example. Let's set we define in the :ref:`pulselib<pulse_lib_chan_delay>` object the following delays:

.. code-block:: python

	pulse.add_channel_delay({
		'chan 1': -20, 
		'chan 2': -20,
		'chan 3': 20, 
		'chan 4': 20,
		'chan 5': 20
	})

It means channel 1 and 2 need to arrive 20 ns before what we call our reference. For the other channels we want the signal to arrive 20 ns after. The image below shows how this is done:

..  figure:: /img/channel_delays.png

As you can see here, for channel 1 and 2, in the segment container object, the segments are artifically made a bit longer by pushing data points in front of it. 
By default the value of this point will be the first value in case there was no correction. The same counts for the last data point.
