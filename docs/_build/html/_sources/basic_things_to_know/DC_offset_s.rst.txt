.. _dc_offset_comp:

Compenstating for DC offset's
=============================

Most setup's for spin qubits use biasT's in their set up's to combine dc and AC signals. Example below.

.. figure:: /img/DC_offset_bias_T.png

The DC signal provides a static voltage and the AC signals are used for fast control.

When applying pulses with the AWG, they are send down through a coax, to the sample.
When the signal hits the biasT, it act as a high pass filter.

.. figure:: /img/DC_offset_pulse.png

Practically, this means that we filter out the DC part of the signal. In other words, assuming we have an infinite amount of block pulses, we need will need to shift the level in such a way the pulse is symmetric around zero, as the DC part is removed.

As you can see in the picture, the pulse that originally had a voltage between 0 to 1V, will over time return a voltage change on the device from to -0.5 to 0.5V as the capacitor of the biasT gets charged.

To correct for this, the integral of the total pulse needs to be zero.
This can be done automatically in software, by setting the :ref:`compensation limits<volt_comp_bound>` and enabling this option in the sequencer [TODO add ref.].

.. _dc_offset_comp_implemenation:

Implementation details
^^^^^^^^^^^^^^^^^^^^^^

To correct for this, in software is done the following:

	1. For the current sequence, calculate for each channel (e.g. B0, B1, B2, P1, P2) the integral (:math:`\int V(t) dt`).
	2. Calculate for each channel the time (:math:`t_{chan}`) that would be needed to compensate the integral (according to the :ref:`maximal range <volt_comp_bound>` given for the channel).
	3. Calculate minimal time (:math:`t_{comp}`) needed to compensate for all the channels (:math:`Max(t_{chan})`).
	4. Append a segment for each channel that contains the compensation voltage :math:`V` for a time :math:`t_{comp}`

Done!