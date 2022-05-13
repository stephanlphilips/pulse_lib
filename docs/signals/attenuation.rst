.. title: Attenuation

Signal attenuation
==================

The signal from AWG to device is electrically attenuated to reduce the amount of noise on the device.
The signal-to-noise ratio of the AWG is high when the signal has a high amplitude. On the device the
required signal amplitude is much lower and must be attenuated.

Pulselib compensates for this attenuation. The value should be specified as the fraction of the signal
transmitted to the device.

Example:
  If the attenuation is 12 dB, then the amplitude at the device will be 0.25 of the amplitude at the AWG.
  An attenuation of 0.25 should be specified for the channel.

  .. code-block:: python

      pl.add_channel_attenuation('P1', 0.25)
