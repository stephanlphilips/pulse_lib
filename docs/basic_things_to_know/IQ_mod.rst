.. _IQ_mod_basics:

IQ modulation
=============

IQ modulation is a technique used in telecomunications to easily provide a multitude of signals with a controlled phase and frequency.
This technique is widely used for qubits as it is fully phase coherent.

An example of a signal you want to generate might be:

<< imgae of 2 sinuesses at different times >>

Principle
^^^^^^^^^

The principle of IQ modulation relies on mixing the I (in-phase) and the Q (quadrature) signal with a carrier. The circuit of an IQ modulator is shown below:

<< image IQ mixer >>

So, if we quickly work out the signal that this circuit generates, we get:

	:math:`I = A(t)*cos(\omega t + \phi)`

	:math:`Q = A(t)*cos(\omega t + \frac{\pi}{2} + \phi) = A*sin(\omega t + \phi)`

	:math:`I \otimes MW_a = A(t)*cos(\omega t + \phi) * cos(\omega_c t)= \frac{A(t)}{2}\left(cos((\omega - \omega_c)t + \phi) + cos((\omega + \omega_c)t + \phi)\right)`

	:math:`Q \otimes MW_b = A(t)*sin(\omega t + \phi) * sin(\omega_c t)= \frac{A(t)}{2}(cos((\omega - \omega_c)t + \phi) - cos((\omega + \omega_c)t + \phi))`

Resulting in:

	:math:`I \otimes MW_a + Q \otimes MW_b = A(t) cos((\omega - \omega_c)t + \phi)`

As seen in the outcome, we can easily generate mircowaves of any frequency with any amplide profile.  This type of modulation is heavily used when performing single qubit gates. *Tip : always make sure to do not modulate on a frequency of* :math:`\omega = 0` *. This will result in a lot of extra* :math:`\frac{1}{f}` *noise, hence worse qubit performace*

Note that next to going down in frequenncy, it is also possible to go up. In priciple yous just have to invert :math:`\omega \rightarrow -\omega`:

	:math:`I = A(t)*cos(\omega t + \phi)`

	:math:`Q = A(t)*cos(\omega t - \frac{\pi}{2} + \phi) = -A(t)*sin(\omega t + \phi)`

Which, when mixing with the carrier results in:

	:math:`I \otimes MW_a + Q \otimes MW_b = A(t) cos((\omega + \omega_c)t + \phi)`

Important note, don't forget that there is noting holding you back from doing:

	:math:`I = \sum_i A_i(t)*cos(\omega_i t + \phi_i)`

	:math:`Q = \sum_i A_i(t)*sin(\omega_i t + \phi_i)`

In this way you can easily generate multiple microwave frequencies at the same know. In the telecom world this is called frequency multiplexing.