.. title: Cross-capacitance

Cross-capacitance
=================
The chemical potential of the quantum dots used in spin-qubit devices determines the amount of electrons that
will be loaded in the quantum dot. The chemical potential is controlled by a plunger gate close to the quantum dot.
However, due to the small distance between quantum dots the potential of a quantum dot is also affected
by neighbouring plunger and barrier gates. This is the capacitive cross-talk, or cross-capacitance, from a gate to
other quantum dots. This cross-capacitance is classical in nature and can be corrected by pulse_lib using
"virtual gates".

A virtual gate is a pulse_lib channel that outputs a pulse sequence on multiple AWG channels
such that only a single chemical potential or single tunnel coupling will be affected by the pulses.
Virtual gates are defined by a virtual gate matrix.


Virtual Gate Matrix
===================

A virtual gate matrix is a matrix that relates the virtual gate voltages to the real voltages.

	:math:`\begin{pmatrix} vP1 \\ vP2 \\ vP3 \end{pmatrix} = M \begin{pmatrix} P1 \\ P2 \\ P3 \end{pmatrix}`

When there would me no cross effects, this matrix would look like:

	:math:`M = \begin{pmatrix} 1  & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}`

Which gives a one to one map of the virtual gate voltages to the real voltages.
In reality, this will not be the case. There will be cross-capacitances.

	:math:`M = \begin{pmatrix} 1  & C_{12} & C_{13} \\ C_{21} & 1 & C_{23} \\ C_{31} & C_{32} & 1 \end{pmatrix}`

The values on the diagonal are kept equal to 1. The lever arm of :math:`vP1` will then be equal to the lever
arm of :math:`P1`.
A correct virtual matrix will remove the slant from the lines in the charge stability diagram without stretching it.

See *Loading a quantum-dot based "Qubyte" register*, C. Volk (2019), for a detailed description
of cross-capacitance and virtual gates.


Virtual gate matrix in pulse_lib
--------------------------------

Pulse_lib inverts cross-capacitance matrix :math:`M` to calculate the voltages on the output channels.

	:math:`\begin{pmatrix} P1 \\ P2 \\ P3 \end{pmatrix} = M^{-1} \begin{pmatrix} vP1 \\ vP2 \\ vP3 \end{pmatrix}`

The cross-capacitance matrix (real-to-virtual) can be passed to pulse_lib, but also the inverted
matrix (virtual-to-real) can be passed. The cross-capacitance matrix has to be a square matrix, because it
must be invertible. The virtual-to-real matrix doesn't have to be square.

Example:
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
        real2virtual=True)


Combining virtual gates to a combined parameter
-----------------------------------------------

The use of the virtual matrix is not restricted to the definition of virtual gates to compensate
for the cross-capacitance. It can also be used to define a new channel for voltage pulses that is
a linear combination of virtual gates, e.g. to define a detuning parameter.

Example:
  A detuning parameter e12 and a energy parameter U12 is defined using a virtual-to-real matrix.

  .. code-block:: python

      pl.add_virtual_matrix(
        name='detuning12',
        real_gate_names=['vP1', 'vP2'],
        virtual_gate_names=['e12', 'U12'],
        matrix=[
            [+0.5, +1.0],
            [-0.5, +1.0],
            ],
        real2virtual=False)


