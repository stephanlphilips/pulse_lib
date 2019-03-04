.. _virt_gates:

Virtual Gates
=============

When working with spin qubits, gates control the interaction between the electrons. 
Often there is cross talk between the gates, e.g. when you change the chemical potential of the first dot with the P1 gate, you would see that the chemical potential of the dots near to the first dot, would also be effected.
These effect, are classical in nature (-> just capacitance) and can be corrected easily when the capacitaces are known.


Defining a virtual gate matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A virtual gate matrix is a matrix that relates the virtual gate voltages to the real voltages.
	
	:math:`\begin{pmatrix} P1 \\ P2 \\ P3 \end{pmatrix} = M \begin{pmatrix} vP1 \\ vP2 \\ vP3 \end{pmatrix}`

When there would me no cross effects, this matrix would look like:
	
	:math:`M = \begin{pmatrix} 1  & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}`

Which gives a one to one map of the virtual gate voltages to the real voltages. In reality, this will not be the case. First you need to measure the cross capacitances between all the dots. This can for example quite easily by extracted <reference> from charge stablility diagram, but there are also many other methods <references>.

Using the cross capacitances, you can generate the following matrix:

	:math:`M^{-1} = \begin{pmatrix} C_{11}  & C_{12} & C_{13} \\ C_{21} & C_{22} & C_{23} \\ C_{31} & C_{32} & C_{33} \end{pmatrix}`

Then when you invert is to :math:`m`, you have your virtual gate matrix.

*tip : When contructing the* :math:`M^{-1}` *matrix, it is recommended to normalize each row of this matrix. Otherwise the ratio of real voltage to virtual gate voltage will always change drasically when adding a gate to the matrix. Also the value of the virtual gate will be kind of similar to the real voltage (easier to keep intution with the voltages).*

How does the software deal with virtual gates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As seen in the example :ref:`init file<example_init_lib>` you can define both real and virtual channels.

In the example case, we have the channel B0, B1, B2, P1, P2, and their virtual versions, perluded by the v character (note you can call them anyway you like).
When assinging pulses to these channels, you can add to the real ones and virtual ones at the same time. When rendering the following will be done, e.g. for B0:

	:math:`B_{0,new}(t) = B_0(t) + \sum_i x_i \: vB_i(t) + \sum_i y_i \: vP_i(t)`

Where :math:`x_i` and :math:`y_i` represent parts out of the virtual gate matrix (-> the first row).