.. _struct_lib:

Structure of the library
========================

The structure to the library is given by the following three classes:

   - :ref:`seg`
   - :ref:`seg_con`
   - :ref:`seq`
These is also the pulse-lib class, which is an organizational element. This element is used to generate :ref:`empty segment containers<mk_segment_container>` and :ref:`set general settings<init_lib>`.

**Overview**

A quick overview of the relation between the different structural elements can be found below,

.. figure:: /img/lib_struct.png

.. _seg:

Segments
--------
A segment is the collection of data points that forms a single waveform. For example, the set of pulses one would apply on one the gates for the sample to perform the readout.
Note that a segment only contains the information for one channel.

There are two types of segments:
   - :ref:`segments_base<segment_base>` : these are segments that should be used for generating block pulses and ramps.
   - segments_IQ : these are segments that are used to work with IQ signals of vector MW sources.
Both types can be used interchangeably. More details can be found in the later chapters (or if you feel really courageous in the source code).


.. _seg_con:

Segment containers
------------------
A segment container, as the name already gives away, is a container of channels. For example, this could be the collection of all the channels that are involved in your readout pulse. This object is generated automatically, so no need for you to specify channels each time.

There are some rules for segment containers though, 
   - All segments in a segment container must have the same length when rendered
   - A segment container contains by default all the AWG channels you have defined in your setup
   - All elements in the segment container must contain times that are :math:`\geq 0`.

.. _seq:

Sequences
---------
Sequences are a collection of segment containers. Basically it means just concatenate different segment containers.

One could for example make a sequence of three segment containers, on that would contain also the pulse data for initialization, one for manipulation and one for readout of the qubits.
When executing the experiment, you would tell the the AWG to play this sequence of elements. 