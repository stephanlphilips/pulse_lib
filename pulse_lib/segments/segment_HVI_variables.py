"""
Marker implementation.
"""

from pulse_lib.segments.segment_base import segment_base, last_edited
from pulse_lib.segments.segment_IQ import segment_IQ
from pulse_lib.segments.utility.data_handling_functions import loop_controller, loop_controller_post_processing
from pulse_lib.segments.data_classes.data_HVI_variables import marker_HVI_variable
from dataclasses import dataclass


class segment_HVI_variables(segment_base):
	"""docstring for segment_HVI_variables"""
	def __init__(self, name):
		"""
		init marker object

		Args:
			name (str) : name of the marker channel.
			marker_voltage (double) : voltage in mV to output when the marker is on (default is 1V),
		"""
		super(segment_HVI_variables, self).__init__(name, marker_HVI_variable(), segment_type = 'render')
		self._data_hvi_variable = self.data

	@last_edited
	@loop_controller
	def _add_HVI_variable(self, name, value, time):
		"""
		add time for the marker.

		Args:
			name (str) : name of the variable
			value (double) : value to assign to the variable
			time (bool) : if the value is a timestamp (determines behaviour when the variable is used in a sequence) (coresponding to a master clock)
		"""
		self.data_tmp.add_HVI_marker(name, value, time)

		return self.data_tmp

	@loop_controller_post_processing
	def _add_global_time_shift(self, time):
		'''
		Time to shift teh current sequence compaered to the clock of the segment.
		'''
		return self.data_tmp._shift_all_time(time)

	def __copy__(self):
		cpy = segment_HVI_variables(self.name)
		return self._copy(cpy)
		
if __name__ == '__main__':
 
	import matplotlib.pyplot as plt
	s1 = segment_HVI_variables("marker_1")
	s1._add_HVI_variable("test",100, True)

	s2 = segment_HVI_variables("marker_2")
	s1._add_HVI_variable("test2",150, True)
	s1._add_global_time_shift(100)
	# s3 = s1+s2
	# s3.name = "marker_3"
	
	print(s1.data[0])
	print(s1.pulse_data_all[0])	
	s1.append(s2)
	# print(s1.data[0])

	# print("total time of marker sequence (ns) : ",s1.total_time)
	# print("maximal voltage (mV) : ", s1.v_max([0]))
	# print("integral of data (always 0 for markers) : " ,s1.integrate([0]))
	# print("memory location of render data : " ,s1.pulse_data_all)
	# print("Last edited : " ,s1.last_edit)
	# print("Shape :" ,s1.shape)

	# s1.plot_segment()
	# s2.plot_segment()
	# s3.plot_segment()
	# plt.legend()
	# plt.show()