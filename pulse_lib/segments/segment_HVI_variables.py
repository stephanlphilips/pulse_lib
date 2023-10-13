"""
Marker implementation.
"""

from pulse_lib.segments.segment_base import segment_base
from pulse_lib.segments.utility.data_handling_functions import loop_controller
from pulse_lib.segments.data_classes.data_HVI_variables import marker_HVI_variable


class segment_HVI_variables(segment_base):
    """docstring for segment_HVI_variables"""

    def __init__(self, name):
        """
        init marker object

        Args:
            name (str) : name of the marker channel.
        """
        super(segment_HVI_variables, self).__init__(name, marker_HVI_variable(), segment_type='render')

    @loop_controller
    def _add_HVI_variable(self, name, value):
        """
        add time for the marker.

        Args:
            name (str) : name of the variable
            value (double) : value to assign to the variable
        """
        self.data_tmp.add_HVI_marker(name, value)

        return self.data_tmp

    def __copy__(self):
        cpy = segment_HVI_variables(self.name)
        return self._copy(cpy)
