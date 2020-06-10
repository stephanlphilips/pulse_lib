"""
Marker implementation.
"""

from pulse_lib.segments.segment_base import segment_base, last_edited
from pulse_lib.segments.segment_IQ import segment_IQ
from pulse_lib.segments.utility.data_handling_functions import loop_controller
from pulse_lib.segments.data_classes.data_markers import marker_data
from dataclasses import dataclass

@dataclass
class reference_marker_4_IQ:
    IQ_channel_ptr : segment_IQ
    pre_delay : float
    post_delay : float


class segment_marker(segment_base):
    """docstring for segment_marker"""
    def __init__(self, name, HVI_variable_data = None, marker_voltage = 1500):
        """s
        init marker object
        Args:
            name (str) : name of the marker channel.
            HVI_variable_data (segment_HVI_variables) : segment used to keep variables that can be used in HVI.
            marker_voltage (double) : voltage in mV to output when the marker is on (default is 1V),
        """
        super(segment_marker, self).__init__(name, marker_data(marker_voltage), HVI_variable_data, segment_type = 'render')

    @last_edited
    @loop_controller
    def add_marker(self, start, stop):
        """
        add time for the marker.
        Args:
            start (double) : start time of the marker
            stop (double) : stop time of the marker
        """
        self.data_tmp.add_marker(start, stop)
        return self.data_tmp

    def add_reference_marker_IQ(self, IQ_channel_ptr, pre_delay, post_delay):
        self.references_markers.append(reference_marker_4_IQ(IQ_channel_ptr, pre_delay, post_delay))
    
    def __copy__(self):
        cpy = segment_marker(self.name)
        return self._copy(cpy)
        
if __name__ == '__main__':
 
    import matplotlib.pyplot as plt
    s1 = segment_marker("marker_1")
    s1.add_marker(50,100)

    s2 = segment_marker("marker_2")
    s2.add_marker(80,100)

    s3 = s1+s2
    s3.name = "marker_3"
    print(s3.data[0].my_marker_data)
    
    s1.append(s2)
    print(s1.data[0].my_marker_data)

    print("total time of marker sequence (ns) : ",s1.total_time)
    print("maximal voltage (mV) : ", s1.v_max([0]))
    print("integral of data (always 0 for markers) : " ,s1.integrate([0]))
    print("memory location of render data : " ,s1.pulse_data_all)
    print("Last edited : " ,s1.last_edit)
    print("Shape :" ,s1.shape)

    s1.plot_segment()
    s2.plot_segment()
    s3.plot_segment()
    plt.legend()
    plt.show()