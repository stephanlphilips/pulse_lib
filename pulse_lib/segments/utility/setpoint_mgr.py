from dataclasses import dataclass
import copy
import numpy as np

"""
file that constains a few classes that do automatric unit management.
The idea of these modules is to provide the units that qcodes needs for plotting.
The setpoint_mgr is meant to run in segment object and can be conbined easily
for all the channels on segment_contianer level and further into the sequence.
"""

class setpoint_mgr():
    """docstring for setpoint_mgr"""
    def __init__(self):
        self._setpoints = dict()

    def __add__(self, other):
        """
        add other setpoint /setpoint_mgr to this object

        Args:
            other (setpoint/setpoint_mgr) : object with setpoints.

        Returns:
            self (setpoint_mgr)
        """

        output = setpoint_mgr()
        output._setpoints = copy.copy(self._setpoints)

        if isinstance(other, setpoint):
            if other.axis in self._setpoints.keys():
                output._setpoints[other.axis] += other
            else:
                output._setpoints.update({other.axis : other})

        elif isinstance(other, self.__class__):
            for setpnt in other:
                output += setpnt
        else:
            raise ValueError("setpoint_mgr does not support counting up of the type {}. Please use the setpoint_mgr or setpoint type".format(type(other)))

        return output

    def __str__(self):
        content = "\rSetpoint_mgr class. Contained data:\r\r"

        for key in sorted(self._setpoints.keys()):
            content += "axis : {}\r".format(key)
            content += self._setpoints[key].__str__()
            content += "\r\r"

        return content

    def __getitem__(self, axis):
        """
        get setpoint data for a certain axis
        """
        set_point = self._setpoints[axis]

        return set_point

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self):
            setpoint = list(self._setpoints.values())[self.n]
            self.n += 1
            return setpoint
        else:
            raise StopIteration

    def __len__(self):
        return len(self._setpoints)

    @property
    def labels(self):
        labels = tuple()
        for key in sorted(self._setpoints.keys()):
            if len(self._setpoints[key].label) >= 1:
                good_labels = np.where(np.asarray(self._setpoints[key].label) != 'no label')[0]
                if len(good_labels)>=1:
                    labels += (self._setpoints[key].label[good_labels[0]] , )
                else:
                    labels += (self._setpoints[key].label[0] , )

            else:
                labels += ("No_label_defined", )
        return labels

    @property
    def units(self):
        units = tuple()
        for key in sorted(self._setpoints.keys()):
            if len(self._setpoints[key].unit) >= 1:
                good_units = np.where(np.asarray(self._setpoints[key].unit) != 'no label')[0]
                if len(good_units)>=1:
                    units += (self._setpoints[key].unit[good_units[0]] , )
                else:
                    units += (self._setpoints[key].unit[0] , )
            else:
                units += ("a.u.", )
        return units

    @property
    def setpoints(self):
        setpnts = tuple()
        for key in sorted(self._setpoints.keys()):
            if len(self._setpoints[key].setpoint) >= 1:
                good_labels = np.where(np.asarray(self._setpoints[key].label) != 'no label')[0]
                if len(good_labels)>=1:
                    setpnts += (self._setpoints[key].setpoint[good_labels[0]] , )
                else:
                    setpnts += (self._setpoints[key].setpoint[0] , )
            else:
                setpnts += (None, )
        return setpnts


@dataclass
class setpoint():
    axis : int
    label : tuple = tuple()
    unit : tuple = tuple()
    setpoint : tuple = tuple()

    def __add__(self, other):
        if self.axis != other.axis:
            raise ValueError("Counting of two setpoint variables on two axis. This is not allowed.")

        my_sum = setpoint(self.axis)
        # prioritize setpoints of variables where units/labels are defined.
        if len(other.label) == 1 or len(other.unit) == 1:
            my_sum.setpoint = other.setpoint + self.setpoint
            my_sum.label = other.label + self.label
            my_sum.unit = other.unit + self.unit
        else:
            my_sum.setpoint = self.setpoint + other.setpoint
            my_sum.label = self.label + other.label
            my_sum.unit = self.unit + other.unit

        return my_sum

if __name__ == '__main__':
    b = setpoint(1)
    print(b.axis)
    b.label = ('Voltage1',)
    b.unit = ('V',)
    a = setpoint(0)
    a.label = ('Voltage',)

    # c = a + b
    # print(c)

    setpoint_manager = setpoint_mgr()

    setpoint_manager += a
    setpoint_manager += a
    setpoint_manager += b
    # print(a, b)

    setpoint_manager += setpoint_manager

    print(setpoint_manager)
    print(setpoint_manager.labels)
    print(setpoint_manager.units)
    print(setpoint_manager.setpoints)