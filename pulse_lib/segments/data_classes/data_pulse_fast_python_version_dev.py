from dataclasses import dataclass
import numpy as np
import copy
import numbers

@dataclass
class base_pulse_element:
	start : float
	stop : float
	v_start : float
	v_stop : float

	index_start : int = 0
	index_stop : int = 0


class pulse_data_single_sequence():
	# cdef list localdata = []
	# cdef bool re_render = True
	# cdef np.ndarray[ndim = 1, dtype = np.double] voltage_data
	# cdef np.ndarray[ndim = 1, dtype = np.double] 
	def __init__(self):
		self.localdata = [base_pulse_element(0,1e-9,0,0)]
		self.re_render = True
		self.voltage_data = np.array([0])
		self._total_time = 0
	def add_pulse(self, pulse):
		self.localdata.append(pulse)
		self.re_render = True
		if self._total_time < pulse.stop:
			self._total_time = pulse.stop
	
	def append(self, other):
		"""
		append other pulse object to this one
		"""
		time_shift = self.t_tot
		other.shift_time(time_shift)
		self.localdata += copy.copy(other.localdata)
		other.shift_time(-time_shift)

	def shift_time(self, time):
		for i in self.localdata:
			i.start += time
			i.stop += time
		self.re_render = True

	def __add__(self, other):
		pulse_data = copy.copy(self)
		if isinstance(other, pulse_data_single_sequence):
			pulse_data.localdata += copy.copy(other.localdata)
		else: #assume float, double or int.
			pulse_data.localdata.append(base_pulse_element(0,-1, other, other))

		pulse_data.re_render = True

		return pulse_data

	def __sub__(self, other):
		return self + other*(-1)

	def __mul__(self, other):
		pulse_data = copy.copy(self)

		if isinstance(other,numbers.Number):
			for i in self.localdata:
				i.v_start *= other
				i.v_stop *= other

		self.re_render = True

		return pulse_data


	def __local_render(self):
		time_steps = []

		t_step = 1e-9
		for i in self.localdata:
			time_steps.append(i.start)
			time_steps.append(i.start+t_step)
			time_steps.append(i.stop-t_step)
			time_steps.append(i.stop)

		time_steps_np, index_inverse = np.unique(np.array(time_steps), return_inverse=True)
		print(time_steps[len(time_steps) -30 :])
		print(index_inverse[len(index_inverse) -30 :])
		for i in range(int(len(index_inverse)/4)):
			self.localdata[i].index_start = index_inverse[i*4+1]
			self.localdata[i].index_stop = index_inverse[i*4+2]


		voltage_data = np.zeros([len(time_steps_np)])


		for i in self.localdata:
			delta_v = i.v_stop-i.v_start
			min_time = time_steps_np[i.index_start]
			max_time = time_steps_np[i.index_stop]
			rescaler = delta_v/(max_time-min_time)

			for j in range(i.index_start, i.index_stop+1):
				print(i.v_start)
				print(rescaler)
				print(j)

				voltage_data[j] += i.v_start + (time_steps_np[j] - min_time)*rescaler
				print(time_steps_np[len(time_steps_np)-5:])
				print(voltage_data[len(time_steps_np)-5:])

		# cleaning up the array (remove 1e-10 spacings between data points):
		new_data_time = []
		new_data_voltage = []
		print(time_steps_np[len(time_steps_np)-30:])
		print(voltage_data[len(time_steps_np)-30:])

		new_data_time.append(time_steps_np[0])
		new_data_voltage.append(voltage_data[0])

		i = 1
		while( i < len(time_steps_np)-1):
			if time_steps_np[i+1] - time_steps_np[i] < t_step*2 and time_steps_np[i] - time_steps_np[i-1] < t_step*2:
				i+=1

			new_data_time.append(time_steps_np[i])
			new_data_voltage.append(voltage_data[i])
			i+=1
		if i < len(time_steps_np):
			new_data_time.append(time_steps_np[i]) 
			new_data_voltage.append(voltage_data[i])


		return new_data_time, new_data_voltage

	@property
	def pulse_data(self):
		if self.re_render == True:
			self.time_data, self.voltage_data = self.__local_render()
		return (self.time_data, self.voltage_data)

	@property
	def total_time(self):
		return self._total_time

	def v_max(self):
		self.pulse_data
		return np.max(voltage_data)

	def v_min(self):
		self.pulse_data
		return np.min(voltage_data)

import data_pulse_core as dp_fast

t1 = dp_fast.pulse_data_single_sequence()

p = dp_fast.base_pulse_element(0,-1,50,50)

t2 = dp_fast.pulse_data_single_sequence()

t2.add_pulse(p)
# import time
# t1 = time.time()
for i in range(10):
	p = dp_fast.base_pulse_element(t2.total_time,t2.total_time + 10,i,i)
	t2.add_pulse(p)
# t2 = time.time()
# print(t2-t1)


# t1.append(t2)
# t1 = t
# t2.append(t1)
# t1.slice_time(50,2000)
print(t1.total_time)
a, b = t2.pulse_data
# print(np.array(a)[:],np.array(b)[:])
# print("render time python",te-t0)
import matplotlib.pyplot as plt
plt.plot(a, b)
plt.show()
