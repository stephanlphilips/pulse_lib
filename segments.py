import numpy as np
import datetime
import matplotlib.pyplot as plt

class segment_container():
	'''
    Class containing all the single segments for for a series of channels.
	This is a organisational class.
    '''
	def __init__(self, name, channels):
		self.channels = channels
		self.name = name

		for i in self.channels:
			setattr(self, i, segment_single())

	def reset_time(self):
		maxtime = 0
		for i in self.channels:
			k = getattr(self, i)
			t = k.get_total_time()
			if t > maxtime:
				maxtime = t
		for i in self.channels:
			getattr(self, i).starttime = maxtime

	@property
	def total_time(self):
		time_segment = 0
		for i in self.channels:
			if time_segment <= getattr(self, i).total_time:
				time_segment = getattr(self, i).total_time

		return time_segment


def last_edited(f):
	def wrapper(*args):
		args[0].last_edit = datetime.datetime.now()
		return f(*args)
	return wrapper

class segment_single():
	def __init__(self):
		self.type = 'default'
		self.to_swing = False
		self.starttime = 0
		self.last_edit = datetime.datetime.now()
		self.my_pulse_data = np.zeros([1,2])
		self.last = None
		self.IQ_data = [] #todo later.

	@last_edited
	def add_pulse(self,array):
		'''
		format :: 
		[[t0, Amp0],[t1, Amp1]]
		'''
		arr = np.asarray(array)
		arr[:,0] = arr[:,0] + self.starttime 

		self.my_pulse_data = np.append(self.my_pulse_data,arr, axis=0)

	@last_edited
	def add_block(self,start,stop, amplitude):
		amp_0 = self.my_pulse_data[-1,1]
		pulse = [ [start, amp_0], [start,amplitude], [stop, amplitude], [stop, amp_0]]
		self.add_pulse(pulse)

	@last_edited
	def wait(self, wait):
		amp_0 = self.my_pulse_data[-1,1]
		t0 = self.my_pulse_data[-1,0]
		pulse = [wait+t0, amp_0]

	def repeat(self, number):
		return

	def add_np(self,start, array):
		return

	def add_IQ_pair():
		return

	def add():
		return
	
	@property
	def total_time(self,):
		return self.my_pulse_data[-1,0]

	def get_total_time(self):
		return self.my_pulse_data[-1,0]

	def get_sequence(self, voltage_range = None, off_set = None):
		'''
		Returns a numpy array that contains the points for each ns
		'''
		return None


	def _generate_sequence(self):
		# 1 make base sequence:
		t_tot = self.total_time

		times = np.linspace(0, int(t_tot-1), int(t_tot))
		my_sequence = np.empty([int(t_tot)])


		for i in range(0,len(self.my_pulse_data)-1):
			my_loc = np.where(times < self.my_pulse_data[i+1,0])[0]
			my_loc = np.where(times[my_loc] >= self.my_pulse_data[i,0])[0]

			if my_loc.size==0:
				continue;

			end_voltage = self.my_pulse_data[i,1] + (self.my_pulse_data[i+1,1] - self.my_pulse_data[i,1])*(times[my_loc[-1]] - self.my_pulse_data[i,0])/(self.my_pulse_data[i+1,0]- self.my_pulse_data[i,0]);

			start_stop_values = np.linspace(self.my_pulse_data[i,1],end_voltage,my_loc.size);
			my_sequence[my_loc] = start_stop_values;

		return times, my_sequence

	def plot_sequence(self):
		x,y = self._generate_sequence()
		plt.plot(x,y)
		plt.show()
		
class marker_single():
	def __init__(self):
		self.type = 'default'
		self.swing = False
		self.latest_time = 0

		self.my_pulse_data = np.zeros([1,2])

	def add(self, start, stop):
		self.my_pulse_data = np,append(self.my_pulse_data, [[start,0],[start,1],[stop,1],[stop,0]])
