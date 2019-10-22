import test
import matplotlib.pyplot as plt

p1 = test.base_pulse_element(0,80,50,50)
p2 = test.base_pulse_element(10,-1,50,50)
p3 = test.base_pulse_element(20,-1,50,50)

seq = test.pulse_data_single_sequence()
seq.add_pulse(p1)
seq.add_pulse(p2)
seq.add_pulse(p3)

plt.plot(*seq.pulse_data)
plt.show()