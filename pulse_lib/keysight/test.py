import numpy as np
import time

# import test
start = time.time()
for i in range(100):
	t = np.linspace(0,0.,500000)
stop=time.time()
print((stop-start)/100)