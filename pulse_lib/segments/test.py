import numpy as np
import pulse_lib.segments.data_classes_markers as mk

t = mk.marker_data()
t.add_marker(10,20)
t.add_marker(30,40)
# t.add_marker(60,1000)
# t.add_marker(400,1000)
# t.slice_time(150,500)
# t.print_all()

# newt = t.append(t, None)

print(t.get_vmin())