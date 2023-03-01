# export profile from VR-6000 software as DXF
# import DXF file and plot it
import matplotlib.pyplot as plt
import ezdxf

# Load the DXF files and add the vectors to the plot
doc = ezdxf.readfile('test-scan.dxf')
msp = doc.modelspace()

fig = plt.figure(figsize=(14, 5))
ax = fig.add_subplot(111)

max_x, min_x = 0, 0 # for setting plot limits
for e in msp:
    if e.dxftype() == 'LINE':
        if e.dxf.start[0] > -5 and e.dxf.start[0] < 5:
            # ignores all lines not part of seal
            # i.e. the scanning fixture
            continue
        x = [e.dxf.start[0], e.dxf.end[0]]
        y = [e.dxf.start[1], e.dxf.end[1]]
        if x[1] > max_x:
            max_x = x[1]
        if x[0] < min_x:
            min_x = x[0]
        ax.plot(x, y, color='black')

padding = 5
ax.set_xlim((min_x - padding), (max_x + padding))

plt.axis('scaled') # preserves scale of scan
plt.tight_layout()
plt.show()

