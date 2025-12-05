import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import fastf1 as ff1

year = 2025
wknd = 9
ses = 'R'
driver = 'HAM'
colormap = mpl.cm.plasma

session = ff1.get_session(year, wknd, ses)
session.load()
weekend = session.event

lap = session.laps.pick_drivers(driver).pick_fastest()

# Position data
pos = lap.get_pos_data()
x = pos['X']
y = pos['Y']

# Telemetry data
tel = lap.get_car_data()
speed = tel['Speed']

points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

fig, ax = plt.subplots(figsize=(12, 6.75))
fig.suptitle(f'{weekend.name} {year} - {driver} - Speed', size=24, y=0.97)

ax.plot(x, y, color='black', linestyle='-', linewidth=16, zorder=0)

norm = plt.Normalize(speed.min(), speed.max())
lc = LineCollection(segments, cmap=colormap, norm=norm, linestyle='-', linewidth=5)
lc.set_array(speed)
ax.add_collection(lc)

cbaxes = fig.add_axes([0.25, 0.05, 0.5, 0.05])
normlegend = mpl.colors.Normalize(vmin=speed.min(), vmax=speed.max())
mpl.colorbar.ColorbarBase(cbaxes, norm=normlegend, cmap=colormap,
                          orientation="horizontal", label="Speed (km/h)")

ax.axis('off')
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.12)
plt.show()
