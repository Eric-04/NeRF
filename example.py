import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

# Create initial values for frequency and amplitude
initial_frequency = 1.0
initial_amplitude = 1.0

# Generate x values (time) from 0 to 10 with 0.1 step
x = np.arange(0, 10, 0.1)

# Create initial sine wave with initial frequency and amplitude
y = initial_amplitude * np.sin(2 * np.pi * initial_frequency * x)

# Create the plot
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)
l, = plt.plot(x, y, lw=2)

# Create sliders
ax_frequency = plt.axes([0.1, 0.1, 0.65, 0.03])
ax_amplitude = plt.axes([0.1, 0.15, 0.65, 0.03])

s_frequency = Slider(ax_frequency, 'Frequency', 0.1, 10.0, valinit=initial_frequency)
s_amplitude = Slider(ax_amplitude, 'Amplitude', 0.1, 5.0, valinit=initial_amplitude)

# Update function
def update(val):
    frequency = s_frequency.val
    amplitude = s_amplitude.val
    y = amplitude * np.sin(2 * np.pi * frequency * x)
    l.set_ydata(y)
    fig.canvas.draw_idle()

s_frequency.on_changed(update)
s_amplitude.on_changed(update)

plt.show()
