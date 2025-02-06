import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import re

# Serial Port Configuration
SERIAL_PORT = 'COM3'  # serial port
BAUD_RATE = 115200      

# Set up the serial connection
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0)
print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud")

# Data Buffer
BUFFER_SIZE = 500    # Number of data points to display on the plot
x_data = deque(maxlen=BUFFER_SIZE)  # X-axis data (time or count)
y_data_ine = deque(maxlen=BUFFER_SIZE)  # Y-axis data for INE
y_data_inw = deque(maxlen=BUFFER_SIZE)  # Y-axis data for INW
y_data_ise = deque(maxlen=BUFFER_SIZE)  # Y-axis data for ISE
y_data_isw = deque(maxlen=BUFFER_SIZE)  # Y-axis data for ISW

# Initialize the figure and axis
fig, ax = plt.subplots()
line_ine, = ax.plot([], [], lw=2, label='INE')
line_inw, = ax.plot([], [], lw=2, label='INW')
line_ise, = ax.plot([], [], lw=2, label='ISE')
line_isw, = ax.plot([], [], lw=2, label='ISW')
ax.set_xlim(0, BUFFER_SIZE)
ax.set_ylim(0, 11000)  
ax.set_xlabel('Data Points')
ax.set_ylabel('Value')
ax.set_title('Real-Time Alignment')
ax.legend()

# Update function for animation
def update(frame):
    try:
        # Read a line of data from the serial port
        line_data = ser.readline().decode('utf-8').strip()
        if line_data:
            # Use regular expressions to extract values for INE, INW, ISE, ISW
            match = re.search(r"INE: (\d+), INW: (\d+), ISE: (\d+), ISW: (\d+)", line_data)
            if match:
                ine = int(match.group(1))
                inw = int(match.group(2))
                ise = int(match.group(3))
                isw = int(match.group(4))

                # Append data to respective buffers
                y_data_ine.append(ine)
                y_data_inw.append(inw)
                y_data_ise.append(ise)
                y_data_isw.append(isw)
                x_data.append(len(y_data_ine))

                # Update the line data
                line_ine.set_data(range(len(y_data_ine)), y_data_ine)
                line_inw.set_data(range(len(y_data_inw)), y_data_inw)
                line_ise.set_data(range(len(y_data_ise)), y_data_ise)
                line_isw.set_data(range(len(y_data_isw)), y_data_isw)

                # Adjust x-axis limits dynamically
                ax.set_xlim(0, len(y_data_ine))
    except Exception as e:
        print(f"Error: {e}")

    return line_ine, line_inw, line_ise, line_isw

# Animation

ani = animation.FuncAnimation(fig, update, blit=True, interval=1)

# Show the plot
plt.show()

# Close serial connection on exit
ser.close()
