import serial
import csv
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class SerialDataReader:
    def __init__(self):
        # Configure serial port
        self.ser = serial.Serial(
            port='COM3',
            baudrate=115200,
            timeout=1
        )
        
        # Initialize data storage
        self.data = {
            'G1': {'values': [], 'line': None},
            'G2': {'values': [], 'line': None},
            'G3': {'values': [], 'line': None},
            'G4': {'values': [], 'line': None}
        }
        
        # Setup CSV logging
        self.timestamp = datetime.now().strftime("%H%M%S%f")[:9]
        self.filename = f'sensor_data_{self.timestamp}.csv'
        self.csv_file = open(self.filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Timestamp', 'Group', 'Value1', 'Value2', 'Value3', 
                                'Value4', 'Value5', 'Value6', 'Value7', 'Value8'])
        
        # set flag for data available state
        self.is_data_avail = False

    def log_data(self):
        while(True):
            if self.ser.in_waiting:
                    # Read line from serial port
                    line = self.ser.readline().decode('utf-8').strip()
                    
                    if line.startswith('G'):
                        # record time stamp
                        current_time = datetime.now().strftime("%H%M%S%f")[:9]

                        # change data availability state if data was not available before
                        if not self.is_data_avail:
                            self.is_data_avail = True
                            # add a seperation line indicate start logging data
                            self.csv_writer.writerow([current_time,'s']+[0.0,0.0,0.0,0.0])

                        # Parse the data
                        group, values = line.split(': ')
                        value_list = [float(x) for x in values.split(',')]
                        
                        # Update data storage
                        self.data[group]['values'].append(value_list)
                        
                        # Write to CSV                    
                        self.csv_writer.writerow([current_time, group] + value_list)
                        
                        # Print data to console
                        print(f"Received: {line}")
                        

                    elif line.startswith("wait trigger") and self.is_data_avail:
                        # update data availability state
                        self.is_data_avail = False
                        # write to csv
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.csv_writer.writerow([current_time, 'W'] + [0.0,0.0,0.0,0.0])
                        print("wait for Odrive trigger")
                        time.sleep(0.1)

    def start(self):
        try:
            # Create animation
            self.log_data()
            
        except KeyboardInterrupt:
            print("\nStopping data collection...")
        except Exception as e:
                print(f"Error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        self.ser.close()
        self.csv_file.close()
        print(f"Data saved to {self.filename}")

if __name__ == "__main__":
    reader = SerialDataReader()
    reader.start()