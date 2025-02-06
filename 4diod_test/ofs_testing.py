import serial
import csv
import re
import time

# UART Configuration
UART_PORT = "COM3"  # Replace with your UART port (e.g., '/dev/ttyUSB0')
BAUD_RATE = 115200  # Match the baud rate with your microcontroller's configuration

# CSV Output File
CSV_FILE = "18m_aligned_live_16%.csv"

# Regex pattern to extract combined data
UART_DATA_PATTERN = (
    r"SETPOINT_X: (\d+), SETPOINT_Y: (\d+), "
    r"INE: (\d+), INW: (\d+), ISE: (\d+), ISW: (\d+), "
    r"ONE: (\d+), ONW: (\d+), OSE: (\d+), OSW: (\d+), CIR: (\d+)"
)

def parse_uart_message(message):
    match = re.search(UART_DATA_PATTERN, message)
    if match:
        return [int(value) for value in match.groups()]
    return None

def wait_for_data(ser):
    """Wait for the UART to start sending data."""
    print("Waiting for data from the microcontroller...")
    while True:
        if ser.in_waiting > 0:
            print("Data detected!")
            return
        time.sleep(0.1)  # Prevent busy-waiting

def main():
    # Open UART connection
    with serial.Serial(UART_PORT, BAUD_RATE, timeout=1) as ser:
        print(f"Connected to UART on {UART_PORT} at {BAUD_RATE} baud.")
        
        # Wait for the microcontroller to send data
        wait_for_data(ser)
        
        # Open CSV file for writing
        with open(CSV_FILE, mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            
            # Write the header row
            header = [
                "Setpoint_X", "Setpoint_Y",
                "INE", "ISW", "ISE", "INW",
                "ONE", "ONW", "OSE", "OSW", "CIR"
            ]
            csv_writer.writerow(header)

            while True:
                try:
                    # Read a line from UART
                    line = ser.readline().decode("utf-8").strip()
                    print(f"Raw line received: {line}")  # Debugging raw data
                    
                    # Attempt to parse the UART message
                    parsed_data = parse_uart_message(line)
                    if parsed_data:
                        print(f"Captured: {parsed_data}")
                        
                        # Write to CSV
                        csv_writer.writerow(parsed_data)
                    else:
                        print("No match for the received line.")  # Debugging failed matches

                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                except Exception as e:
                    print(f"Error: {e}")


if __name__ == "__main__":
    main()
