import serial, time

arduino = serial.Serial('COM6', 115200, timeout=.1)
time.sleep(1)

print("sent")
arduino.write("FUCK".encode())
    
        