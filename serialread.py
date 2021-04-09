import serial
port1 = serial.Serial('COM7', baudrate=115200, timeout=1)
port2 = serial.Serial('COM6', baudrate=115200, timeout=1)

while True:
    line1 = port1.readline()
    line2 = port2.readline()
    line1 = line1.decode(encoding='utf8')
    line2 = line2.decode(encoding='utf8')
    if len(line1) < 10 or len(line2) < 10: continue
    line1 = list(map(int, line1.split(",")))
    line2 = list(map(int, line2.split(",")))
    if line1[0] % 2 == 1:
        line1 , line2 = line2, line1

    res = line1[3:11] + line2[3:11]
    print(res)


