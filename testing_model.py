import serial
import numpy as np
import matplotlib.pyplot as plt
import cmath
from YDLidarGS2 import YDLidar_GS2 as ydlidar
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the model
model = load_model('./ydlidargs2.h5')

# Create Scaler
scaler = StandardScaler()

# Test Model Function
def test_model(data):
    arr_data = np.array(data)
    scaled_data = scaler.transform(arr_data.reshape(1,-1))
    prediction = model.predict(scaled_data)
    return prediction

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='polar')
ax.set_theta_direction(-1)
ax.set_theta_zero_location('N')
ax.set_title('lidar (exit: Key E)',fontsize=18)
plt.connect('key_press_event', lambda event: exit(1) if event.key == 'e' else None)

ydl = ydlidar(port='/dev/ttyUSB0')

ydl.startlidar()


counter = 0
while counter < 2000:
    rt = ydl.getData()
    ltheta = rt.thetas[:80]
    rtheta = rt.thetas[80:]

    # Start Prediction
    guess = test_model(rt.distance)
    print(guess)
    
    if('line' in locals()):
        line.remove()
    if('line2' in locals()):
        line2.remove()
    line = ax.scatter(ltheta, rt.distance[:80], c="red", s=5)
    line2 = ax.scatter(rtheta, rt.distance[80:], c="blue", s=5)
    ax.set_theta_offset(cmath.pi / 2)
    ax.set_theta_offset(cmath.pi / 2)
    plt.pause(0.00001)
    counter+=1
    


ydl.ser.write([0xA5,0xA5,0xA5,0xA5,0x00,0x64,0x00,0x00,0x64])
ydl.ser.close()


