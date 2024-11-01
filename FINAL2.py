import numpy as np
import cv2 as cv
import pickle
from spatialmath import SE3
from spatialmath.base import transl, trotx
import math
import serial

# Load calibration results
with open('calibration.pkl', 'rb') as f:
    cameraMatrix, dist = pickle.load(f)

# Initialize the camera
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize the serial connection to Arduino or MATLAB
ser = serial.Serial('COM9', 9600)  # Adjust COM port and baud rate as needed

# Define DH parameters and end-effector pose
dh = np.array([
    [0, 0.22, 0, -np.pi/2],
    [0, 0, 0.22, 0],
    [np.pi/2, 0, 0, np.pi/2],
    [0, 0.22, 0, -np.pi/2],
    [-np.pi/2, 0, 0.09, 0]
])

T = transl(0.3, 0.2, 0.4) * trotx(np.pi/2)  # End-effector pose (adjust as needed)

# Function to handle click event and calculate real-world coordinates
def click_event(event, x, y, flags, param):
    global frame
    if event == cv.EVENT_LBUTTONDOWN:
        print(f'Clicked Coordinates: {x}, {y}')
        
        # Convert pixel coordinates to real-world coordinates
        pixel_coords = np.array([[x, y]], dtype=np.float32)
        normalized_coords = cv.undistortPoints(np.expand_dims(pixel_coords, axis=1), cameraMatrix, dist, None, cameraMatrix)
        normalized_coords = normalized_coords.flatten()

        # Assuming a plane parallel to the camera sensor (Z=0 in camera coordinate system)
        real_coords_homogeneous = np.array([normalized_coords[0], normalized_coords[1], 1])

        # Calculate real-world coordinates (in meters, assuming square_size_mm is in millimeters)
        square_size_mm = 50  # Size of a square in millimeters (adjust as per your calibration)
        real_coords = np.linalg.inv(cameraMatrix) @ real_coords_homogeneous
        real_coords = real_coords[:2] * square_size_mm / 1000.0  # Convert to meters

        print(f'Real-World Coordinates (meters): {real_coords}')

        # Calculate inverse kinematics
        try:
            robot = SE3(dh)
            q = robot.ikine(T)
            print(f'Joint Angles (degrees): {np.degrees(q.q)}')

            # Send joint angles to Arduino or MATLAB
            coord_str = ','.join(map(str, np.degrees(q.q))) + '\n'
            ser.write(coord_str.encode())
        except ValueError as e:
            print(f'Error: {e}')

        # Display coordinates on the frame
        font = cv.FONT_HERSHEY_SIMPLEX
        str_real_coords = f'{real_coords[0]:.2f}, {real_coords[1]:.2f} m'
        cv.putText(frame, str_real_coords, (x, y), font, 0.5, (0, 255, 0), 2)
        cv.imshow('image', frame)

# Main loop to capture frames and handle events
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    cv.imshow('image', frame)
    cv.setMouseCallback('image', click_event)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv.destroyAllWindows()
ser.close()
