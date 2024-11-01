import numpy as np
import cv2
import pickle

# Load calibration results
with open('calibration.pkl', 'rb') as f:
    cameraMatrix, dist = pickle.load(f)

# Function to capture click events
def click_event(event, x, y, flags, param):
    global points, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('image', frame)

        if len(points) == 2:
            # Calculate pixel distance
            pixel_dist = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
            print(f'Pixel Distance: {pixel_dist}')

            # Convert pixel coordinates to normalized coordinates
            points_np = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
            normalized_points = cv2.undistortPoints(points_np, cameraMatrix, dist, None, cameraMatrix)
            normalized_points = normalized_points.reshape(-1, 2)

            # Convert normalized coordinates to real-world coordinates
            square_size_mm = 50  # Size of a square in millimeters (adjust as per your calibration)
            real_coords = []
            for pt in normalized_points:
                real_coord = np.linalg.inv(cameraMatrix) @ np.array([pt[0], pt[1], 1])
                real_coords.append(real_coord[:2] * square_size_mm)

            # Calculate real-world distance
            real_dist = np.linalg.norm(real_coords[0] - real_coords[1])
            print(f'Real-World Distance (mm): {real_dist}')

            # Compare with actual measured distance
            actual_distance_mm = 50  # Adjust this based on your actual measurement
            error = abs(real_dist - actual_distance_mm)
            print(f'Actual Distance (mm): {actual_distance_mm}')
            print(f'Error (mm): {error}')

# Initialize global variables
points = []

# Open a connection to the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    cv2.imshow('image', frame)
    cv2.setMouseCallback('image', click_event)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
