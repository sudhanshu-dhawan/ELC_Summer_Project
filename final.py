# import numpy as np
# import cv2
# import pickle

# # Load calibration results
# with open('calibration.pkl', 'rb') as f:
#     cameraMatrix, dist = pickle.load(f)

# # Function to capture click event
# def click_event(event, x, y, flags, param):
#     global frame
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(f'Clicked Coordinates: {x}, {y}')
        
#         # Convert pixel coordinates to normalized coordinates
#         pixel_coords = np.array([[x, y]], dtype=np.float32)
#         normalized_coords = cv2.undistortPoints(np.expand_dims(pixel_coords, axis=1), cameraMatrix, dist, None, cameraMatrix)
#         normalized_coords = normalized_coords.flatten()

#         # Assuming a plane parallel to the camera sensor (Z=0 in camera coordinate system)
#         real_coords_homogeneous = np.array([normalized_coords[0], normalized_coords[1], 1])

#         # Calculate real-world coordinates (in millimeters)
#         # Projecting back to real-world coordinates assuming Z=0
#         real_coords = np.linalg.inv(cameraMatrix) @ real_coords_homogeneous
#         square_size_mm = 50  # Size of a square in millimeters (adjust as per your calibration)
#         real_coords = real_coords[:2] * square_size_mm

#         print(f'Real-World Coordinates (mm): {real_coords}')

#         # Display coordinates on the frame
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         str_real_coords = f'{real_coords[0]:.2f}, {real_coords[1]:.2f} mm'
#         cv2.putText(frame, str_real_coords, (x, y), font, 0.5, (0, 255, 0), 2)
#         cv2.imshow('image', frame)

# # Open a connection to the webcam
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Could not read frame.")
#         break

#     cv2.imshow('image', frame)
#     cv2.setMouseCallback('image', click_event)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()   



import numpy as np
import cv2
import pickle

# Load calibration results
with open('calibration.pkl', 'rb') as f:
    cameraMatrix, dist = pickle.load(f)

# Global variable to track whether an image has been saved
image_saved = False
saved_image = None

# Function to capture click event
def click_event(event, x, y, flags, param):
    global frame, image_saved, saved_image
    if event == cv2.EVENT_LBUTTONDOWN:
        if not image_saved:
            # Save the current frame as an image on the first click
            cv2.imwrite('clicked_image.png', frame)
            saved_image = frame.copy()
            image_saved = True
            print("Image saved as 'clicked_image.png'")
        else:
            print(f'Clicked Coordinates on saved image: {x}, {y}')
            
            # Convert pixel coordinates to normalized coordinates
            pixel_coords = np.array([[x, y]], dtype=np.float32)
            normalized_coords = cv2.undistortPoints(np.expand_dims(pixel_coords, axis=1), cameraMatrix, dist, None, cameraMatrix)
            normalized_coords = normalized_coords.flatten()

            # Assuming a plane parallel to the camera sensor (Z=0 in camera coordinate system)
            real_coords_homogeneous = np.array([normalized_coords[0], normalized_coords[1], 1])

            # Calculate real-world coordinates (in millimeters)
            square_size_mm = 50  # Size of a square in millimeters (adjust as per your calibration)
            real_coords = np.linalg.inv(cameraMatrix) @ real_coords_homogeneous
            real_coords = real_coords[:2] * square_size_mm

            print(f'Real-World Coordinates (mm): {real_coords}')

            # Display coordinates on the saved image
            font = cv2.FONT_HERSHEY_SIMPLEX
            str_real_coords = f'{real_coords[0]:.2f}, {real_coords[1]:.2f} mm'
            cv2.putText(saved_image, str_real_coords, (x, y), font, 0.5, (0, 255, 0), 2)
            cv2.imshow('saved_image', saved_image)

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

    if not image_saved:
        cv2.imshow('image', frame)
        cv2.setMouseCallback('image', click_event)
    else:
        cv2.imshow('saved_image', saved_image)
        cv2.setMouseCallback('saved_image', click_event)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
