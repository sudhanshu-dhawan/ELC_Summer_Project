import numpy as np
import cv2
import pickle
import serial as pyserial  # Using an alias for pyserial

# Load calibration results
with open('calibration.pkl', 'rb') as f:
    cameraMatrix, dist = pickle.load(f)

# Initialize the serial connection to MATLAB
ser = pyserial.Serial('COM9', 9600)

# Function to capture click event
def click_event(event, x, y, flags, param):
    global frame
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'Clicked Coordinates: {x}, {y}')
        
        # Convert pixel coordinates to normalized coordinates
        pixel_coords = np.array([[x, y]], dtype=np.float32)
        normalized_coords = cv2.undistortPoints(np.expand_dims(pixel_coords, axis=1), cameraMatrix, dist, None, cameraMatrix)
        normalized_coords = normalized_coords.flatten()

        # Assuming a plane parallel to the camera sensor (Z=0 in camera coordinate system)
        real_coords_homogeneous = np.array([normalized_coords[0], normalized_coords[1], 1])

        # Calculate real-world coordinates (in millimeters)
        # Projecting back to real-world coordinates assuming Z=0
        real_coords = np.linalg.inv(cameraMatrix) @ real_coords_homogeneous
        square_size_mm = 50  # Size of a square in millimeters (adjust as per your calibration)
        real_coords = real_coords[:2] * square_size_mm

        print(f'Real-World Coordinates (mm): {real_coords}')

        # Display coordinates on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        str_real_coords = f'{real_coords[0]:.2f}, {real_coords[1]:.2f} mm'
        cv2.putText(frame, str_real_coords, (x, y), font, 0.5, (0, 255, 0), 2)
        cv2.imshow('image', frame)

        # Send the coordinates to MATLAB via serial
        coord_str = f"{real_coords[0]/1000:.3f},{real_coords[1]/1000:.3f},{0.4}\n"  # Assuming Z=0.4 meters
        ser.write(coord_str.encode())

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
ser.close()

# Calibration code (for reference and future use)
def perform_calibration():
    import numpy as np
    import cv2 as cv
    import glob
    import pickle

    ################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

    # Chessboard and frame size settings
    chessboardSize = (7, 7)  # Number of inner corners per a chessboard row and column
    frameSize = (640, 480)   # Size of the images (width, height)

    # Termination criteria for corner refinement
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points based on chessboard size and square size (assuming square size in mm)
    square_size_mm = 50
    objp = np.zeros((np.prod(chessboardSize), 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2) * square_size_mm

    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Load images from folder
    images = glob.glob('images/*.png')

    for image in images:
        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

        # If corners are found, refine them and add to lists
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners (optional)
            cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(1000)  # Show image for 1 second (1000 ms)

    cv.destroyAllWindows()

    ############## CALIBRATION #######################################################

    # Perform camera calibration
    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

    # Save calibration results
    pickle.dump((cameraMatrix, dist), open("calibration.pkl", "wb"))
    pickle.dump(cameraMatrix, open("cameraMatrix.pkl", "wb"))
    pickle.dump(dist, open("dist.pkl", "wb"))

    # Print calibration results
    print("Camera matrix:\n", cameraMatrix)
    print("Distortion coefficients:\n", dist)

    ############## UNDISTORTION #####################################################

    # Example: Undistort an image
    img = cv.imread('img5.png')

    if img is not None:
        # Determine optimal new camera matrix and ROI
        h, w = img.shape[:2]
        newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

        # Undistort the image
        dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

        # Crop the image using the ROI
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        # Save the undistorted image
        cv.imwrite('caliResult.png', dst)

        # Display the undistorted image
        cv.imshow('Undistorted Image', dst)
        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print("Error: Failed to load image 'cali5.png'.")

    # Reprojection Error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("Total reprojection error: {}".format(mean_error / len(objpoints)))
