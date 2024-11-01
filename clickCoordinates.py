import numpy as np
import cv2

capture = False

def click_event_capture(event, x, y, flags, param):
    global capture, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        capture = True
        cv2.imwrite('captured_image.jpg', frame)
        # print(f'Image captured at coordinates: {x}, {y}')
        cv2.imshow('captured_image', frame)
        cv2.setMouseCallback('captured_image', click_event_coords)

def click_event_coords(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(f'Coordinates: {x}, {y}')
        font = cv2.FONT_HERSHEY_SIMPLEX
        strxy = str(x) + ',' + str(y)
        cv2.putText(frame, strxy, (x, y), font, 1, (0, 255, 0), 2)
        cv2.imshow('captured_image', frame)

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    if not capture:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the frame
        cv2.imshow('image', frame)

        # Set the mouse callback function to capture image on click
        cv2.setMouseCallback('image', click_event_capture)
    else:
        # Set the mouse callback function to get coordinates on double click
        cv2.setMouseCallback('captured_image', click_event_coords)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
