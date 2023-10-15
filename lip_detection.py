import cv2
import numpy as np

# video file
cap = cv2.VideoCapture("C:/Users/mohit/MyFaceDetection/new_output_video.mp4")

while True:

    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if no more frames are available

    # Convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([0, 100, 100])
    upper_bound = np.array([10, 255, 255])

    # Create a binary mask
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

    # Apply morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around detected lips
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with detected lips
    cv2.imshow('Detected Lips', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
