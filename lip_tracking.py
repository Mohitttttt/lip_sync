import cv2

#video file
cap = cv2.VideoCapture("C:/Users/mohit/MyFaceDetection/new_output_video.mp4", cv2.CAP_FFMPEG)


#MOSSE tracker
tracker = cv2.legacy_TrackerMOSSE.create()

ret, frame = cap.read()
bbox = cv2.selectROI(frame, False)
tracker.init(frame, bbox)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # Update the tracker
    ret, bbox = tracker.update(frame)

    if ret:
        # Draw bounding box around the tracked lips
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with the tracked lips
    cv2.imshow('Lip Tracking (MOSSE)', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
