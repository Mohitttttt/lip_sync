import cv2

# Load the cascade for face detection
face_cascade = cv2.CascadeClassifier('path_to_haarcascade.xml')  # Replace with the correct path

# Open the video file
cap = cv2.VideoCapture("C:/Users/mohit/MyFaceDetection/new_output_video.mp4")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Lip Analysis', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
