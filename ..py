import dlib
import cv2

# Load the pre-trained facial landmark model
predictor_path = "C:/Users/mohit/MyFaceDetection/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Create an OpenCV video capture object
video_capture = cv2.VideoCapture("C:/Users/mohit/MyFaceDetection/new_output_video.mp4")

while True:
    # Read a frame from the video
    success, frame = video_capture.read()

    if not success:
        break

    # Convert the frame to grayscale (Dlib works with grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Loop through the detected faces and find facial landmarks
    for face in faces:
        landmarks = predictor(gray, face)

        # Loop through the 68 facial landmarks and extract their x, y coordinates
        for i in range(0, 68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y

            # Draw a point on the frame at each facial landmark
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    # Display the frame with facial landmarks
    cv2.imshow("Facial Landmarks", frame)

    # Check for a key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
