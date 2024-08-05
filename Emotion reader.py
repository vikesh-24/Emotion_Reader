from facial_emotion_recognition import EmotionRecognition
import cv2

# Initialize the emotion recognition model
er = EmotionRecognition(device='cpu')

# Initialize the webcam (0 for the default camera, 1 for an external camera if connected)
cam = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cam.isOpened():
    print("Error: Could not open video capture")
else:
    while True:
        # Capture a frame
        success, frame = cam.read()

        if not success:
            print("Error: Could not read frame")
            break

        # Recognize emotion in the frame
        frame = er.recognise_emotion(frame, return_type='BGR')

        # Display the frame
        cv2.imshow("Frame", frame)

        # Check for the ESC key to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Release the camera and destroy all windows
cam.release()
cv2.destroyAllWindows()
