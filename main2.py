import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=5,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Draw landmarks (optional for visual debugging)
mp_drawing = mp.solutions.drawing_utils 
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Start video capture
cap = cv2.VideoCapture(0)

# Lip landmark indices from MediaPipe (upper and lower lips)
UPPER_LIP = [13]
LOWER_LIP = [14]

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty frame.")
        continue

    # Convert image color to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame to detect face mesh
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get image dimensions
            h, w, _ = frame.shape

            # Get upper and lower lip y-coordinates
            upper_lip_y = int(face_landmarks.landmark[UPPER_LIP[0]].y * h)
            lower_lip_y = int(face_landmarks.landmark[LOWER_LIP[0]].y * h)

            # Calculate lip opening distance
            lip_distance = abs(lower_lip_y - upper_lip_y)

            # Set a threshold to detect speaking (tweak this value based on environment)
            speaking_threshold =5
            # Draw face landmarks (optioal)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

            # Get mouth center (for labeling)
            mouth_x = int(face_landmarks.landmark[0].x * w)
            mouth_y = int(face_landmarks.landmark[0].y * h)

            # Check if mouth is open (speaking)
            if lip_distance > speaking_threshold:
                cv2.putText(frame, 'Speaking', (mouth_x-20, mouth_y-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(frame, (mouth_x-50, mouth_y-50),
                              (mouth_x+50, mouth_y+50), (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Silent', (mouth_x-20, mouth_y-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.rectangle(frame, (mouth_x-50, mouth_y-50),
                              (mouth_x+50, mouth_y+50), (0, 0, 255), 2)

    # Show output window
    cv2.imshow('Real-time Speaker Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()