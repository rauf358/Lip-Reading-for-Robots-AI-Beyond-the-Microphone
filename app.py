import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=5,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Drawing specs (optional)
mp_drawing = mp.solutions.drawing_utils 
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Start video capture
cap = cv2.VideoCapture(0)

# Lip landmark indices (more points for better accuracy)
UPPER_LIP = [13, 312, 82]     # Top center and edges
LOWER_LIP = [14, 317, 87]     # Bottom center and edges

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty frame.")
        continue

    # Convert image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Calculate average Y for upper and lower lips
            upper_avg = np.mean([face_landmarks.landmark[i].y * h for i in UPPER_LIP])
            lower_avg = np.mean([face_landmarks.landmark[i].y * h for i in LOWER_LIP])
            lip_distance = abs(lower_avg - upper_avg)

            # Estimate face height using forehead to chin (landmark 10 to 152)
            top_y = int(face_landmarks.landmark[10].y * h)
            bottom_y = int(face_landmarks.landmark[152].y * h)
            face_height = abs(bottom_y - top_y)

            # Dynamic speaking threshold (tweak multiplier as needed)
            speaking_threshold = face_height * 0.03

            # Draw face mesh (optional)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

            # Get mouth center (landmark 0 for simplicity)
            mouth_x = int(face_landmarks.landmark[0].x * w)
            mouth_y = int(face_landmarks.landmark[0].y * h)

            # Check if speaking
            if lip_distance > speaking_threshold:
                cv2.putText(frame, 'Speaking', (mouth_x - 30, mouth_y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(frame, (mouth_x - 50, mouth_y - 50),
                              (mouth_x + 50, mouth_y + 50), (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Silent', (mouth_x - 30, mouth_y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.rectangle(frame, (mouth_x - 50, mouth_y - 50),
                              (mouth_x + 50, mouth_y + 50), (0, 0, 255), 2)

    # Show result
    cv2.imshow('Real-time Speaker Detection', frame)

    # Quit on 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
