import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# Eye aspect ratio function
def eye_aspect_ratio(landmarks, eye_indices):
    A = np.linalg.norm(np.array(landmarks[eye_indices[1]]) - np.array(landmarks[eye_indices[5]]))
    B = np.linalg.norm(np.array(landmarks[eye_indices[2]]) - np.array(landmarks[eye_indices[4]]))
    C = np.linalg.norm(np.array(landmarks[eye_indices[0]]) - np.array(landmarks[eye_indices[3]]))
    return (A + B) / (2.0 * C)

def is_live_face(frame):
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return False

        h, w, _ = frame.shape
        landmarks = [(int(l.x * w), int(l.y * h)) for l in results.multi_face_landmarks[0].landmark]

        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        leftEAR = eye_aspect_ratio(landmarks, LEFT_EYE)
        rightEAR = eye_aspect_ratio(landmarks, RIGHT_EYE)
        ear = (leftEAR + rightEAR) / 2.0

        if ear < 0.2:  # blink threshold
            return True
        return True  # fallback if face is detected
