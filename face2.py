import cv2
import face_recognition
import numpy as np
import dlib
import os
import csv
from datetime import datetime
from scipy.spatial import distance

# Load known faces
KNOWN_FACES_DIR = "known_faces"
CSV_FILE = "attendance.csv"
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

known_encodings = []
known_names = []

# Ensure the directory exists
if not os.path.exists(KNOWN_FACES_DIR):
    print(f"Error: Directory '{KNOWN_FACES_DIR}' not found.")
    exit()

# Load known faces
for file in os.listdir(KNOWN_FACES_DIR):
    image_path = os.path.join(KNOWN_FACES_DIR, file)
    image = face_recognition.load_image_file(image_path)
    
    encodings = face_recognition.face_encodings(image)
    if encodings:
        encoding = encodings[0]
        known_encodings.append(encoding)
        known_names.append(os.path.splitext(file)[0])

print("✅ Face data loaded successfully!")

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Load dlib face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

# Track attendance
attendance_list = set()

# Ensure CSV file has headers
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Timestamp"])

# Eye landmark indices
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

def eye_aspect_ratio(eye):
    """Calculate the eye aspect ratio (EAR)."""
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

BLINK_THRESHOLD = 0.2  # EAR threshold for a blink
BLINK_CONSEC_FRAMES = 2  # Number of frames required for a blink
blink_counter = {}
blink_detected = {}

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("❌ Error: Could not access webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Extract eye coordinates
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Get face encoding
        face_encoding = face_recognition.face_encodings(frame, [(face.top(), face.right(), face.bottom(), face.left())])
        if not face_encoding:
            continue  # Skip if encoding fails
        
        face_encoding = face_encoding[0]
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            match_index = np.argmin(face_recognition.face_distance(known_encodings, face_encoding))
            name = known_names[match_index]

            # Initialize blink tracking for this person
            if name not in blink_counter:
                blink_counter[name] = 0
                blink_detected[name] = False

            # Detect blinking
            if avg_ear < BLINK_THRESHOLD:
                blink_counter[name] += 1
            else:
                if blink_counter[name] >= BLINK_CONSEC_FRAMES and not blink_detected[name]:
                    # Blink detected
                    blink_detected[name] = True
                    if name not in attendance_list:
                        attendance_list.add(name)
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        # Save attendance to CSV
                        with open(CSV_FILE, mode="a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow([name, timestamp])

                        print(f"✅ Attendance marked for: {name} at {timestamp}")

                blink_counter[name] = 0  # Reset counter after blink

        # Draw bounding box
        top, right, bottom, left = face.top(), face.right(), face.bottom(), face.left()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Show video feed
    cv2.imshow("Facial Attendance System (Blink to Mark)", frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
