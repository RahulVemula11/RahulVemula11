import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
from datetime import datetime

# Path to the folder containing images of known faces
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"

# Load known faces
known_encodings = []
known_names = []

for file in os.listdir(KNOWN_FACES_DIR):
    image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{file}")
    encoding = face_recognition.face_encodings(image)[0]
    known_encodings.append(encoding)
    known_names.append(os.path.splitext(file)[0])  # Get name from filename

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Function to mark attendance
def mark_attendance(name):
    try:
        df = pd.read_csv(ATTENDANCE_FILE)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Name", "Time"])

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if name not in df["Name"].values:
        new_entry = pd.DataFrame({"Name": [name], "Time": [now]})
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)

# Start video processing
while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # Resize for faster processing
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_names[best_match_index]
            mark_attendance(name)  # Save attendance

        # Draw box around face
        top, right, bottom, left = [v * 4 for v in face_location]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Show the video frame
    cv2.imshow("Facial Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
