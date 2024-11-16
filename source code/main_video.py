import cv2
import face_recognition
import numpy as np
from simple_facerec import SimpleFacerec
from attendance import mark_attendance
from deepface import DeepFace

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

frame_resizing = 0.25
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_known_faces(frame_in):
    small_frame = cv2.resize(frame_in, (0, 0), fx=frame_resizing, fy=frame_resizing)
    # Find all the faces and face encodings in the current frame of video
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations_in = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations_in)
    known_face_encodings = np.load('ImageEncoding.npy')
    image_names = open("ImageNames.txt", "r")
    known_face_names = []
    for known_face_names_var in image_names:
        known_face_names.append(known_face_names_var)
    face_names_in = []

    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name_in = "Unknown "

        # # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]
        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name_in = known_face_names[best_match_index]
            mark_attendance(name_in[:-1])
        face_names_in.append(name_in[:-1])

    # Convert to numpy array to adjust coordinates with frame resizing quickly
    face_locations_in = np.array(face_locations_in)
    face_locations_in = face_locations_in / frame_resizing
    return face_locations_in.astype(int), face_names_in


# def mark_attendance(name_in):


# Load Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)
# cap = cv2.resize(cap, (1920, 1080))

while True:
    ret, frame = cap.read()
    # emotion_detect(ret, frame)
    # Detect Faces
    # faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    face_locations, face_names = detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        face_roi = frame[y1:y2, x1:x2]
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        emot = result[0]['dominant_emotion']
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.putText(frame, emot, (x1, y1 + 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 0, 0), 4)

        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

    cv2.imshow("FaceDetect", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
