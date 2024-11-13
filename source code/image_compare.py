import dlib
import cv2
import glob
import shutil

def compare_faces(image1_path, image2_path):
    # Load the images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Detect faces in the images
    faces1 = face_detector(gray1)
    faces2 = face_detector(gray2)

    # Check if any faces are detected
    if len(faces1) == 0 or len(faces2) == 0:
        return "No faces detected in one or both images"

    # Create a face recognition model
    face_recognizer = dlib.face_recognition_model_v1("shape_predictor_68_face_landmarks.dat")

    # Compute face descriptors for the first face in each image
    face_descriptor1 = face_recognizer.compute_face_descriptor(image1, faces1[0])
    face_descriptor2 = face_recognizer.compute_face_descriptor(image2, faces2[0])

    # Calculate the Euclidean distance between the face descriptors
    distance = dlib.distance(face_descriptor1, face_descriptor2)

    # Compare the distance against a threshold (you can adjust this value)
    threshold = 0.6
    if distance < threshold:
        return "Faces are a match"
    else:
        return "Faces are not a match"

# import glob


def get_image_files(folder_path):
    image_files = glob.glob(folder_path + "/*.jpg")  # Modify the file extension if needed
    # image_files.extend(glob.glob(folder_path + "/*.jpeg"))
    # image_files.extend(glob.glob(folder_path + "/*.png"))
    # image_files.extend(glob.glob(folder_path + "/*.gif"))
    # Add more file extensions as required

    return image_files

# Provide the folder path
folder_path = r"E:\00. Wedding\810 haldi"  # Using a raw string


def move_file(source_path, destination_path):
    shutil.move(source_path, destination_path)

# Provide the paths of the source file and the destination folder
# source_path = "/path/to/source/file./jpg"
destination_path: str = r"D:\guddiDidiPics"

# Move the file
# move_file(folder_path, destination_path)
# main

image_files = get_image_files(folder_path)

for image_file in image_files:
    image1_path = r"D:\Archmage\Sharingan\source code\A7403022.JPG"
    result = compare_faces(image1_path, image_file)
    if result:
        print(image_file)
        # move_file(folder_path, destination_path)