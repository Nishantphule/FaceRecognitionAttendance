'''
pip install --upgrade pip setuptools - (if required)

pip install cmake

Step1 -- Install CMake. Download and install CMake from the official website. Or use pip install cmake.

Step2 -- Install Visual Studio Build Tools. dlib requires a C++ compiler, and on Windows, you can use the Visual Studio Build Tools.
You can download and install them from the official website. During installation, make sure to select the "Desktop development with C++" workload.

Step3 -- Install Boost. dlib depends on the Boost C++ libraries. You can manually install Boost by following these steps:

Download the Boost library from the official website: Boost C++ Libraries. Extract the Boost archive to a location on your machine.
Open a Command Prompt and navigate to the Boost directory. Run the following command to build Boost: bootstrap.bat then run: b2

pip install dlib
pip install opencv-python
pip install face_recognition
pip install numpy

'''

import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

# Load Known faces
harry_image = face_recognition.load_image_file("faces/harry.jpg")
harry_encoding = face_recognition.face_encodings(harry_image)[0]

rohan_image = face_recognition.load_image_file("faces/rohan.jpg")
rohan_encoding = face_recognition.face_encodings(rohan_image)[0]

nishant_image = face_recognition.load_image_file("faces/nishant.jpg")
nishant_encoding = face_recognition.face_encodings(nishant_image)[0]

varad_image = face_recognition.load_image_file("faces/varad.jpg")
varad_encoding = face_recognition.face_encodings(varad_image)[0]

known_face_encodings = [harry_encoding, rohan_encoding, nishant_encoding, varad_encoding]
known_face_names = ["Harry", "Rohan", "Nishant", "Varad"]

# list of expected students
students = known_face_names.copy()

face_locations = []
face_encodings = []

# Get the date and time

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)   #create a csv file after every attendance taken

while True:  # infinite loop to capture all attendace
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognize faces from webcam
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if(matches[best_match_index]):
            name = known_face_names[best_match_index]
        else:
            name = "Unknown Student"

    # #  Add the text if a person is present
    if name in known_face_names:
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerText = (10, 100)
        fontScale = 1.5
        color = (255, 0, 0)
        thickness = 3
        lineType = 2
        cv2.putText(frame, name + " Present", bottomLeftCornerText, font, fontScale, color, thickness, lineType)
    else:
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerText = (10, 100)
        fontScale = 1.5
        color = (255, 0, 0)
        thickness = 3
        lineType = 2
        cv2.putText(frame, name + " Present", bottomLeftCornerText, font, fontScale, color, thickness, lineType)

    if name in students:
        students.remove(name)
        current_time = now.strftime("%H:%M:%S")
        lnwriter.writerow([name, current_time])

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  #exit the while loop and close cam when presssed "q" on keyboard
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()

