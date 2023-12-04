import cv2
import numpy as np
import time
import mediapipe
import dlib
import face_recognition as fr
import glob
import imutils

known_face_encodings = []
all_names = ['me','elon musk','mark zuckerberg']
cap = cv2.VideoCapture(0)



images = glob.glob('faces/*')
for fname in images:
    image = fr.load_image_file(fname)
    image_encoding = fr.face_encodings(image)[0]
    known_face_encodings.append(image_encoding)

print(images)

while True:
    ret,image = cap.read()
    camera_encoding = fr.face_encodings(image)
    camera_location = fr.face_locations(image)

    for (top, right, bottom, left),element_encoding in zip(camera_location,camera_encoding):
        result = fr.compare_faces(known_face_encodings,element_encoding)
        print(result)
        if(True in result):
            index = result.index(True)
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(image, all_names[index], (left + 10, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 0, 255),
                            thickness=2)
        cv2.imshow('image',image)
    if cv2.waitKey(1) & 0xFF == 27:
        break






