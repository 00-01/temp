from time import time
import concurrent.futures as cf
import cv2
import face_recognition as fr
import numpy as np
from db import access_db
# import concurrent.futures as cf

_, _, col, error_col = access_db()
n = list(col.find({}))
names = [i['name'] for i in n]
id = [j['id'] for j in n]
cap = cv2.VideoCapture(0)
distance = 0.35

def run():
    while True:
        s = time()
        ret, frame = cap.read()
        # frame1 = frame[:, :, ::-1]
        # frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        face_locations = fr.face_locations(frame)
        face_encodings = fr.face_encodings(frame, face_locations)
        for (y, x1, y1, x), face_encoding in zip(face_locations, face_encodings):
            name = "- - -"
            face_distances = fr.face_distance(id, face_encoding)  # calculate the distances from input to matches
            best_match_index = np.argmin(face_distances)  # select the lowest distance index
            if face_distances[best_match_index] < distance:
                name = names[best_match_index]
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
            cv2.putText(frame, name, (x - 1, y1 + 24), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 1)
            print('name : ', name)
            # calculate fps
            seconds = time() - s
            fps = 1 / seconds
            fps = ("%.2f" % fps)
            print(f"fps : {fps}", '\n')
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == 27:
            break

run()