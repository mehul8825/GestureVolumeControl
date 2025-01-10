import cv2 as cv
import mediapipe as mp


mpdraw = mp.solutions.drawing_utils

mpface = mp.solutions.face_mesh
face = mpface.FaceMesh()

video = cv.VideoCapture(0)

while True:
    isTrue, frame = video.read()

    rgbImg = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = face.process(rgbImg)
    lmarks = result.multi_face_landmarks
    if lmarks:
        for lmark in lmarks:
            mpdraw.draw_landmarks(frame, lmark,mpface.FACEMESH_TESSELATION )


    cv.imshow("Camera", frame)
    cv.waitKey(1)

