import cv2 as cv
import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import HandDetectionModule as hdm


# finding the distance between 2 points
def distance(point1, point2):
    (x1, y1), (x2, y2) = point1, point2
    return ((x1-x2)**2 + (y1-y2)**2)**0.5

if __name__ == "__main__":
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = interface.QueryInterface(IAudioEndpointVolume)
    # print(volume.GetMasterVolumeLevel())
    # print(volume.GetVolumeRange())
    detect = hdm.HandDetector()

    video = cv.VideoCapture(0)

    while True:
        isTrue, frame = video.read()
        lm = detect.landmarks(frame, True)
        if len(lm) > 0:
            cv.circle(frame, lm[4], radius=10, color=(0,255,0), thickness=cv.FILLED)
            cv.circle(frame, lm[8], radius=10, color=(0,255,0), thickness=cv.FILLED)

            (x1,y1),(x2,y2)  = lm[4], lm[8]
            center = (int((x1+x2)/2), int((y1+y2)/2))

            cv.line(frame, lm[4], lm[8], color=(255,0,0), thickness=4)
            cv.circle(frame, center, radius=10, color=(0,255,0), thickness=cv.FILLED)

            # setting the volume
            fingerdRange = [30, 235]
            nvol = np.interp(distance(lm[4], lm[8]), fingerdRange, volume.GetVolumeRange()[:2])
            volume.SetMasterVolumeLevel(nvol,None)

        cv.imshow("Camera", frame)
        cv.waitKey(1)


    video.release()
    cv.destroyAllWindows()


