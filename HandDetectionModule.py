import cv2 as cv
import mediapipe as mp

class HandDetector:

    def __init__(self,
                 static_image_mode=False,
                 max_num_hands=2,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.mpdraw = mp.solutions.drawing_utils
        self.mphands = mp.solutions.hands

        stImgMode = static_image_mode
        noHands = max_num_hands
        model_complexity = model_complexity
        minDetectionCon = min_detection_confidence
        minTrackingCon = min_tracking_confidence
        self.lmCoordinates = []
        self.hands = self.mphands.Hands(stImgMode, noHands,
                                   model_complexity, minDetectionCon, minTrackingCon)

    def landmarks(self, img, draw=False):
        lst = []  # Initialize lst to an empty list
        rgbImg = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        result = self.hands.process(rgbImg)
        lmarks = result.multi_hand_landmarks
        if lmarks:
            for lmark in lmarks:
                for id, lm in enumerate(lmark.landmark):
                    h, w, c = img.shape  # Get frame dimensions (fix variable from frame to img)
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmCoordinates.append((cx, cy))
                lst = self.lmCoordinates.copy()
                self.lmCoordinates.clear()
                if draw:
                    self.mpdraw.draw_landmarks(img, lmark, self.mphands.HAND_CONNECTIONS)
        return lst


def beutifyAll( img, lms, radius=3, color=(0, 255, 0), thickness=-1):
    for point in lms:
        cv.circle(img, point, radius=radius, color=color, thickness=thickness)


if __name__ == "__main__":
    hands = HandDetector()

    video = cv.VideoCapture(0)

    while True:
        isTrue, frame = video.read()
        lm = hands.landmarks(frame, True)
        if len(lm) > 0:
            beutifyPoint(frame, lm[4], radius=10)
            beutifyPoint(frame, lm[8], radius=10)
        # beutifyAll(frame, lm)


        cv.imshow("Camera", frame)
        cv.waitKey(1)


    video.release()
    cv.destroyAllWindows()
