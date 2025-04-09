import cv2
import mediapipe as mp
import time
import math

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList

    def fingersUp(self):
        fingers = []
        # Tip ids for fingers: [thumb, index, middle, ring, pinky]
        tipIds = [4, 8, 12, 16, 20]

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[0]
            lmList = []
            for id, lm in enumerate(myHand.landmark):
                lmList.append(lm)

            # Thumb
            fingers.append(1 if lmList[tipIds[0]].x < lmList[tipIds[0] - 1].x else 0)

            # Other 4 fingers
            for id in range(1, 5):
                fingers.append(1 if lmList[tipIds[id]].y < lmList[tipIds[id] - 2].y else 0)

        return fingers

    def findDistance(self, p1, p2, img=None, draw=True):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        length = math.hypot(x2 - x1, y2 - y1)

        if draw and img is not None:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.circle(img, (x1, y1), 5, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

        return length, img, [x1, y1, x2, y2, cx, cy]
