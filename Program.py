import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time

class HandDetector:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.8, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.lmList = []

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append((id, cx, cy))
        return self.lmList

    def fingersUp(self):
        fingers = []
        tipIds = [4, 8, 12, 16, 20]
        if self.lmList:
            # Thumb
            fingers.append(1 if self.lmList[tipIds[0]][1] > self.lmList[tipIds[0] - 1][1] else 0)
            # 4 Fingers
            for id in range(1, 5):
                fingers.append(1 if self.lmList[tipIds[id]][2] < self.lmList[tipIds[id] - 2][2] else 0)
        return fingers

    def findDistance(self, p1, p2, img=None):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        return length

# Initialize
wCam, hCam = 640, 480
frameR = 100
smoothening = 5
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = HandDetector(maxHands=1)
wScr, hScr = pyautogui.size()

pTime = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    if lmList:
        x1, y1 = lmList[8][1:]  # Index
        x2, y2 = lmList[12][1:] # Middle

        fingers = detector.fingersUp()

        # Moving Mode: Only Index Finger
        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            pyautogui.moveTo(wScr - clocX, clocY)
            plocX, plocY = clocX, clocY

        # Clicking Mode: Index + Middle Fingers
        if fingers[1] == 1 and fingers[2] == 1:
            length = detector.findDistance(8, 12, img)
            if length < 40:
                pyautogui.click()

    # FPS Counter
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Mouse", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to quit
        break
