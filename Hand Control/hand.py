import cv2
import mediapipe as mp


class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        # इनिशियलाइज़ेशन
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # MediaPipe के हाथ डिटेक्शन मॉडल को लोड करना
        self.mpHands = mp.solutions.hands # pyright: ignore[reportAttributeAccessIssue]
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

        # ड्रॉइंग यूटिलिटी (जोड़ों को जोड़ने के लिए)
        self.mpDraw = mp.solutions.drawing_utils # pyright: ignore[reportAttributeAccessIssue]

    def find_hands(self, img, draw=True):
        # BGR को RGB में कन्वर्ट करना (क्योंकि MediaPipe RGB यूज़ करता है)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # हाथों को प्रोसेस करना
        self.results = self.hands.process(imgRGB)

        # अगर हाथ मिले हैं
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # हाथ के जोड़ और कनेक्शन ड्रा करना
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def find_position(self, img, handNo=0, draw=True):
        lmList = []

        # अगर हाथ मिले हैं
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, c = img.shape

            for id, lm in enumerate(myHand.landmark):
                # x और y को पिक्सल वैल्यू में बदलना
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

                if draw:
                    # हर जोड़ पर सर्कल ड्रा करना
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)

        return lmList


# =======================
# 📷 Main Program (Webcam)
# =======================

def main():
    cap = cv2.VideoCapture(0)  # 0 = डिफ़ॉल्ट कैमरा
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lmList = detector.find_position(img)

        if lmList:
            print(lmList[4])  # उंगली ID 4 (अंगूठे की टिप) की पोजिशन प्रिंट करना

        cv2.imshow("Hand Tracking", img)

        # 'q' दबाने पर बंद कर दें
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()