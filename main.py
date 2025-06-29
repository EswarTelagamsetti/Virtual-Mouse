import cv2
import mediapipe as mp
import pyautogui
import time

cap = cv2.VideoCapture(0)

hand_detector = mp.solutions.hands.Hands(max_num_hands=1)
drawing_utils = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)

    if output.multi_hand_landmarks:
        for hand in output.multi_hand_landmarks:
            drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)

            landmarks = hand.landmark

            index_tip = landmarks[8]
            thumb_tip = landmarks[4]

            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            screen_x = int(index_tip.x * screen_w)
            screen_y = int(index_tip.y * screen_h)
            pyautogui.moveTo(screen_x, screen_y)

            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            thumb_x = int(thumb_tip.x * w)
            thumb_y = int(thumb_tip.y * h)

            distance = ((thumb_x - x) ** 2 + (thumb_y - y) ** 2) ** 0.5

            if distance < 30:
                pyautogui.click()
                cv2.circle(frame, (x, y), 15, (0, 0, 255), -1) 
                time.sleep(0.3)  

    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
