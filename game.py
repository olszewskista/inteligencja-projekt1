import cv2
import time
import tkinter as tk
from PIL import Image, ImageTk

import numpy as np
from mediapipe.python.solutions import face_detection
from mediapipe.python.solutions import drawing_utils
from pynput.keyboard import Key, Controller
from keras.api.models import load_model

keyboard = Controller()

model = load_model("./test.model.keras")

transateDirection = {0: "up", 1: "down", 2: "left", 3: "right", 4: "front"}

cap = cv2.VideoCapture(0)

faceDetection = face_detection.FaceDetection()
pTime = 0

def main():
    global pTime
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(img)

    if results.detections:

        for detection in results.detections:
            drawing_utils.draw_detection(img, detection)

            keypoints = detection.location_data.relative_keypoints

            l1 = abs(keypoints[2].x - keypoints[5].x)
            l2 = abs(keypoints[2].x - keypoints[4].x)
            l3 = abs(keypoints[0].y - keypoints[4].y)
            l4 = abs(keypoints[1].y - keypoints[5].y)
            l5 = abs(keypoints[0].y - keypoints[2].y)
            l6 = abs(keypoints[1].y - keypoints[2].y)

            points = np.asarray([l1, l2, l3, l4, l5, l6])
            points = points.reshape(1, 6)

            result = model.predict(points)
            result = result.argmax()

            if result == 0:
                keyboard.press(Key.up)
                keyboard.release(Key.down)
                keyboard.release(Key.right)
                keyboard.release(Key.left)
                print("up")
            elif result == 1:
                keyboard.press(Key.down)
                keyboard.release(Key.up)
                keyboard.release(Key.right)
                keyboard.release(Key.left)
                print("down")
            elif result == 2:
                keyboard.press(Key.left)
                keyboard.release(Key.down)
                keyboard.release(Key.right)
                keyboard.release(Key.up)
                print("left")
            elif result == 3:
                keyboard.press(Key.right)
                keyboard.release(Key.down)
                keyboard.release(Key.up)
                keyboard.release(Key.left)
                print("right")
            else:
                keyboard.release(Key.up)
                keyboard.release(Key.down)
                keyboard.release(Key.right)
                keyboard.release(Key.left)
                print("front")

            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = (
                int(bboxC.xmin * iw),
                int(bboxC.ymin * ih),
                int(bboxC.width * iw),
                int(bboxC.height * ih),
            )
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(
                img,
                f"{int(detection.score[0]*100)}%",
                (bbox[0], bbox[1] - 20),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                (0, 255, 0),
                2,
            )
            # Display result
            cv2.putText(
                img,
                f"{transateDirection[result]}",
                (20, 410),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                (255, 0, 0),
                3,
            )
    
    # Display FPS
    # cTime = time.time()
    # fps = 1 / (cTime - pTime)
    # pTime = cTime
    # cv2.putText(
    #     img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3
    # )


    # Showing image
    img2 = Image.fromarray(img)
    img3 = ImageTk.PhotoImage(image=img2)
    label_widget.img3 = img3
    label_widget.configure(image=img3)

    m.after(10, main)


# GUI
m = tk.Tk()
m.geometry("600x500")
m.title("Looking Direction Steering")
start_button = tk.Button(m, text="Start", width=25, command=main)
start_button.pack()
stop_button = tk.Button(m, text="Exit", width=25, command=exit)
stop_button.pack()
label_widget = tk.Label(m)
label_widget.pack(fill="both", expand=True)
m.mainloop()
