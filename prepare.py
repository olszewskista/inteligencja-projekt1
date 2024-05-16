import cv2
import mediapipe as mp
import numpy as np
import os

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

dirs = ["down", "up", "right", "left", "front"]

translateDirection = {"up": 0, "down": 1, "left": 2, "right": 3, "front": 4}

labels = []
points = []

# Create a face detector instance with the image mode:
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path="./face_detection.tflite"),
    running_mode=VisionRunningMode.IMAGE,
)
with FaceDetector.create_from_options(options) as detector:
    for directory in dirs:
        for photo in os.listdir(f"./dataset/{directory}"):
            # Load the input image from an image file.
            mp_image = mp.Image.create_from_file(f"./dataset/{directory}/{photo}")

            # Perform face detection on the provided single image.
            face_detector_result = detector.detect(mp_image)

            # Convert the mediapipe image to numpy array for OpenCV
            image = cv2.cvtColor(np.copy(mp_image.numpy_view()), cv2.COLOR_BGR2RGB)

            # Draw facial keypoints
            if face_detector_result.detections:
                for detection in face_detector_result.detections:
                    keypoints = detection.keypoints
                    # Calculate distances
                    d1 = abs(keypoints[2].x - keypoints[5].x)
                    d2 = abs(keypoints[2].x - keypoints[4].x)
                    d3 = abs(keypoints[0].y - keypoints[4].y)
                    d4 = abs(keypoints[1].y - keypoints[5].y)
                    d5 = abs(keypoints[0].y - keypoints[2].y)
                    d6 = abs(keypoints[1].y - keypoints[2].y)
                    points.append([d1, d2, d3, d4, d5, d6])
                    labels.append(translateDirection[directory])
                    # print([d1, d2, d3, d4, d5, d6], directory)

                    # Draw points on image
                    # for keypoint in keypoints:
                    #     x, y = int(keypoint.x * image.shape[1]), int(keypoint.y * image.shape[0])
                    #     # print(image.shape)
                    #     cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            # Display the image with keypoints
            # cv2.imshow("Face Keypoints Detection Result", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

points = np.asarray(points)
labels = np.asarray(labels)

np.save("keypoints.npy", points)
np.save("labels.npy", labels)