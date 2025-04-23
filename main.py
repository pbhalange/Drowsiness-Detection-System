import cv2
import dlib
import numpy as np
from math import sqrt
from imutils import face_utils
import time
import pygame


# Constants
MIN_EAR = 0.2
MIN_DROWSY_EAR = 0.3
MAX_DROWSY_FRAMES = 35

LEFT_EYE_INDICES = list(range(36, 42))
RIGHT_EYE_INDICES = list(range(42, 48))


pygame.mixer.init()


def distance(pt1, pt2):
    """Calculate Euclidean distance between two points."""
    return sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def eye_aspect_ratio(eye_indices, landmarks):
    """Compute the Eye Aspect Ratio (EAR)."""
    points = [landmarks.part(i) for i in eye_indices]
    left = (points[0].x, points[0].y)
    right = (points[3].x, points[3].y)
    top = distance((points[1].x, points[1].y), (points[5].x, points[5].y))
    bottom = distance((points[2].x, points[2].y), (points[4].x, points[4].y))
    return (top + bottom) / (2.0 * distance(left, right))


def draw_eye_contours(frame, landmarks):
    """Draw convex hull contours around eyes."""
    np_landmarks = face_utils.shape_to_np(landmarks)
    left_eye_hull = cv2.convexHull(np_landmarks[36:42])
    right_eye_hull = cv2.convexHull(np_landmarks[42:48])
    cv2.drawContours(frame, [left_eye_hull], -1, (0, 0, 255), 2)
    cv2.drawContours(frame, [right_eye_hull], -1, (0, 0, 255), 2)


def show_text(frame, text, position, color, scale=1.2, thickness=2):
    """Render text on the frame."""
    cv2.putText(
        frame, text, position, cv2.FONT_HERSHEY_DUPLEX,
        scale, color, thickness, cv2.LINE_AA
    )

# def play_alert():
    
#     pygame.mixer.music.load("Alert/alert.wav")  
#     pygame.mixer.music.play()

def main():
    cap = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./Dataset/shape_predictor_68_face_landmarks.dat")

    drowsy_start_time = None
    alert_playing = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces, _, _ = detector.run(frame, 0, 0.0)

        for face in faces:
            landmarks = predictor(frame, face)
            left_ear = eye_aspect_ratio(LEFT_EYE_INDICES, landmarks)
            right_ear = eye_aspect_ratio(RIGHT_EYE_INDICES, landmarks)
            avg_ear = (left_ear + right_ear) / 2.0

            draw_eye_contours(frame, landmarks)

            if avg_ear < MIN_EAR:
                show_text(frame, "You are blinking.", (10, 100), (0, 0, 255), scale=1)

            if avg_ear < MIN_DROWSY_EAR:
                if drowsy_start_time is None:
                    drowsy_start_time = time.time()
                else:
                    elapsed = time.time() - drowsy_start_time
                    if elapsed >= 10:  # Eye closed for 10+ seconds
                        show_text(frame, "You are drowsy!", (10, 400), (255, 0, 0), scale=1, thickness=2)
                        if not alert_playing:
                            pygame.mixer.music.load("Alert/alert.wav")
                            pygame.mixer.music.play(-1)  # Loop indefinitely
                            alert_playing = True
            else:
                drowsy_start_time = None
                if alert_playing:
                    pygame.mixer.music.stop()
                    alert_playing = False

            show_text(frame, f"EAR: {avg_ear:.2f}", (10, 60), (255, 255, 0), scale=0.7, thickness=1)

        cv2.imshow("Drowsiness Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty("Drowsiness Detection", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    pygame.mixer.music.stop()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
