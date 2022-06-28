import cv2
from math import hypot


def applyMask(frame, landmarks, mask):

    for f in range(len(landmarks)):
        face = landmarks[f][0]
        # face = list(face)
        # Nose 30 31 35
        top_nose = (face[29][0], face[29][1])
        center_nose = (face[30][0], face[30][1])
        left_nose = (face[31][0], face[31][1])
        right_nose = (face[35][0], face[35][1])

        nose_width = int(
            hypot(left_nose[0] - right_nose[0], left_nose[1] - right_nose[1]) * 1.7
        )
        nose_height = int(nose_width * 0.58)

        # New nose position
        top_left = (
            int(center_nose[0] - nose_width / 2),
            int(center_nose[1] - nose_height / 2),
        )
        bottom_right = (
            int(center_nose[0] + nose_width / 2),
            int(center_nose[1] + nose_height / 2),
        )

        # Adding the new nose
        nose_pig = cv2.resize(mask, (nose_width, nose_height))
        nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
        _, nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)

        nose_area = frame[
            top_left[1] : top_left[1] + nose_height,
            top_left[0] : top_left[0] + nose_width,
        ]
        nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
        final_nose = cv2.add(nose_area_no_nose, nose_pig)

        frame[
            top_left[1] : top_left[1] + nose_height,
            top_left[0] : top_left[0] + nose_width,
        ] = final_nose

    return frame
