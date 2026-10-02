import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math

# CONFIG
WINDOW_SIZE = 18

# Rule 1: right arm motion
SHOULDER_STILL_THRESHOLD = 0.04
ELBOW_MOVE_THRESHOLD = 0.03
WRIST_MOVE_THRESHOLD = 0.05
ANGLE_CHANGE_THRESHOLD = 10
DIRECTION_CHANGE_THRESHOLD = 1

# Rule 2: right hand near hip or thigh
HIP_DISTANCE_THRESHOLD = 0.12
THIGH_DISTANCE_THRESHOLD = 0.10

# MEDIAPIPE SETUP
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# CAMERA
cap = cv2.VideoCapture(0)

# HISTORY
shoulder_y_history = deque(maxlen=WINDOW_SIZE)

elbow_y_history = deque(maxlen=WINDOW_SIZE)
elbow_x_history = deque(maxlen=WINDOW_SIZE)

wrist_y_history = deque(maxlen=WINDOW_SIZE)
wrist_x_history = deque(maxlen=WINDOW_SIZE)

elbow_angle_history = deque(maxlen=WINDOW_SIZE)


def motion_range(values):
    if len(values) < 4:
        return 0.0
    arr = np.array(values, dtype=float)
    return float(np.max(arr) - np.min(arr))


def count_direction_changes(values, small_noise=0.0015):
    if len(values) < 5:
        return 0

    arr = np.array(values, dtype=float)
    smooth = np.convolve(arr, np.ones(3) / 3, mode='valid')
    diffs = np.diff(smooth)
    diffs[np.abs(diffs) < small_noise] = 0

    changes = 0
    prev = 0
    for d in diffs:
        curr = 1 if d > 0 else (-1 if d < 0 else 0)
        if curr != 0:
            if prev != 0 and curr != prev:
                changes += 1
            prev = curr
    return changes


def angle_abc(a, b, c):
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def dist2d(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def point_to_segment_distance(p, a, b):
    px, py = p
    ax, ay = a
    bx, by = b

    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq == 0:
        return dist2d(p, a)

    t = (apx * abx + apy * aby) / ab_len_sq
    t = max(0.0, min(1.0, t))

    closest_x = ax + t * abx
    closest_y = ay + t * aby

    return dist2d(p, (closest_x, closest_y))


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    status = "Normal"
    color = (0, 255, 0)
    reason = "-"

    if result.pose_landmarks:
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lms = result.pose_landmarks.landmark

        rs = lms[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        re = lms[mp_pose.PoseLandmark.RIGHT_ELBOW]
        rw = lms[mp_pose.PoseLandmark.RIGHT_WRIST]
        rh = lms[mp_pose.PoseLandmark.RIGHT_HIP]
        rk = lms[mp_pose.PoseLandmark.RIGHT_KNEE]

        visible = (
            rs.visibility > 0.5 and
            re.visibility > 0.5 and
            rw.visibility > 0.5 and
            rh.visibility > 0.5 and
            rk.visibility > 0.5
        )

        if visible:
            shoulder = (rs.x, rs.y)
            elbow = (re.x, re.y)
            wrist = (rw.x, rw.y)
            hip = (rh.x, rh.y)
            knee = (rk.x, rk.y)

            # save history
            shoulder_y_history.append(shoulder[1])

            elbow_y_history.append(elbow[1])
            elbow_x_history.append(elbow[0])

            wrist_y_history.append(wrist[1])
            wrist_x_history.append(wrist[0])

            elbow_angle = angle_abc(shoulder, elbow, wrist)
            elbow_angle_history.append(elbow_angle)

            # motion features
            shoulder_still = motion_range(shoulder_y_history)

            elbow_move_y = motion_range(elbow_y_history)
            elbow_move_x = motion_range(elbow_x_history)
            elbow_total_move = max(elbow_move_x, elbow_move_y)

            wrist_move_y = motion_range(wrist_y_history)
            wrist_move_x = motion_range(wrist_x_history)
            wrist_total_move = max(wrist_move_x, wrist_move_y)

            elbow_angle_change = motion_range(elbow_angle_history)

            elbow_dir_changes = count_direction_changes(elbow_y_history)
            wrist_dir_changes = count_direction_changes(wrist_y_history)

            # rule motion more senstive
            motion_alert = (
                shoulder_still <= SHOULDER_STILL_THRESHOLD and
                (
                    elbow_total_move >= ELBOW_MOVE_THRESHOLD or
                    wrist_total_move >= WRIST_MOVE_THRESHOLD
                ) and
                elbow_angle_change >= ANGLE_CHANGE_THRESHOLD and
                (
                    elbow_dir_changes >= DIRECTION_CHANGE_THRESHOLD or
                    wrist_dir_changes >= DIRECTION_CHANGE_THRESHOLD
                )
            )

            # rule near hip/thigh
            wrist_to_hip = dist2d(wrist, hip)
            wrist_to_thigh = point_to_segment_distance(wrist, hip, knee)

            hand_near_hip_or_thigh = (
                wrist_to_hip <= HIP_DISTANCE_THRESHOLD or
                wrist_to_thigh <= THIGH_DISTANCE_THRESHOLD
            )

            # priority
            if motion_alert:
                status = "DUNG LO NGAY CHO TAO "
                color = (0, 0, 255)
                reason = "shoulder still, elbow/wrist moving"
            elif hand_near_hip_or_thigh:
                status = "PHAT HIEN CHUAN BI LAM QUA LO "
                color = (0, 140, 255)
                reason = "right hand close to hip or thigh"
            else:
                status = "Normal"
                color = (0, 255, 0)
                reason = "-"

            # draw 
            rs_px = (int(shoulder[0] * w), int(shoulder[1] * h))
            re_px = (int(elbow[0] * w), int(elbow[1] * h))
            rw_px = (int(wrist[0] * w), int(wrist[1] * h))
            rh_px = (int(hip[0] * w), int(hip[1] * h))
            rk_px = (int(knee[0] * w), int(knee[1] * h))

            cv2.circle(frame, rs_px, 8, (255, 0, 0), -1)
            cv2.circle(frame, re_px, 8, (0, 0, 255), -1)
            cv2.circle(frame, rw_px, 8, (0, 255, 255), -1)
            cv2.circle(frame, rh_px, 8, (255, 0, 255), -1)
            cv2.circle(frame, rk_px, 8, (0, 255, 0), -1)

            cv2.line(frame, rs_px, re_px, (255, 255, 0), 2)
            cv2.line(frame, re_px, rw_px, (255, 255, 0), 2)
            cv2.line(frame, rh_px, rk_px, (0, 200, 255), 3)

            # debug 
            cv2.putText(frame, f"Shoulder still: {shoulder_still:.3f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"Elbow move: {elbow_total_move:.3f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"Wrist move: {wrist_total_move:.3f}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"Angle change: {elbow_angle_change:.1f}", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"Elbow dir: {elbow_dir_changes}", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"Wrist dir: {wrist_dir_changes}", (20, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"Wrist-Hip: {wrist_to_hip:.3f}", (20, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"Wrist-Thigh: {wrist_to_thigh:.3f}", (20, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"Reason: {reason}", (20, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        else:
            cv2.putText(frame, "Right landmarks not clear", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

    else:
        cv2.putText(frame, "No pose detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)

    cv2.rectangle(frame, (0, h - 60), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, status, (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow("Right Arm Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
