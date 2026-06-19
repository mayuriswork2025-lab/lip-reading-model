import os

import cv2
import numpy as np

LIP_LANDMARKS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
]

_FACE_CASCADE = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
)

_mp_face_mesh = None


def _get_face_mesh():
    global _mp_face_mesh
    if _mp_face_mesh is None:
        import mediapipe as mp

        _mp_face_mesh = mp.solutions.face_mesh
    return _mp_face_mesh


def _crop_mouth_haar(frame, output_size=(96, 96)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    x1 = max(0, x + int(w * 0.2))
    x2 = min(frame.shape[1], x + int(w * 0.8))
    y1 = max(0, y + int(h * 0.55))
    y2 = min(frame.shape[0], y + int(h * 0.95))
    mouth = frame[y1:y2, x1:x2]
    if mouth.size == 0:
        return None
    return cv2.resize(mouth, output_size)


def crop_mouth(frame, padding=0.3, output_size=(96, 96)):
    return crop_mouth_sequence([frame], padding=padding, output_size=output_size)[0]


def crop_mouth_sequence(frames, padding=0.3, output_size=(96, 96)):
    if not frames:
        return []

    crops = []
    try:
        mp_face_mesh = _get_face_mesh()
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:
            for frame in frames:
                crops.append(_crop_mouth_frame(frame, face_mesh, padding, output_size))
    except Exception:
        crops = [_crop_mouth_haar(frame, output_size=output_size) for frame in frames]

    return crops


def _crop_mouth_frame(frame, face_mesh, padding, output_size):
    h, w = frame.shape[:2]
    try:
        result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            return _crop_mouth_haar(frame, output_size=output_size)

        landmarks = result.multi_face_landmarks[0].landmark
        xs = [landmarks[i].x * w for i in LIP_LANDMARKS]
        ys = [landmarks[i].y * h for i in LIP_LANDMARKS]
        pad_x = padding * (max(xs) - min(xs))
        pad_y = padding * (max(ys) - min(ys))
        x1 = max(0, int(min(xs) - pad_x))
        y1 = max(0, int(min(ys) - pad_y))
        x2 = min(w, int(max(xs) + pad_x))
        y2 = min(h, int(max(ys) + pad_y))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return _crop_mouth_haar(frame, output_size=output_size)
        return cv2.resize(crop, output_size)
    except Exception:
        return _crop_mouth_haar(frame, output_size=output_size)
